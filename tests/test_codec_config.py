import json

import numpy as np
import pytest
import zarr
from helpers import SHAPE, make_array, smooth_field

from skipzfp import SkipZFPCodec

CONFIGS = [
    dict(rate=8.0),
    dict(rate=4.0, block_order=(1, 2, 0)),
    dict(rate=8.0, layers=(2.0, 4.0, 8.0)),
    dict(rate=8.0, layers=(2.0, 8.0), block_order=(2, 0, 1), sub_chunk=(32, 16, 32)),
]


@pytest.mark.parametrize("kw", CONFIGS)
def test_dict_round_trip(kw):
    c = SkipZFPCodec(**kw)
    assert SkipZFPCodec.from_dict(c.to_dict()) == c


def test_default_layers_is_rate():
    assert SkipZFPCodec(rate=6.0).layers == (6.0,)


@pytest.mark.parametrize("kw", CONFIGS)
def test_config_persists_in_zarr_json(tmp_path, kw):
    make_array(tmp_path / "a", smooth_field(), **kw)
    meta = json.loads((tmp_path / "a" / "data" / "zarr.json").read_text())
    assert meta["codecs"] == [SkipZFPCodec(**kw).to_dict()]
    z = zarr.open_array(str(tmp_path / "a"), path="data", mode="r")
    assert z.metadata.codecs == (SkipZFPCodec(**kw),)


@pytest.mark.parametrize(
    "kw, msg",
    [
        (dict(rate=0), "rate must be greater than 0"),
        (dict(rate=-1.0), "rate must be greater than 0"),
        (dict(rate=8.0, layers=(8.0, 2.0)), "layers must be increasing"),
        (dict(rate=8.0, layers=(2.0, 4.0)), "end at rate"),
        (dict(rate=8.0, layers=(2.0, 2.0, 8.0)), "layers must be increasing"),
        (dict(rate=8.0, layers=(2.1, 8.0)), "multiple of 0.125"),
        (dict(block_order=(0, 0, 1)), "permutation"),
        (dict(block_order=(0, 1)), "permutation"),
        (dict(sub_chunk=(30, 16, 32)), "multiples of block_dim"),
        (dict(sub_chunk=(32, 16)), "multiples of block_dim"),
        (dict(sub_chunk=(0, 16, 32)), "multiples of block_dim"),
        (dict(block_dim=8), "block_dim=4"),
    ],
)
def test_invalid_config(kw, msg):
    with pytest.raises(ValueError, match=msg):
        SkipZFPCodec(**kw)


def _create(path, dtype="float32", chunks=SHAPE, shape=SHAPE):
    g = zarr.open_group(str(path), mode="w")
    return g.create_array(
        "data",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        serializer=SkipZFPCodec(rate=8.0),
        compressors=None,
        filters=None,
    )


def test_rejects_float64(tmp_path):
    with pytest.raises(TypeError, match="float32"):
        _create(tmp_path / "a", dtype="float64")


def test_rejects_chunk_not_multiple_of_block(tmp_path):
    with pytest.raises(ValueError, match="divisible by block_dim"):
        _create(tmp_path / "a", chunks=(62, 16, 32), shape=(62, 16, 32))


@pytest.mark.parametrize("n", (70, 72))
def test_rejects_shape_not_multiple_of_chunk(tmp_path, n):
    with pytest.raises(ValueError, match="partial edge chunks"):
        _create(tmp_path / "a", shape=(n, 16, 32))


def test_rejects_chunk_not_multiple_of_sub_chunk(tmp_path):
    g = zarr.open_group(str(tmp_path / "a"), mode="w")
    with pytest.raises(ValueError, match="not divisible by sub_chunk"):
        g.create_array(
            "data",
            shape=SHAPE,
            chunks=SHAPE,
            dtype="float32",
            serializer=SkipZFPCodec(rate=8.0, sub_chunk=(48, 16, 32)),
            compressors=None,
            filters=None,
        )


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_rejects_non_finite_values(tmp_path, bad):
    z = _create(tmp_path / "a")
    a = smooth_field()
    a[3, 4, 5] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        z[:] = a
