"""Arrays with missing values (NaN) under masked=True."""

import numpy as np
import pytest
import zarr
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype

from skipzfp import SkipZFPCodec, codec

from helpers import SHAPE, smooth_field

DATA_SHAPE = (128, 32, 64)  # 2 x 2 x 2 chunks of SHAPE
CONFIGS = [
    dict(rate=8.0),
    dict(rate=4.0, block_order=(1, 2, 0), sub_chunk=(32, 16, 32)),
    dict(rate=8.0, layers=(2.0, 4.0, 8.0)),
    dict(rate=8.0, block_error_bits=4),
]


def land(shape=DATA_SHAPE):
    """A (y, x) land mask: a wavy coast, and one chunk's whole plane on land."""
    y, x = np.indices(shape[1:])
    m = x > 40 + 6 * np.sin(y / 4)
    m[:16, 32:] = True
    return m


def masked_field(shape=DATA_SHAPE, seed=0):
    """A smooth field with NaN over land at every step along axis 0."""
    a = smooth_field(shape, seed)
    a[:, land(shape)] = np.nan
    return a


def create(path, a, fill_value=np.nan, chunks=SHAPE, **kw):
    g = zarr.open_group(str(path), mode="w")
    return g.create_array(
        "data",
        shape=a.shape,
        chunks=chunks,
        dtype="float32",
        serializer=SkipZFPCodec(masked=True, **kw),
        compressors=None,
        filters=None,
        fill_value=fill_value,
    )


def test_dict_round_trip_and_old_arrays_unchanged():
    c = SkipZFPCodec(rate=8.0, masked=True)
    assert c.to_dict()["configuration"]["masked"] is True
    assert SkipZFPCodec.from_dict(c.to_dict()) == c
    assert "masked" not in SkipZFPCodec(rate=8.0).to_dict()["configuration"]


@pytest.mark.parametrize("kw", CONFIGS)
def test_reads_return_nan_where_written(tmp_path, kw):
    a = masked_field()
    z = create(tmp_path / "m", a, **kw)
    z[:] = a
    got = z[:]
    assert np.array_equal(np.isnan(got), np.isnan(a))
    # the valid values decode exactly as the filled field does without a mask
    plain = zarr.open_group(str(tmp_path / "p"), mode="w").create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=SkipZFPCodec(**kw),
        compressors=None,
        filters=None,
    )
    plain[:] = codec.fill_nan(a)
    ok = ~np.isnan(a)
    assert np.array_equal(got[ok], plain[:][ok])
    bound = 0.1 if kw["rate"] >= 8 else 1.0
    assert np.abs(got[ok] - a[ok]).max() < bound


def test_a_chunk_of_land_only_reads_back_as_nan(tmp_path):
    a = masked_field()
    z = create(tmp_path / "m", a)
    z[:] = a
    assert np.isnan(z[0:64, 0:16, 32:64]).all()


@pytest.mark.parametrize("kw", CONFIGS)
def test_encoded_size_counts_the_mask(tmp_path, kw):
    a = masked_field()
    z = create(tmp_path / "m", a, **kw)
    z[:] = a
    spec = ArraySpec(
        shape=z.chunks,
        dtype=z.metadata.data_type,
        fill_value=z.metadata.fill_value,
        config=ArrayConfig.from_dict({}),
        prototype=default_buffer_prototype(),
    )
    c = SkipZFPCodec(masked=True, **kw)
    size = c.compute_encoded_size(0, spec)
    payload = codec.native().layout(SHAPE, c.rate, 4)[0]
    assert size == payload + SHAPE[1] * SHAPE[2] // 8
    assert size == (tmp_path / "m" / "data" / "c" / "0" / "0" / "0").stat().st_size


def test_rejects_nan_that_moves_along_axis_0(tmp_path):
    a = masked_field()
    a[5, 3, 4] = np.nan
    z = create(tmp_path / "m", a)
    with pytest.raises(ValueError, match="differ between steps 0 and 5 along axis 0"):
        z[:] = a


@pytest.mark.parametrize("fill", (np.nan, 0.0))
def test_writing_part_of_a_chunk_explains_why_it_fails(tmp_path, fill):
    """The unwritten steps hold the fill value, so the NaN positions differ."""
    a = masked_field()
    z = create(tmp_path / "m", a, fill_value=fill)
    with pytest.raises(ValueError, match="Write whole chunks along axis 0"):
        z[0:32] = a[0:32]


def test_rejects_infinite_values(tmp_path):
    a = masked_field()
    a[3, 0, 0] = np.inf
    z = create(tmp_path / "m", a)
    with pytest.raises(ValueError, match="infinite"):
        z[:] = a


def test_fill_uses_the_block_mean_and_ignores_how_the_array_is_cut():
    a = masked_field()
    f = codec.fill_nan(a)
    assert not np.isnan(f).any()
    ok = ~np.isnan(a)
    assert np.array_equal(f[ok], a[ok])
    # a block on the coast is filled with the mean of its valid values
    valid = (~np.isnan(a[:4])).reshape(4, 8, 4, 16, 4).sum(axis=(0, 2, 4))
    by, bx = np.argwhere((valid > 0) & (valid < 64))[0]
    b = (slice(0, 4), slice(4 * by, 4 * by + 4), slice(4 * bx, 4 * bx + 4))
    blk, mine = a[b], f[b]
    assert np.allclose(mine[np.isnan(blk)], np.nanmean(blk.astype(np.float64)))
    # a block with no valid value is 0
    assert (f[0:4, 0:4, 60:64] == 0).all()
    # filling one unit gives the same values as filling the whole array
    for sl in codec.unit_slices(a.shape, (32, 16, 32)):
        assert np.array_equal(codec.fill_nan(a[sl]), f[sl])
