import numpy as np
import pytest
import zarr
from helpers import make_array, smooth_field

from skipzfp import SkipZFPCodec

CONFIGS = {
    "plain": dict(rate=8.0),
    "order120": dict(rate=8.0, block_order=(1, 2, 0)),
    "order201": dict(rate=8.0, block_order=(2, 0, 1)),
    "layers": dict(rate=8.0, layers=(2.0, 4.0, 8.0)),
    "sub": dict(rate=8.0, sub_chunk=(32, 8, 16), block_order=(1, 2, 0)),
}
SLICES = [
    (slice(None), slice(None), slice(None)),
    (0, 0, 0),
    (slice(60, 70), slice(14, 18), slice(30, 34)),
    (slice(3, 125), slice(1, 31), slice(2, 63)),
    (slice(None, None, 3), slice(1, None, 5), slice(None, None, 7)),
    (100, slice(None), 17),
]


@pytest.fixture(scope="module")
def arrays(tmp_path_factory):
    a = smooth_field(shape=(128, 32, 64), seed=4)
    base = tmp_path_factory.mktemp("read")
    ref = make_array(base / "ref", a, rate=8.0)[:]
    arrs = {name: make_array(base / name, a, **kw) for name, kw in CONFIGS.items()}
    return a, ref, arrs, base


@pytest.mark.parametrize("name", CONFIGS)
def test_layout_does_not_change_values(arrays, name):
    _, ref, arrs, _ = arrays
    assert np.array_equal(arrs[name][:], ref)


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("sl", SLICES)
def test_partial_read_matches_full_read(arrays, name, sl):
    _, ref, arrs, _ = arrays
    assert np.array_equal(arrs[name][sl], ref[sl])


@pytest.mark.parametrize("name", CONFIGS)
def test_reopened_array_reads_same(arrays, name):
    _, ref, _, base = arrays
    z = zarr.open_array(str(base / name), path="data", mode="r")
    assert np.array_equal(z[:], ref)


def test_error_within_rate_budget(arrays):
    a, ref, _, _ = arrays
    err = np.abs(ref.astype(np.float64) - a)
    assert 0 < err.max() < 0.1 * (a.max() - a.min())


def test_memory_store_round_trip():
    a = smooth_field(seed=6)
    g = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=a.shape,
        dtype="float32",
        serializer=SkipZFPCodec(rate=8.0),
        compressors=None,
        filters=None,
    )
    z[:] = a
    assert np.abs(z[:].astype(np.float64) - a).max() < 0.1
