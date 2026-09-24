import numpy as np
import pytest
import zarr
from helpers import RATES, SHAPE, classify, smooth_field, true_blocks

from skipzfp import codec, query


@pytest.fixture(params=RATES)
def stored(request, tmp_path):
    rate = request.param
    a = smooth_field(shape=(128, 32, 64), seed=1)
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=rate),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a)
    return a, z, rate


@pytest.mark.parametrize("q", (0.01, 0.1, 0.5, 0.9, 0.99))
def test_query_gt_count_is_exact_on_reconstruction(stored, q):
    a, z, _ = stored
    rec = z[:]
    th = float(np.quantile(a, q))
    res = query.query_gt(z, th)
    assert res.count == int((rec > th).sum())
    assert res.in_blocks + res.out_blocks + res.maybe_blocks == res.total_blocks


@pytest.mark.parametrize("q", (0.1, 0.5, 0.9))
def test_in_out_guarantees_hold_for_original_and_reconstruction(stored, q):
    a, z, rate = stored
    rec = z[:]
    th = np.float32(np.quantile(a, q))
    lt = query.get_layout(z)
    n_chunk = int(np.prod(lt.grid_shape))

    for c in range(n_chunk):
        coord = np.unravel_index(c, lt.grid_shape)
        sl = tuple(slice(ci * s, (ci + 1) * s) for ci, s in zip(coord, SHAPE))
        meta = codec.native().meta(a[sl], rate, 4)
        _, _, n_in, n_out = query.native().plan(
            meta, 1, lt.meta_size, lt.blocks_per_chunk, th, 1
        )
        states = classify(meta, lt, th)
        assert (states == 1).sum() == n_in and (states == 0).sum() == n_out
        ob, rb = true_blocks(a[sl]), true_blocks(rec[sl])
        assert np.all(ob[states == 1] > th) and np.all(rb[states == 1] > th)
        assert np.all(ob[states == 0] <= th) and np.all(rb[states == 0] <= th)


def test_threshold_compared_in_double(stored):
    a, z, _ = stored
    rec = z[:]
    v = np.float32(np.median(rec))
    # just below v in double, but rounds to v in float32
    th = float(v) - float(np.spacing(v)) / 4
    assert np.float32(th) == v
    res = query.query_gt(z, th)
    assert res.count == int((rec.astype(np.float64) > th).sum())
