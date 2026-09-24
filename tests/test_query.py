import numpy as np
import pytest
import zarr
from helpers import RATES, SHAPE, classify, make_array, smooth_field, true_blocks

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


@pytest.fixture(scope="module")
def one(tmp_path_factory):
    a = smooth_field(shape=(128, 32, 64), seed=8)
    z = make_array(tmp_path_factory.mktemp("q") / "a", a, rate=8.0)
    return z, z[:].astype(np.float64)


def test_threshold_below_min_reads_no_payload(one):
    z, rec = one
    res = query.query_gt(z, float(rec.min()) - 1.0)
    assert res.count == rec.size and res.in_blocks == res.total_blocks
    assert res.payload_bytes_read == 0 and res.payload_requests == 0


def test_threshold_above_max_reads_no_payload(one):
    z, rec = one
    res = query.query_gt(z, float(rec.max()) + 1.0)
    assert res.count == 0 and res.out_blocks == res.total_blocks
    assert res.payload_bytes_read == 0 and res.payload_requests == 0


@pytest.mark.parametrize("gap", (0, 64, 10**6))
def test_accounting_is_consistent(one, gap):
    z, rec = one
    res = query.query_gt(z, float(np.median(rec)), merge_gap_blocks=gap)
    assert res.bytes_read == res.metadata_bytes_read + res.payload_bytes_read
    assert res.range_requests == res.metadata_requests + res.payload_requests
    assert (
        res.useful_payload_bytes + res.payload_overread_bytes == res.payload_bytes_read
    )
    assert res.maybe_blocks > 0 and res.payload_requests > 0
    if gap == 0:
        assert res.payload_overread_bytes == 0


@pytest.mark.parametrize(
    "kw",
    [
        dict(threads=1),
        dict(threads=4),
        dict(request_concurrency=1),
        dict(request_batch_size=1),
        dict(request_batch_size=7),
        dict(merge_gap_blocks=10**6),
        dict(max_chunk_requests=10**9),
    ],
)
def test_tuning_does_not_change_count(one, kw):
    z, rec = one
    th = float(np.quantile(rec, 0.7))
    assert query.query_gt(z, th, **kw).count == int((rec > th).sum())


def test_query_by_path(one):
    z, rec = one
    root = z.store_path.store.root
    res = query.query_gt(str(root), 285.0, array_path="data")
    assert res.count == int((rec > 285.0).sum())


def test_memory_store():
    a = smooth_field(shape=(128, 32, 64), seed=9)
    g = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a)
    rec = z[:].astype(np.float64)
    assert query.query_gt(z, 283.0).count == int((rec > 283.0).sum())


def test_rejects_array_without_codec(tmp_path):
    g = zarr.open_group(str(tmp_path / "a"), mode="w")
    z = g.create_array("data", shape=SHAPE, chunks=SHAPE, dtype="float32")
    with pytest.raises(ValueError, match="does not use the skipzfp codec"):
        query.query_gt(z, 0.0)


def test_rejects_array_without_metadata(tmp_path):
    g = zarr.open_group(str(tmp_path / "a"), mode="w")
    z = g.create_array(
        "data",
        shape=SHAPE,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0),
        compressors=None,
        filters=None,
    )
    z[:] = smooth_field()
    with pytest.raises(ValueError, match="write_meta"):
        query.query_gt(z, 0.0)


@pytest.mark.parametrize(
    "kw, msg",
    [
        (dict(array_path="x"), "array_path"),
        (dict(storage_options={"anon": True}), "storage_options"),
    ],
)
def test_open_skipzfp_rejects_options_for_open_array(one, kw, msg):
    z, _ = one
    with pytest.raises(ValueError, match=msg):
        query.open_skipzfp(z, **kw)
