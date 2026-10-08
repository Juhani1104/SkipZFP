"""query_gt counts exactly, accounts for what it reads, and reports bad input."""

import os

import numpy as np
import pytest
import zarr

from skipzfp import codec, query

from helpers import RATES, SHAPE, classify, make_array, smooth_field, true_blocks


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
    """IN and OUT blocks are right for both the original and the decoded values.

    Every value of an IN block must exceed the threshold and every value of an OUT
    block must not, which is what lets a query count or skip them without reading.
    """
    a, z, rate = stored
    rec = z[:]
    th = np.float32(np.quantile(a, q))
    lt = query.get_layout(z)
    n_chunk = int(np.prod(lt.grid_shape))

    for c in range(n_chunk):
        coord = np.unravel_index(c, lt.grid_shape)
        sl = tuple(slice(ci * s, (ci + 1) * s) for ci, s in zip(coord, SHAPE))
        meta = codec.native().meta(a[sl], rate, 4)
        _, _, n_in, n_out = query.native().plan_chunks(
            meta, lt.meta_size, (1, 1, 1), (1, 1, 1), lt.blocks_per_chunk, th, 1
        )
        states = classify(meta, lt, th)
        assert (states == 1).sum() == n_in and (states == 0).sum() == n_out
        ob, rb = true_blocks(a[sl]), true_blocks(rec[sl])
        assert np.all(ob[states == 1] > th) and np.all(rb[states == 1] > th)
        assert np.all(ob[states == 0] <= th) and np.all(rb[states == 0] <= th)


def test_threshold_compared_in_double(stored):
    """A threshold between two float32 values is compared in double.

    The threshold lies just below a stored value but rounds to it in float32, so
    casting it to float32 would drop the values equal to it from the count.
    """
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
    """The byte and request counts in QueryResult add up.

    Each total equals the sum of its parts, and with no merge gap nothing is read
    beyond the MAYBE blocks.
    """
    z, rec = one
    res = query.query_gt(
        z, float(np.median(rec)), options=query.QueryOptions(merge_gap_blocks=gap)
    )
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
    assert query.query_gt(z, th, options=query.QueryOptions(**kw)).count == int(
        (rec > th).sum()
    )


def test_query_by_path(one):
    z, rec = one
    root = z.store_path.store.root
    res = query.query_gt(str(root), 285.0, array_path="data")
    assert res.count == int((rec > 285.0).sum())


def test_query_on_memory_store():
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


# one call per public query function, so shared checks run on all three
CALLS = {
    "gt": lambda z, **kw: query.query_gt(z, 285.0, **kw),
    "lt": lambda z, **kw: query.query_lt(z, 285.0, **kw),
    "range": lambda z, **kw: query.query_range(z, 280.0, 285.0, **kw),
}
ASYNC_CALLS = {
    "gt": lambda z: query.query_gt_async(z, 285.0),
    "lt": lambda z: query.query_lt_async(z, 285.0),
    "range": lambda z: query.query_range_async(z, 280.0, 285.0),
}


@pytest.mark.parametrize(
    "kw, msg",
    [
        (dict(threads=-1), "threads"),
        (dict(request_concurrency=0), "request_concurrency"),
        (dict(request_batch_size=0), "request_batch_size"),
        (dict(merge_gap_blocks=-1), "merge_gap_blocks"),
        (dict(max_chunk_requests=0), "max_chunk_requests"),
    ],
)
def test_rejects_bad_query_options(kw, msg):
    with pytest.raises(ValueError, match=msg):
        query.QueryOptions(**kw)


@pytest.mark.parametrize("layer", (1, -1))
@pytest.mark.parametrize("fn", ("gt", "lt", "range"))
def test_rejects_layer_the_array_does_not_have(one, fn, layer):
    z, _ = one
    with pytest.raises(ValueError, match="layer must be in"):
        CALLS[fn](z, options=query.QueryOptions(layer=layer))


def test_threads_sizes_the_decode_pool(one):
    """threads also sets the worker count of the pool that decodes layer-0 reads."""
    z, rec = one
    th = float(np.quantile(rec, 0.7))
    res = query.query_gt(z, th, options=query.QueryOptions(threads=3))
    assert res.count == int((rec > th).sum())
    assert query.pool(3)._max_workers == 3
    assert query.pool(0)._max_workers == (os.cpu_count() or 4)


@pytest.mark.parametrize("fn", ("gt", "lt", "range"))
def test_async_matches_sync(one, fn):
    """Each async entry gives the same answer as its sync counterpart."""
    z, _ = one
    got = zarr.core.sync.sync(ASYNC_CALLS[fn](z))
    want = CALLS[fn](z)
    assert (got.count, got.maybe_blocks) == (want.count, want.maybe_blocks)


@pytest.mark.parametrize("fn", ("gt", "lt", "range"))
def test_async_entry_requires_open_array(fn):
    with pytest.raises(TypeError, match="opened zarr.Array"):
        zarr.core.sync.sync(ASYNC_CALLS[fn]("not-an-array"))


def test_rejects_metadata_of_wrong_shape(tmp_path):
    a = smooth_field(shape=(128, 32, 64), seed=10)
    z = make_array(tmp_path / "a", a, rate=8.0)
    g = zarr.open_group(str(tmp_path / "a"), mode="a")
    g.create_array("wrong_meta", data=np.zeros((1, 1, 1, 4), np.uint8))
    z.update_attributes({codec.META_ATTR: "wrong_meta"})
    with pytest.raises(ValueError, match="does not match the array layout"):
        query.query_gt(z, 285.0)


def _damaged(tmp_path, damage):
    a = smooth_field(shape=(128, 32, 64), seed=11)
    z = make_array(tmp_path / "a", a, rate=8.0)
    chunk = tmp_path / "a" / "data" / "c" / "1" / "1" / "1"
    damage(chunk)
    return z, float(np.median(a))


def test_missing_chunk_is_reported(tmp_path):
    z, th = _damaged(tmp_path, lambda f: f.unlink())
    with pytest.raises(FileNotFoundError):
        query.query_gt(z, th)


def test_truncated_chunk_is_reported(tmp_path):
    z, th = _damaged(tmp_path, lambda f: f.write_bytes(f.read_bytes()[:100]))
    with pytest.raises(OSError, match="short range read"):
        query.query_gt(z, th)


def test_merge_ranges_sorts_its_input():
    rng = np.random.default_rng(0)
    chunks = rng.integers(0, 3, 200).astype(np.uint32)
    blocks = rng.integers(0, 512, 200).astype(np.uint32)
    key = np.unique((chunks.astype(np.uint64) << np.uint64(32)) | blocks)
    sc, sb = (key >> np.uint64(32)).astype(np.uint32), key.astype(np.uint32)
    keys = ["a", "b", "c"]
    perm = rng.permutation(len(sc))
    got, got_blocks = query.merge_ranges(keys, sc[perm], sb[perm], 64, 4)
    want, want_blocks = query.merge_ranges(keys, sc, sb, 64, 4)
    assert got == want and np.array_equal(got_blocks, want_blocks)
