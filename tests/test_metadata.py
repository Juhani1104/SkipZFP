"""The zone map bounds the original and decoded values, and write_meta stores it."""

import asyncio

import numpy as np
import pytest
import zarr

from skipzfp import aggregate, codec, query

from helpers import RATES, SHAPE, smooth_field, split, true_blocks


@pytest.mark.parametrize("rate", RATES)
def test_metadata_bounds_original_and_reconstructed(rate):
    a = smooth_field()
    buf, cmin, cmax, eps, offs = split(a, rate)
    rec = codec.native().decode(buf, SHAPE, rate, 4)

    assert cmin == a.min() and cmax == a.max()
    assert eps >= np.abs(rec.astype(np.float64) - a).max()

    span = cmax - cmin
    lo = cmin + offs[:, 0] / 255 * span
    hi = cmin + offs[:, 1] / 255 * span
    orig_b, rec_b = true_blocks(a), true_blocks(rec)
    assert np.all(lo <= orig_b.min(1)) and np.all(hi >= orig_b.max(1))
    assert np.all(lo - eps <= rec_b.min(1)) and np.all(hi + eps >= rec_b.max(1))


@pytest.fixture
def array_and_data(tmp_path):
    a = smooth_field(shape=(128, 32, 64), seed=7)
    g = zarr.open_group(str(tmp_path / "a"), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=(32, 16, 32)),
        compressors=None,
        filters=None,
    )
    z[:] = a
    return z, a


def test_write_meta_layout(array_and_data):
    z, a = array_and_data
    m = codec.write_meta(z, a)
    unit = z.metadata.codecs[0].unit(SHAPE)
    grid = tuple(s // u for s, u in zip(a.shape, unit))
    assert m.shape == (*grid, codec.meta_size(z.metadata.codecs[0], unit))
    assert z.attrs[codec.META_ATTR] == "data_meta"


def test_write_meta_custom_path_and_t_chunk(array_and_data):
    z, a = array_and_data
    ref = codec.write_meta(z, a)[:]
    m = codec.write_meta(z, a, path="elsewhere", t_chunk=1)
    assert z.attrs[codec.META_ATTR] == "elsewhere"
    assert np.array_equal(m[:], ref)


def read_both(z):
    """Read the zone map by key and by opening it as a Zarr array."""
    lt = query.get_layout(z)
    direct = asyncio.run(query.read_meta(z, lt))
    opened = asyncio.run(query.read_meta_zarr(z, lt, z.attrs[codec.META_ATTR]))
    return direct, opened


# the array has 4 units along t, so t_chunk=3 leaves a padded last object
@pytest.mark.parametrize("t_chunk", (16, 3, 1))
def test_read_meta_by_key_matches_zarr(array_and_data, t_chunk):
    z, a = array_and_data
    codec.write_meta(z, a, t_chunk=t_chunk)
    assert z.attrs[codec.META_T_CHUNK_ATTR] == min(t_chunk, 4)
    (data, nbytes, n_obj), (ref, _, ref_obj) = read_both(z)
    assert np.array_equal(data, ref)
    assert n_obj == ref_obj == -(-4 // min(t_chunk, 4))
    assert nbytes >= data.nbytes


def test_query_does_not_open_zone_map_json(array_and_data, tmp_path):
    """With t_chunk recorded, the query never reads the zone map's zarr.json."""
    z, a = array_and_data
    codec.write_meta(z, a)
    th = float(np.median(a))
    want = query.query_gt(z, th)
    (tmp_path / "a" / "data_meta" / "zarr.json").unlink()
    got = query.query_gt(z, th)
    assert (got.count, got.maybe_blocks) == (want.count, want.maybe_blocks)
    assert got.metadata_requests == 1


def test_read_meta_falls_back_without_t_chunk(array_and_data):
    """Zone maps written before t_chunk was recorded are opened as arrays."""
    z, a = array_and_data
    codec.write_meta(z, a)
    want = query.query_gt(z, float(np.median(a)))
    del z.attrs[codec.META_T_CHUNK_ATTR]
    assert codec.META_T_CHUNK_ATTR not in z.attrs
    got = query.query_gt(z, float(np.median(a)))
    assert (got.count, got.maybe_blocks) == (want.count, want.maybe_blocks)


def test_write_meta_writes_all_zero_objects(tmp_path):
    """An all-zero zone map object is still written, so it can be read by key."""
    a = np.zeros((64, 16, 32), np.float32)
    g = zarr.open_group(str(tmp_path / "z"), mode="w")
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
    (data, _, _), (ref, _, _) = read_both(z)
    assert not data.any() and np.array_equal(data, ref)
    assert query.query_gt(z, -1.0).count == a.size


def test_read_meta_reports_missing_object(array_and_data, tmp_path):
    z, a = array_and_data
    codec.write_meta(z, a, t_chunk=1)
    (tmp_path / "a" / "data_meta" / "c" / "2" / "0" / "0" / "0").unlink()
    with pytest.raises(FileNotFoundError, match="data_meta/c/2/0/0/0"):
        query.query_gt(z, 285.0)


def test_read_meta_rejects_object_of_wrong_size(array_and_data, tmp_path):
    z, a = array_and_data
    codec.write_meta(z, a)
    obj = tmp_path / "a" / "data_meta" / "c" / "0" / "0" / "0" / "0"
    obj.write_bytes(obj.read_bytes()[:-1])
    with pytest.raises(ValueError, match="does not match the array layout"):
        query.query_gt(z, 285.0)


def test_write_meta_rejects_wrong_shape(array_and_data):
    z, a = array_and_data
    with pytest.raises(ValueError, match="must match the array shape"):
        codec.write_meta(z, a[:64])


def test_write_meta_requires_group(tmp_path):
    a = smooth_field()
    z = zarr.create_array(
        store=str(tmp_path / "root"),
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0),
        compressors=None,
        filters=None,
    )
    z[:] = a
    with pytest.raises(ValueError, match="inside a group"):
        codec.write_meta(z, a)


def sub_chunk_array(tmp_path, layers=()):
    """A 4x2x2 chunk grid of 4x2x2 units each, so t_chunk can split a chunk."""
    a = smooth_field(shape=(128, 32, 64), seed=3)
    g = zarr.open_group(str(tmp_path / "s"), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=(32, 16, 32),
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=(8, 8, 16), layers=layers),
        compressors=None,
        filters=None,
    )
    z[:] = a
    return z, a


@pytest.mark.parametrize("layers", [(), (2.0, 4.0, 8.0)])
@pytest.mark.parametrize(
    "t_chunk, pipelined", [(4, True), (8, True), (100, True), (2, False), (3, False)]
)
def test_pipelined_query_matches_whole_zone_map(
    tmp_path, monkeypatch, layers, t_chunk, pipelined
):
    """Planning one zone map object at a time gives the same answer as planning
    the whole zone map, and is used only when objects cover whole chunks."""
    z, a = sub_chunk_array(tmp_path, layers)
    codec.write_meta(z, a, t_chunk=t_chunk)
    calls = []
    real = query._query_pipelined

    async def spy(*args):
        calls.append(1)
        return await real(*args)

    monkeypatch.setattr(query, "_query_pipelined", spy)
    opts = query.QueryOptions(layer=0, merge_gap_blocks=2)
    lo, hi = np.quantile(a, [0.3, 0.7])
    got = [
        query.query_gt(z, float(lo), options=opts),
        query.query_range(z, float(lo), float(hi), options=opts),
    ]
    assert bool(calls) == pipelined

    del z.attrs[codec.META_T_CHUNK_ATTR]
    want = [
        query.query_gt(z, float(lo), options=opts),
        query.query_range(z, float(lo), float(hi), options=opts),
    ]
    for g_, w in zip(got, want):
        assert (g_.count, g_.in_blocks, g_.out_blocks, g_.maybe_blocks) == (
            w.count,
            w.in_blocks,
            w.out_blocks,
            w.maybe_blocks,
        )
        assert g_.payload_bytes_read == w.payload_bytes_read
        assert g_.payload_requests == w.payload_requests


def test_pack_meta_round_trips_any_bytes():
    rng = np.random.default_rng(0)
    rec = rng.integers(0, 256, (37, 8 + 4 * 2 + 2 * 64 + 32), dtype=np.uint8)
    packed = codec.pack_meta(rec, 16, 64)
    assert packed.shape == (rec.size,)
    assert np.array_equal(codec.unpack_meta(packed, rec.shape[1], 16, 64), rec)


# t_chunk=3 leaves a padded last object, and 4-bit codes and layers lengthen records
@pytest.mark.parametrize("t_chunk", (None, 3, 1))
@pytest.mark.parametrize(
    "kw", [dict(), dict(block_error_bits=4), dict(layers=(2.0, 4.0, 8.0))]
)
def test_compressed_zone_map_reads_like_raw(tmp_path, t_chunk, kw):
    """A compressed zone map reads back, by key and as a Zarr array, as the raw
    one does, and queries on it give the same answers from fewer bytes."""
    a = smooth_field(shape=(128, 32, 64), seed=5)
    g = zarr.open_group(str(tmp_path / "c"), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=(32, 16, 32), **kw),
        compressors=None,
        filters=None,
    )
    z[:] = a
    lt = query.get_layout(z)
    lo, hi = (float(x) for x in np.quantile(a, [0.3, 0.7]))
    opts = [query.QueryOptions(layer=0), query.QueryOptions(layer=0, pipeline=False)]

    def answers():
        return [
            (r.count, r.in_blocks, r.out_blocks, r.maybe_blocks, r.metadata_bytes_read)
            for o in opts
            for r in (
                query.query_gt(z, lo, options=o),
                query.query_range(z, lo, hi, options=o),
                aggregate.aggregate_gt(z, lo, options=o),
            )
        ]

    codec.write_meta(z, a, t_chunk=t_chunk)
    raw, raw_bytes, _ = asyncio.run(query.read_meta(z, lt))
    want = answers()
    codec.write_meta(z, a, t_chunk=t_chunk, compress=True)
    assert z.attrs[codec.META_PACKING_ATTR] == codec.PLANAR_ZSTD
    (data, nbytes, _), (opened, _, _) = read_both(z)
    assert np.array_equal(data, raw) and np.array_equal(opened, raw)
    assert nbytes < raw_bytes
    got = answers()
    assert [g_[:4] for g_ in got] == [w[:4] for w in want]
    assert all(g_[4] < w[4] for g_, w in zip(got, want))

    codec.write_meta(z, a, t_chunk=t_chunk)
    assert z.attrs[codec.META_PACKING_ATTR] is None
    assert answers() == want


def test_read_meta_rejects_unknown_packing(array_and_data):
    z, a = array_and_data
    codec.write_meta(z, a)
    z.attrs[codec.META_PACKING_ATTR] = "lz4"
    with pytest.raises(ValueError, match="unknown zone map packing"):
        asyncio.run(query.read_meta(z, query.get_layout(z)))
