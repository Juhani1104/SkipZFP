"""The zone map bounds the original and decoded values, and write_meta stores it."""

import asyncio

import numpy as np
import pytest
import zarr

from skipzfp import codec, query

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
