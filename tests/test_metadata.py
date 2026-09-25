import numpy as np
import pytest
import zarr

from skipzfp import codec

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
