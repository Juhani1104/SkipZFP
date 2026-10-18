"""The zone map goes stale when chunks are rewritten; check_meta finds them."""

import numpy as np
import pytest

from skipzfp import check_meta, codec, query

from helpers import SHAPE, make_array, smooth_field


@pytest.fixture
def stored(tmp_path):
    a = smooth_field(shape=(128, 32, 64), seed=10)
    return a, make_array(tmp_path, a, rate=8.0)


def test_fresh_zone_map_matches(stored):
    _, z = stored
    assert check_meta(z) == []
    src = z.attrs[codec.META_SOURCE_ATTR]
    assert src["shape"] == list(z.shape) and src["chunks"] == list(SHAPE)
    assert len(src["chunk_crc32"]) == 8 and None not in src["chunk_crc32"]


def test_rewritten_chunk_makes_answers_wrong_until_rebuilt(stored):
    """A plain Zarr write leaves the zone map stale, and a query then goes wrong.

    Chunk (1, 0, 1) is rewritten to 1000 while the zone map still says its
    values are near 280, so x > 500 skips it and counts nothing.
    """
    a, z = stored
    sl = np.s_[64:128, 0:16, 32:64]
    b = a.copy()
    b[sl] = 1000.0
    z[sl] = b[sl]
    assert check_meta(z) == [(1, 0, 1)]
    assert query.query_gt(z, 500.0).count == 0  # stale: the true count is below
    assert int((z[:] > 500.0).sum()) == 64 * 16 * 32

    codec.write_meta(z, b)
    assert check_meta(z) == []
    assert query.query_gt(z, 500.0).count == 64 * 16 * 32


def test_zone_map_for_another_layout_is_refused(stored):
    _, z = stored
    src = dict(z.attrs[codec.META_SOURCE_ATTR])
    for key, value in (
        ("shape", [1, 2, 3]),
        ("chunks", [64, 16, 16]),
        ("codec", codec.SkipZFPCodec(rate=4.0).to_dict()),
    ):
        z.update_attributes({codec.META_SOURCE_ATTR: {**src, key: value}})
        with pytest.raises(ValueError, match="write_meta again"):
            query.query_gt(z, 280.0)
        with pytest.raises(ValueError, match="write_meta again"):
            check_meta(z)
    z.update_attributes({codec.META_SOURCE_ATTR: src})
    assert query.query_gt(z, 280.0).count == int((z[:] > 280.0).sum())


def test_zone_map_without_a_recorded_source_still_works(stored):
    """Arrays written before the source was recorded keep working, unchecked."""
    _, z = stored
    attrs = dict(z.attrs)
    del attrs[codec.META_SOURCE_ATTR]
    z.attrs.clear()
    z.update_attributes(attrs)
    assert query.query_gt(z, 280.0).count == int((z[:] > 280.0).sum())
    with pytest.raises(ValueError, match="no zone map source"):
        check_meta(z)


def test_crc_reads_few_chunks_at_once(tmp_path, monkeypatch):
    """write_meta and check_meta read chunks a few at a time, not all together.

    Reading them all at once would hold the whole payload in memory.
    """
    a = smooth_field(shape=(512, 16, 32), seed=11)
    z = make_array(tmp_path, a, rate=8.0)  # 8 chunks
    store = type(z.store_path.store)
    orig = store.get
    now, most = [0], [0]

    async def counting(self, *args, **kwargs):
        now[0] += 1
        most[0] = max(most[0], now[0])
        try:
            import asyncio

            await asyncio.sleep(0.01)  # let other reads start if they may
            return await orig(self, *args, **kwargs)
        finally:
            now[0] -= 1

    monkeypatch.setattr(store, "get", counting)
    assert len(codec.chunk_crc32(z, concurrency=2)) == 8
    assert most[0] == 2
    most[0] = 0
    codec.chunk_crc32(z)
    assert 1 < most[0] <= 16
