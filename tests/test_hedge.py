import asyncio
import time
from collections import Counter

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, WrapperStore

from skipzfp import query

from helpers import make_array, smooth_field


class SlowFirstRead(WrapperStore):
    """Stalls the first read of the byte ranges numbered in `victims`."""

    def __init__(self, store, stall=0.0, victims=()):
        super().__init__(store)
        self.stall = stall
        self.victims = set(victims)
        self.reads = Counter()
        self.order = {}

    async def get(self, key, prototype, byte_range=None):
        if byte_range is not None:
            tag = (key, byte_range.start, byte_range.end)
            self.order.setdefault(tag, len(self.order))
            self.reads[tag] += 1
            if self.reads[tag] == 1 and self.order[tag] in self.victims:
                await asyncio.sleep(self.stall)
        return await self._store.get(key, prototype, byte_range)


@pytest.fixture()
def stored(tmp_path):
    a = smooth_field(shape=(128, 32, 64), seed=4)
    make_array(tmp_path / "h", a, rate=8.0)
    return a, tmp_path / "h"


def open_with(path, **kw):
    store = SlowFirstRead(LocalStore(str(path), read_only=True), **kw)
    return zarr.open_array(store=store, path="data", mode="r"), store


def test_hedged_reads_skip_stalled_requests(stored):
    a, path = stored
    th = float(np.median(a))
    ref = query.query_gt(zarr.open_array(str(path), path="data", mode="r"), th)
    z, store = open_with(path, stall=5.0, victims=(3, 7))
    t0 = time.perf_counter()
    r = query.query_gt(z, th, hedge=True, hedge_min_samples=1, hedge_min_delay=0.01)
    assert time.perf_counter() - t0 < 4.0
    assert (r.count, r.maybe_blocks, r.payload_requests) == (
        ref.count,
        ref.maybe_blocks,
        ref.payload_requests,
    )
    assert max(store.reads.values()) == 2


def test_no_duplicate_reads_by_default(stored):
    a, path = stored
    z, store = open_with(path)
    r = query.query_gt(z, float(np.median(a)))
    assert r.count > 0
    assert max(store.reads.values()) == 1


@pytest.mark.parametrize(
    "kw",
    (dict(hedge_factor=0), dict(hedge_min_samples=0), dict(hedge_min_delay=-1)),
)
def test_hedge_arguments_are_checked(stored, kw):
    a, path = stored
    z, _ = open_with(path)
    with pytest.raises(ValueError):
        query.query_gt(z, float(np.median(a)), hedge=True, **kw)
