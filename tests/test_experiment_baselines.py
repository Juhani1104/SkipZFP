"""The Zarr sharding baseline of the experiments merges reads within each shard."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import baselines  # noqa: E402


def reads(shard, offs, sizes, gap):
    shard, offs = np.asarray(shard), np.asarray(offs, np.uint64)
    ends = offs + np.asarray(sizes, np.uint64)
    starts, stops = baselines.merge_runs(shard, offs, ends, gap)
    return [
        (int(shard[a]), int(offs[a]), int(ends[a:b].max()))
        for a, b in zip(starts, stops)
    ]


def test_every_shard_merges_on_its_own():
    # chunks 0 and 9 of two shards of ten 100-byte chunks: with no gap allowed,
    # the second shard must not be read from its first chunk to its last
    got = reads([0, 0, 1, 1], [0, 900, 0, 900], [100] * 4, gap=0)
    assert got == [(0, 0, 100), (0, 900, 1000), (1, 0, 100), (1, 900, 1000)]


def test_gap_merges_within_a_shard_only():
    got = reads([0, 0, 1], [0, 150, 0], [100] * 3, gap=50)
    assert got == [(0, 0, 250), (1, 0, 100)]
    got = reads([0, 0, 1], [0, 150, 0], [100] * 3, gap=49)
    assert got == [(0, 0, 100), (0, 150, 250), (1, 0, 100)]


def test_matches_a_loop_over_shards():
    rng = np.random.default_rng(3)
    shard = np.sort(rng.integers(0, 40, 3000))
    sizes = rng.integers(1, 5000, shard.size).astype(np.uint64)
    offs = np.empty(shard.size, np.uint64)
    for s in np.unique(shard):  # chunks of a shard laid out back to back, then thinned
        i = np.flatnonzero(shard == s)
        offs[i] = np.cumsum(sizes[i] + rng.integers(0, 3000, i.size).astype(np.uint64))
    for gap in (0, 1000, 10_000):
        want = []
        for s in np.unique(shard):
            i = np.flatnonzero(shard == s)
            o, e = offs[i].astype(int), (offs[i] + sizes[i]).astype(int)
            cur = [o[0], e[0]]
            for a, b in zip(o[1:], e[1:]):
                if a - cur[1] <= gap:
                    cur[1] = max(cur[1], b)
                else:
                    want.append((int(s), *cur))
                    cur = [a, b]
            want.append((int(s), *cur))
        assert reads(shard, offs, sizes, gap) == [tuple(map(int, r)) for r in want]
