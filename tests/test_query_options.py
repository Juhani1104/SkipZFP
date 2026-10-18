"""pipeline, zero_copy, max_inflight_bytes, meta_by_key and trace change how a query
reads, not what."""

import asyncio
import threading
import time

import numpy as np
import pytest
import zarr
from zarr.core.buffer import cpu

from skipzfp import QueryOptions, aggregate_gt, codec, query

from helpers import SHAPE, smooth_field


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """Chunks of sub-chunks, the zone map in objects, so the query can pipeline."""
    a = smooth_field(shape=(256, 32, 64), seed=8)
    g = zarr.open_group(str(tmp_path_factory.mktemp("opts")), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=(128, 32, 64),
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=SHAPE),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a, t_chunk=2)
    return a, z[:], z


def same_answer(r, ref):
    assert r.count == ref.count
    assert (r.in_blocks, r.out_blocks, r.maybe_blocks) == (
        ref.in_blocks,
        ref.out_blocks,
        ref.maybe_blocks,
    )
    assert r.useful_payload_bytes == ref.useful_payload_bytes


@pytest.mark.parametrize("q", (0.01, 0.5))
def test_pipeline_off_takes_the_plain_path(stored, q, monkeypatch):
    a, rec, z = stored
    th = float(np.quantile(a, q))
    on = query.query_gt(z, th)
    assert on.count == int((rec > th).sum())

    async def no_pipeline(*args, **kwargs):
        raise AssertionError("pipeline=False must not pipeline")

    monkeypatch.setattr(query, "_query_pipelined", no_pipeline)
    off = query.query_gt(z, th, options=QueryOptions(pipeline=False))
    same_answer(off, on)
    with pytest.raises(AssertionError, match="must not pipeline"):
        query.query_gt(z, th)  # the default really is the pipelined path


@pytest.mark.parametrize("pipeline", (True, False))
def test_zero_copy_off_copies_every_read(stored, pipeline, monkeypatch):
    a, _, z = stored
    th = float(np.quantile(a, 0.5))
    ref = query.query_gt(z, th, options=QueryOptions(pipeline=pipeline))
    calls = []
    orig = cpu.Buffer.to_bytes

    def counting(self):
        calls.append(1)
        return orig(self)

    monkeypatch.setattr(cpu.Buffer, "to_bytes", counting)
    query.query_gt(z, th, options=QueryOptions(pipeline=pipeline))
    assert not calls  # zero copy by default
    r = query.query_gt(z, th, options=QueryOptions(pipeline=pipeline, zero_copy=False))
    same_answer(r, ref)
    assert len(calls) == r.payload_requests


def peak(z, th, limit, **kw):
    opts = QueryOptions(max_inflight_bytes=limit, **kw)
    return query.query_gt(z, th, options=opts)


@pytest.mark.parametrize("pipeline", (True, False))
def test_inflight_limit_holds(stored, pipeline):
    """The bytes held never exceed the limit, or one read when a read is larger.

    With a 1-byte limit every read runs alone, so the peak is the largest read.
    """
    a, _, z = stored
    th = float(np.quantile(a, 0.5))
    free = query.query_gt(z, th, options=QueryOptions(pipeline=pipeline))
    largest = peak(z, th, 1, pipeline=pipeline).peak_inflight_bytes
    assert 0 < largest < free.peak_inflight_bytes  # the limit does bite
    block = query.get_layout(z).block_size
    for limit in (block, 8 * block, 64 * block):
        r = peak(z, th, limit, pipeline=pipeline)
        same_answer(r, free)
        assert r.peak_inflight_bytes <= max(limit, largest)


def test_inflight_limit_lets_a_large_read_run_alone(stored):
    a, _, z = stored
    th = float(np.quantile(a, 0.5))
    opts = QueryOptions(merge_gap_blocks=10**6, max_inflight_bytes=1)
    r = query.query_gt(z, th, options=opts)
    assert r.count == query.query_gt(z, th).count
    # merged reads are many blocks long, each held alone
    assert 1 < r.peak_inflight_bytes <= r.payload_bytes_read


def test_inflight_limit_needs_layer_0(tmp_path):
    a = smooth_field(shape=(64, 16, 32), seed=9)
    g = zarr.open_group(str(tmp_path), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0, layers=(4.0, 8.0)),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a)
    with pytest.raises(ValueError, match="layer 0"):
        query.query_gt(z, 280.0, options=QueryOptions(max_inflight_bytes=10**6))
    largest = peak(z, 280.0, 1, layer=0).peak_inflight_bytes
    r = peak(z, 280.0, 64, layer=0)
    assert 0 < r.peak_inflight_bytes <= max(64, largest)


def test_aggregate_honours_zero_copy_and_limit(stored, monkeypatch):
    a, _, z = stored
    th = float(np.quantile(a, 0.5))
    ref = aggregate_gt(z, th)
    calls = []
    orig = cpu.Buffer.to_bytes
    monkeypatch.setattr(cpu.Buffer, "to_bytes", lambda s: calls.append(1) or orig(s))
    r = aggregate_gt(z, th, options=QueryOptions(zero_copy=False))
    assert (r.count, r.sum) == (ref.count, ref.sum)
    assert len(calls) == r.payload_requests
    monkeypatch.undo()
    largest = aggregate_gt(
        z, th, options=QueryOptions(max_inflight_bytes=1)
    ).peak_inflight_bytes
    assert 0 < largest < ref.peak_inflight_bytes
    block = query.get_layout(z).block_size
    r = aggregate_gt(z, th, options=QueryOptions(max_inflight_bytes=4 * block))
    assert (r.count, r.sum) == (ref.count, ref.sum)
    assert 0 < r.peak_inflight_bytes <= max(4 * block, largest)


def test_byte_budget_waits_and_records_the_peak():
    async def run():
        b = query.ByteBudget(10)
        held, peak = [], []

        async def task(n):
            await b.acquire(n)
            held.append(n)
            peak.append(sum(held))
            await asyncio.sleep(0.01)
            held.remove(n)
            await b.release(n)

        await asyncio.gather(*(task(n) for n in (6, 6, 3, 12, 1)))
        return b, max(peak)

    b, seen = asyncio.run(run())
    assert b.used == 0
    assert b.peak == seen == 12  # the 12-byte task ran alone; others stayed <= 10
    with pytest.raises(ValueError):
        QueryOptions(max_inflight_bytes=0)


@pytest.mark.parametrize("pipeline", (True, False))
def test_limit_counts_reads_waiting_for_a_decoder(stored, pipeline, monkeypatch):
    """Bytes read but not yet decoded stay within the limit too.

    Decoding is slowed down on one thread, so finished reads pile up in front
    of it. The test tracks those bytes itself, from when a read returns until
    its decode ends, and they must never exceed what the limit allows.
    """
    a, _, z = stored
    th = float(np.quantile(a, 0.9))
    largest = peak(z, th, 1, pipeline=pipeline).peak_inflight_bytes
    nat = query.native()
    lock = threading.Lock()
    alive = [0, 0]  # bytes now held, most ever held
    orig_read, orig_count = query.read_one, nat.count_offsets

    async def tracked_read(*args, **kwargs):
        data = await orig_read(*args, **kwargs)
        with lock:
            alive[0] += len(data)
            alive[1] = max(alive[1], alive[0])
        return data

    def slow_count(data, *args, **kwargs):
        time.sleep(0.002)
        try:
            return orig_count(data, *args, **kwargs)
        finally:
            with lock:
                alive[0] -= len(data)

    monkeypatch.setattr(query, "read_one", tracked_read)
    monkeypatch.setattr(nat, "count_offsets", slow_count)
    limit = 4 * query.get_layout(z).block_size
    r = peak(z, th, limit, pipeline=pipeline, threads=1)
    assert alive[0] == 0 and r.payload_requests > 8
    assert alive[1] <= max(limit, largest)


@pytest.mark.parametrize("q", (0.01, 0.5, 0.99))
def test_aggregate_paths_agree(stored, q, monkeypatch):
    """The pipelined aggregate, the one run in turn and the one reading the zone
    map as a Zarr array give the same answers and read the same blocks."""
    a, rec, z = stored
    th = float(np.quantile(a, q))
    from skipzfp import aggregate as agg

    calls = []
    real = agg._aggregate_pipelined

    async def spy(*args, **kwargs):
        calls.append(1)
        return await real(*args, **kwargs)

    monkeypatch.setattr(agg, "_aggregate_pipelined", spy)
    runs = [
        aggregate_gt(z, th, options=QueryOptions(**o), return_blocks=True)
        for o in ({}, {"pipeline": False}, {"meta_by_key": False})
    ]
    assert calls == [1]  # only the default pipelined
    m = rec > th
    for r in runs:
        assert r.count == int(m.sum())
        assert r.sum == pytest.approx(float(rec[m].astype(np.float64).sum()), rel=1e-12)
        assert r.original_count[0] <= int((a > th).sum()) <= r.original_count[1]
        assert (r.original_count, r.original_mean) == (
            runs[0].original_count,
            runs[0].original_mean,
        )
        assert (r.in_blocks, r.maybe_blocks, r.useful_payload_bytes) == (
            runs[0].in_blocks,
            runs[0].maybe_blocks,
            runs[0].useful_payload_bytes,
        )
        got = r.read_blocks[np.lexsort(r.read_blocks.T[::-1])]
        want = runs[0].read_blocks[np.lexsort(runs[0].read_blocks.T[::-1])]
        assert np.array_equal(got, want)


def test_meta_by_key_off_opens_the_zone_map(stored):
    """meta_by_key=False reads the zone map through zarr, with the same answer."""
    a, rec, z = stored
    th = float(np.quantile(a, 0.5))
    on = query.query_gt(z, th, options=QueryOptions(pipeline=False))
    off = query.query_gt(z, th, options=QueryOptions(meta_by_key=False))
    same_answer(off, on)
    assert off.metadata_bytes_read == on.metadata_bytes_read


@pytest.mark.parametrize("pipeline", (True, False))
def test_trace_covers_every_read(stored, pipeline):
    a, rec, z = stored
    th = float(np.quantile(a, 0.5))
    r = query.query_gt(z, th, options=QueryOptions(pipeline=pipeline, trace=True))
    assert len(r.trace["reads"]) == r.payload_requests
    assert len(r.trace["meta"]) == (r.metadata_requests if pipeline else 1)
    for issued, arrived, decoded in r.trace["reads"]:
        assert 0 <= issued <= arrived <= decoded <= r.total_seconds
    assert query.query_gt(z, th).trace is None
