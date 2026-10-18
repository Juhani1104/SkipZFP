"""Filtered aggregates: COUNT, SUM and AVG of the values in a range.

A COUNT never reads an IN block, since all of its values match, but SUM and AVG
need those values, so an aggregate reads the IN blocks as well as the MAYBE
blocks and skips only the OUT blocks.

The exact answer is for the decoded values, as with the count queries. Each
original value lies within its unit's error bound of its decoded value, so the
same pass also bounds COUNT, SUM and AVG over the original values.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from zarr.core.sync import sync

from .query import (
    QueryOptions,
    _MergedRange,
    _Range,
    chunk_coords,
    full_key,
    get_layout,
    merge_ranges,
    native,
    open_skipzfp,
    plan_meta,
    pool,
    range_bounds,
    read_meta,
    read_one,
)

# unit roundoff of a double: one rounding changes a value by at most this fraction
_U = 2.0**-53


@dataclass(frozen=True)
class AggregateResult:
    """Answer of a filtered aggregate, plus where its bytes and time went.

    count, sum and mean are exact for the decoded values. The original_*
    fields are guaranteed bounds on the same aggregate over the original
    values, which can differ because ZFP compression is lossy. original_mean
    is None when no original value can match, and when original_count[0] is
    0 the original selection may be empty, so its mean may not exist.

    The block and byte fields mean the same as in QueryResult, except that
    useful_payload_bytes counts the IN blocks too, since they are read.

    Attributes:
        count: Number of matching decoded values.
        sum: Their sum, computed in double precision.
        mean: sum / count, or None when count is 0.
        original_count: (lo, hi) bounds on the number of matching original values.
        original_sum: (lo, hi) bounds on their sum.
        original_mean: (lo, hi) bounds on their mean, or None.
        in_blocks: Blocks whose values all match, read and summed.
        out_blocks: Blocks with no matching value, skipped without reading.
        maybe_blocks: Blocks that partly match, read and filtered.
        total_blocks: Blocks in the whole array.
        metadata_bytes_read: Bytes read from the zone map.
        payload_bytes_read: Bytes read from the compressed blocks.
        metadata_requests: Objects read from the zone map.
        payload_requests: Range reads issued for the compressed blocks.
        useful_payload_bytes: Bytes of the IN and MAYBE blocks themselves.
        payload_overread_bytes: Extra bytes read because nearby reads were merged.
        metadata_read_seconds: Time to read the zone map.
        planning_seconds: Time to classify blocks as IN, OUT or MAYBE.
        payload_read_seconds: Time to read the blocks, including decoding,
            which overlaps with the reads.
        decode_seconds: Time spent decoding, summed over reads running in
            parallel, so it can exceed the wall time.
        total_seconds: Wall time of the whole query.
        read_blocks: With return_blocks=True, the (chunk, block) id of every
            block read, one row each in reading order, where the chunk id
            counts chunks in C order and the block id is the block's position
            in its chunk. None otherwise.
    """

    count: int
    sum: float
    mean: float | None
    original_count: tuple[int, int]
    original_sum: tuple[float, float]
    original_mean: tuple[float, float] | None
    in_blocks: int
    out_blocks: int
    maybe_blocks: int
    total_blocks: int
    metadata_bytes_read: int
    payload_bytes_read: int
    metadata_requests: int
    payload_requests: int
    useful_payload_bytes: int
    payload_overread_bytes: int
    metadata_read_seconds: float
    planning_seconds: float
    payload_read_seconds: float
    decode_seconds: float
    total_seconds: float
    read_blocks: np.ndarray | None = None

    @property
    def bytes_read(self) -> int:
        """Total bytes read from the zone map and the compressed blocks."""
        return self.metadata_bytes_read + self.payload_bytes_read

    @property
    def range_requests(self) -> int:
        """Total read requests for the zone map and the compressed blocks."""
        return self.metadata_requests + self.payload_requests


def block_eps(
    meta: np.ndarray,
    meta_size: int,
    unit_grid: Sequence[int],
    subs: Sequence[int],
    grid_shape: Sequence[int],
    blocks_per_unit: int,
    chunk_ids: np.ndarray,
    block_ids: np.ndarray,
) -> np.ndarray:
    """Return the error bound of the unit holding each (chunk, block).

    meta holds single-layer records, as plan_meta returns them, so the error
    bound is the float at bytes 8 to 12 of each record. Blocks are numbered
    unit by unit inside a chunk, as in the C planner.
    """
    rec = np.ascontiguousarray(meta, dtype=np.uint8).reshape(-1, meta_size)
    eps = np.ascontiguousarray(rec[:, 8:12]).view(np.float32).reshape(-1)

    _, gy, gz = (int(g) for g in grid_shape)
    sx, sy, sz = (int(s) for s in subs)
    _, uy, uz = (int(u) for u in unit_grid)
    c = chunk_ids.astype(np.int64)
    sub = block_ids.astype(np.int64) // blocks_per_unit
    ix = (c // (gy * gz)) * sx + sub // (sy * sz)
    iy = ((c // gz) % gy) * sy + (sub // sz) % sy
    iz = (c % gz) * sz + sub % sz
    return eps[(ix * uy + iy) * uz + iz]


async def stream_aggregate(
    store: Any,
    merged: Sequence[_MergedRange],
    blocks: np.ndarray,
    eps: np.ndarray,
    in_block: np.ndarray,
    block_size: int,
    rate: float,
    block_dim: int,
    lo: float,
    hi: float,
    concurrency: int,
    threads: int,
) -> tuple[list[tuple[Any, int]], int, float]:
    """Read each merged range and aggregate lo < x <= hi as soon as it arrives.

    It returns the totals of every read with its number of blocks, the bytes
    read, and the decode seconds summed over all reads.
    """
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    def agg(data: np.ndarray, req: _MergedRange) -> tuple[Any, float]:
        t0 = time.perf_counter()
        sl = slice(req.item_start, req.item_end)
        rows = blocks[sl].astype(np.int64) - req.first_block
        out = native().aggregate_offsets(
            data,
            (rows * block_size).astype(np.uint64),
            eps[sl],
            in_block[sl],
            block_size,
            rate,
            block_dim,
            lo,
            hi,
        )
        return out, time.perf_counter() - t0

    async def one(req: _MergedRange) -> tuple[Any, int, int, float]:
        data = await read_one(
            store, _Range(key=req.key, start=req.start, end=req.end), sem
        )
        out, sec = await loop.run_in_executor(pool(threads), agg, data, req)
        return out, req.item_end - req.item_start, len(data), sec

    parts = await asyncio.gather(*(one(r) for r in merged))
    return (
        [(p[0], p[1]) for p in parts],
        sum(p[2] for p in parts),
        sum(p[3] for p in parts),
    )


def _down(x: float) -> float:
    return math.nextafter(x, -math.inf)


def _up(x: float) -> float:
    return math.nextafter(x, math.inf)


def combine(parts: Sequence[tuple[Any, int]], values_per_block: int) -> dict[str, Any]:
    """Merge the totals of every read into the exact and the original answers.

    Each read sums its terms one by one in double precision, which is off by
    at most (n + 2) * u * sum(|term|) for n terms, so the original bounds are
    widened by that much and then rounded outward.
    """
    count = sum(int(p.rec_count) for p, _ in parts)
    total = math.fsum(p.rec_sum for p, _ in parts)
    sure = sum(int(p.sure_count) for p, _ in parts)
    maybe = sum(int(p.maybe_count) for p, _ in parts)
    slack = math.fsum((n * values_per_block + 2) * _U * p.abs_total for p, n in parts)
    sure_lo = math.fsum(p.sure_sum_lo for p, _ in parts) - slack
    sure_hi = math.fsum(p.sure_sum_hi for p, _ in parts) + slack
    maybe_neg = math.fsum(p.maybe_sum_neg for p, _ in parts)
    maybe_pos = math.fsum(p.maybe_sum_pos for p, _ in parts)
    maybe_min = min((p.maybe_min for p, _ in parts), default=math.inf)
    maybe_max = max((p.maybe_max for p, _ in parts), default=-math.inf)

    # A matching original value is a sure one, or a maybe one that turns out
    # to match. Including a maybe value moves the sum by at least min(0, low)
    # and at most max(0, high). With k of the maybe values included, each at
    # least maybe_min, the mean is at least (sure_lo + k * maybe_min) / (sure + k),
    # which is monotone in k, so its smallest value is at k = 0 or k = maybe;
    # the upper bound is the mirror image.
    if sure:
        lo_mean = _down(sure_lo / sure)
        hi_mean = _up(sure_hi / sure)
        if maybe:
            n = sure + maybe
            lo_mean = min(lo_mean, _down(_down(sure_lo + _down(maybe * maybe_min)) / n))
            hi_mean = max(hi_mean, _up(_up(sure_hi + _up(maybe * maybe_max)) / n))
        mean = (lo_mean, hi_mean)
    elif maybe:
        mean = (maybe_min, maybe_max)
    else:
        mean = None

    return {
        "count": count,
        "sum": total,
        "mean": total / count if count else None,
        "original_count": (sure, sure + maybe),
        "original_sum": (_down(sure_lo + maybe_neg), _up(sure_hi + maybe_pos)),
        "original_mean": mean,
    }


async def _aggregate_async(
    arr: zarr.Array,
    lo: float,
    hi: float,
    options: QueryOptions | None = None,
    return_blocks: bool = False,
) -> AggregateResult:
    """Aggregate the values with lo < x <= hi. Either bound may be infinite."""
    if not isinstance(arr, zarr.Array):
        raise TypeError("the aggregate expects an opened zarr.Array")

    opts = options or QueryOptions()
    t0 = time.perf_counter()

    lt = get_layout(arr)
    if opts.layer not in (None, 0) or (opts.layer is None and len(lt.layers) > 1):
        raise ValueError(
            "aggregates read layer 0 only; pass layer=0 for a layered array"
        )
    rate = lt.layers[0]
    block_size = lt.layer_bytes[0]
    keys = [
        full_key(arr, arr.metadata.encode_chunk_key(c))
        for c in chunk_coords(lt.grid_shape)
    ]

    # 1. read the zone map
    t1 = time.perf_counter()
    meta, meta_bytes, meta_requests = await read_meta(arr, lt, opts.request_concurrency)
    meta_read_sec = time.perf_counter() - t1

    # 2. plan: list the IN and MAYBE blocks, with each one's error bound
    t1 = time.perf_counter()
    single = plan_meta(meta, lt, 0)
    meta_size = lt.meta_size - 4 * (len(lt.layers) - 1)
    chunk_ids, block_ids, states, n_in, n_out = await asyncio.to_thread(
        native().plan_read,
        single,
        meta_size,
        lt.unit_grid,
        lt.unit_grid_per_chunk,
        lt.blocks_per_unit,
        lo,
        hi,
        opts.threads,
    )
    eps = block_eps(
        single,
        meta_size,
        lt.unit_grid,
        lt.unit_grid_per_chunk,
        lt.grid_shape,
        lt.blocks_per_unit,
        chunk_ids,
        block_ids,
    )
    in_block = (states == 1).astype(np.uint8)
    plan_sec = time.perf_counter() - t1

    # 3. read and aggregate the listed blocks; the planner lists them in chunk
    # order, so merge_ranges keeps that order and eps and in_block still line up
    merged, blocks = merge_ranges(
        keys, chunk_ids, block_ids, block_size, opts.merge_gap_blocks
    )
    if len(blocks) and blocks is not block_ids:
        raise RuntimeError("the planner must list blocks in chunk order")
    t1 = time.perf_counter()
    parts, payload_bytes, decode_sec = await stream_aggregate(
        arr.store_path.store,
        merged,
        blocks,
        eps,
        in_block,
        block_size,
        rate,
        lt.block_dim,
        lo,
        hi,
        opts.request_concurrency,
        opts.threads,
    )
    payload_read_sec = time.perf_counter() - t1

    useful_bytes = len(blocks) * block_size
    return AggregateResult(
        **combine(parts, lt.values_per_block),
        in_blocks=n_in,
        out_blocks=n_out,
        maybe_blocks=len(blocks) - n_in,
        total_blocks=len(keys) * lt.blocks_per_chunk,
        metadata_bytes_read=meta_bytes,
        payload_bytes_read=payload_bytes,
        metadata_requests=meta_requests,
        payload_requests=len(merged),
        useful_payload_bytes=useful_bytes,
        payload_overread_bytes=payload_bytes - useful_bytes,
        metadata_read_seconds=meta_read_sec,
        planning_seconds=plan_sec,
        payload_read_seconds=payload_read_sec,
        decode_seconds=decode_sec,
        total_seconds=time.perf_counter() - t0,
        read_blocks=np.column_stack((chunk_ids, block_ids)) if return_blocks else None,
    )


def _run(
    source: Any,
    lo: float,
    hi: float,
    array_path: str,
    storage_options: Mapping[str, Any] | None,
    options: QueryOptions | None,
    return_blocks: bool,
) -> AggregateResult:
    arr = open_skipzfp(source, array_path=array_path, storage_options=storage_options)
    return sync(_aggregate_async(arr, lo, hi, options, return_blocks))


def aggregate_gt(
    source: Any,
    threshold: float,
    *,
    inclusive: bool = False,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    options: QueryOptions | None = None,
    return_blocks: bool = False,
) -> AggregateResult:
    """COUNT, SUM and AVG of the values greater than threshold.

    The zone map shows which blocks lie entirely below the threshold, and
    those are skipped. Every other block is read and decoded, including the
    blocks that lie entirely above it, since their values are needed for the
    sum.

    Args:
        source: An opened zarr.Array, or a local path or URL to open.
        threshold: Aggregate values strictly greater than this.
        inclusive: Use x >= threshold instead of x > threshold.
        array_path, storage_options, options: Same as in :func:`query_gt`.
            options.layer must be None or 0, and 0 for a layered array.
        return_blocks: Also return the id of every block read, see
            AggregateResult.read_blocks.

    Returns:
        AggregateResult with the exact answer for the decoded values, bounds
        for the original values, and the bytes, requests and seconds spent.

    Raises:
        ValueError: If threshold is NaN, options.layer is not 0 where needed,
            or in any case where :func:`query_gt` raises it.

    Example:
        >>> r = aggregate_gt("data/t2m.zarr", 295.0)
        >>> print(r.mean, r.original_mean)
    """
    lo, hi = range_bounds(threshold, None, lo_inclusive=inclusive)
    return _run(source, lo, hi, array_path, storage_options, options, return_blocks)


def aggregate_lt(
    source: Any,
    threshold: float,
    *,
    inclusive: bool = False,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    options: QueryOptions | None = None,
    return_blocks: bool = False,
) -> AggregateResult:
    """COUNT, SUM and AVG of the values less than threshold.

    It is the mirror image of :func:`aggregate_gt` and takes the same arguments.
    """
    lo, hi = range_bounds(None, threshold, hi_inclusive=inclusive)
    return _run(source, lo, hi, array_path, storage_options, options, return_blocks)


def aggregate_range(
    source: Any,
    lo: float,
    hi: float,
    *,
    lo_inclusive: bool = False,
    hi_inclusive: bool = True,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    options: QueryOptions | None = None,
    return_blocks: bool = False,
) -> AggregateResult:
    """COUNT, SUM and AVG of the values between two bounds, by default lo < x <= hi.

    It works like :func:`aggregate_gt`, and its bounds work as in :func:`query_range`.

    Raises:
        TypeError: If lo or hi is None.
        ValueError: If lo or hi is NaN, if lo > hi, or in any case where
            :func:`aggregate_gt` raises it.
    """
    if lo is None or hi is None:
        raise TypeError(
            "aggregate_range needs both bounds; use aggregate_gt or aggregate_lt"
        )
    a, b = range_bounds(lo, hi, lo_inclusive, hi_inclusive)
    return _run(source, a, b, array_path, storage_options, options, return_blocks)
