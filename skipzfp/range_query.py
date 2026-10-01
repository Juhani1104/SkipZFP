"""Range queries, lo < x <= hi, kept apart from query_gt until the two are merged."""

from __future__ import annotations

import asyncio
import collections
import ctypes
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zarr
from zarr.core.sync import sync

from .query import (
    QueryResult,
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
    read_meta,
    read_one,
    read_ranges,
)


class _RangeNative:
    """ctypes bindings for csrc/range.c, the range versions of query._Native."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib

        self.lib.szfp_plan_range_chunks.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            *[ctypes.c_size_t] * 8,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_plan_range_chunks.restype = ctypes.c_int

        self.lib.szfp_count_range_blocks.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_count_range_blocks.restype = ctypes.c_int

        self.lib.szfp_count_offsets_range.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_count_offsets_range.restype = ctypes.c_int

    @staticmethod
    def check(code: int, name: str) -> None:
        if code != 0:
            raise RuntimeError(f"{name} failed with code {code}")

    def plan_chunks(
        self,
        meta: np.ndarray,
        meta_size: int,
        unit_grid: tuple[int, int, int],
        subs: tuple[int, int, int],
        blocks_per_unit: int,
        lo: float,
        hi: float,
        threads: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Blocks for lo < x <= hi, in chunk order: MAYBE ids plus IN/OUT counts."""
        src = np.ascontiguousarray(meta, dtype=np.uint8)
        cap = int(np.prod(unit_grid)) * blocks_per_unit
        chunk_ids = np.empty(cap, dtype=np.uint32)
        block_ids = np.empty(cap, dtype=np.uint32)

        n_maybe = ctypes.c_size_t()
        n_in = ctypes.c_size_t()
        n_out = ctypes.c_size_t()

        code = self.lib.szfp_plan_range_chunks(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            meta_size,
            *unit_grid,
            *subs,
            blocks_per_unit,
            lo,
            hi,
            threads,
            chunk_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            cap,
            ctypes.byref(n_maybe),
            ctypes.byref(n_in),
            ctypes.byref(n_out),
        )
        self.check(code, "szfp_plan_range_chunks")

        n = int(n_maybe.value)
        return chunk_ids[:n], block_ids[:n], int(n_in.value), int(n_out.value)

    def count(
        self,
        blocks: np.ndarray,
        n_block: int,
        block_size: int,
        rate: float,
        block_dim: int,
        lo: float,
        hi: float,
        threads: int,
    ) -> int:
        if n_block == 0:
            return 0

        src = np.ascontiguousarray(blocks, dtype=np.uint8)
        out = ctypes.c_size_t()

        code = self.lib.szfp_count_range_blocks(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            n_block,
            block_size,
            rate,
            block_dim,
            lo,
            hi,
            threads,
            ctypes.byref(out),
        )
        self.check(code, "szfp_count_range_blocks")
        return int(out.value)

    def count_offsets(
        self,
        data: bytes,
        offsets: np.ndarray,
        block_size: int,
        rate: float,
        block_dim: int,
        lo: float,
        hi: float,
    ) -> int:
        offs = np.ascontiguousarray(offsets, dtype=np.uint64)
        out = ctypes.c_size_t()

        code = self.lib.szfp_count_offsets_range(
            data,
            len(data),
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(offs),
            block_size,
            rate,
            block_dim,
            lo,
            hi,
            ctypes.byref(out),
        )
        self.check(code, "szfp_count_offsets_range")
        return int(out.value)


_RANGE_NATIVE: _RangeNative | None = None


def range_native() -> _RangeNative:
    global _RANGE_NATIVE

    if _RANGE_NATIVE is None:
        _RANGE_NATIVE = _RangeNative(native().lib)

    return _RANGE_NATIVE


async def stream_count_range(
    store: Any,
    merged: Sequence[_MergedRange],
    blocks: np.ndarray,
    block_size: int,
    rate: float,
    block_dim: int,
    lo: float,
    hi: float,
    concurrency: int,
) -> tuple[int, int, float]:
    """Range version of query.stream_count."""
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    def count(data: bytes, offs: np.ndarray) -> tuple[int, float]:
        t0 = time.perf_counter()
        n = range_native().count_offsets(
            data, offs, block_size, rate, block_dim, lo, hi
        )
        return n, time.perf_counter() - t0

    async def one(req: _MergedRange) -> tuple[int, int, float]:
        data = await read_one(
            store, _Range(key=req.key, start=req.start, end=req.end), sem
        )
        rows = blocks[req.item_start : req.item_end].astype(np.int64) - req.first_block
        n, sec = await loop.run_in_executor(
            pool(), count, data, (rows * block_size).astype(np.uint64)
        )
        return n, len(data), sec

    parts = await asyncio.gather(*(one(r) for r in merged))
    return sum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[2] for p in parts)


async def _query_range_async(
    arr: zarr.Array,
    lo: float,
    hi: float,
    *,
    threads: int = 0,
    request_concurrency: int = 32,
    request_batch_size: int | None = None,
    merge_gap_blocks: int = 0,
    layer: int | None = None,
    max_chunk_requests: int = 1,
) -> QueryResult:
    """Count values with lo < x <= hi. Either bound may be infinite."""
    # TODO: mirrors query_gt_async line for line, with lo/hi in place of the
    # threshold. See the comments there. Share the code once range is merged.
    if not isinstance(arr, zarr.Array):
        raise TypeError("query_range_async expects an opened zarr.Array")

    if request_concurrency <= 0:
        raise ValueError("request_concurrency must be greater than 0")
    if request_batch_size is None:
        request_batch_size = max(request_concurrency, request_concurrency * 4)
    if request_batch_size <= 0:
        raise ValueError("request_batch_size must be greater than 0")

    t0 = time.perf_counter()

    lt = get_layout(arr)
    layer = len(lt.layers) - 1 if layer is None else layer
    if not 0 <= layer < len(lt.layers):
        raise ValueError(f"layer must be in [0, {len(lt.layers) - 1}]")
    rate = lt.layers[layer]
    block_size = int(8 * rate)
    coords = chunk_coords(lt.grid_shape)
    keys = [full_key(arr, arr.metadata.encode_chunk_key(coord)) for coord in coords]
    store = arr.store_path.store

    t1 = time.perf_counter()
    meta, meta_bytes, meta_requests = await read_meta(arr, lt)
    meta_read_sec = time.perf_counter() - t1

    t1 = time.perf_counter()
    chunk_ids, block_ids, n_in, n_out = await asyncio.to_thread(
        range_native().plan_chunks,
        plan_meta(meta, lt, layer),
        lt.meta_size - 4 * (len(lt.layers) - 1),
        lt.unit_grid,
        tuple(c // u for c, u in zip(lt.chunk_shape, lt.unit_shape)),
        lt.blocks_per_unit,
        lo,
        hi,
        threads,
    )
    plan_sec = time.perf_counter() - t1

    per_layer = [
        merge_ranges(
            keys,
            chunk_ids,
            block_ids,
            lt.layer_bytes[j],
            merge_gap_blocks,
            lt.layer_starts[j],
        )
        for j in range(layer + 1)
    ]
    sorted_blocks = per_layer[0][1]
    sorted_chunks = chunk_ids[np.lexsort((block_ids, chunk_ids))] if layer > 0 else None
    cols = np.cumsum((0, *lt.layer_bytes[: layer + 1]))

    dense: set[str] = set()
    if layer > 0:
        per_key = collections.Counter(
            req.key for merged, _ in per_layer for req in merged
        )
        dense = {k for k, n in per_key.items() if n > max_chunk_requests}
    prefix_end = int(cols[-1]) * lt.blocks_per_chunk

    if layer == 0:
        merged0, sorted0 = per_layer[0]
        t1 = time.perf_counter()
        n_maybe_val, payload_bytes, decode_sec = await stream_count_range(
            store,
            merged0,
            sorted0,
            lt.layer_bytes[0],
            rate,
            lt.block_dim,
            lo,
            hi,
            request_concurrency,
        )
        payload_read_sec = time.perf_counter() - t1
        n_payload_reqs = len(merged0)
    else:
        payload_reqs = []
        fills = []
        for j, (merged, _) in enumerate(per_layer):
            for req in merged:
                if req.key not in dense:
                    payload_reqs.append(
                        _Range(key=req.key, start=req.start, end=req.end)
                    )
                    fills.append((j, req))
        chunk_of = {k: c for c, k in enumerate(keys)}
        for k in sorted(dense):
            payload_reqs.append(_Range(key=k, start=0, end=prefix_end))
            fills.append((None, k))

        t1 = time.perf_counter()
        payload_parts = await read_ranges(
            store,
            payload_reqs,
            concurrency=request_concurrency,
            batch_size=request_batch_size,
        )
        payload_read_sec = time.perf_counter() - t1

        out = np.empty((len(sorted_blocks), int(cols[-1])), dtype=np.uint8)
        for data, (j, item) in zip(payload_parts, fills):
            buf = np.frombuffer(data, dtype=np.uint8)
            if j is not None:
                rows = (
                    sorted_blocks[item.item_start : item.item_end].astype(np.int64)
                    - item.first_block
                )
                out[item.item_start : item.item_end, cols[j] : cols[j + 1]] = (
                    buf.reshape(-1, lt.layer_bytes[j])[rows]
                )
            else:
                c = chunk_of[item]
                b_lo, b_hi = np.searchsorted(sorted_chunks, [c, c + 1])
                ids = sorted_blocks[b_lo:b_hi].astype(np.int64)
                for jj in range(layer + 1):
                    seg = buf[
                        lt.layer_starts[jj] : lt.layer_starts[jj]
                        + lt.blocks_per_chunk * lt.layer_bytes[jj]
                    ]
                    out[b_lo:b_hi, cols[jj] : cols[jj + 1]] = seg.reshape(
                        -1, lt.layer_bytes[jj]
                    )[ids]
        blocks = out.reshape(-1)

        t1 = time.perf_counter()
        n_maybe_val = await asyncio.to_thread(
            range_native().count,
            blocks,
            len(sorted_blocks),
            block_size,
            rate,
            lt.block_dim,
            lo,
            hi,
            threads,
        )
        decode_sec = time.perf_counter() - t1
        payload_bytes = sum(len(x) for x in payload_parts)
        n_payload_reqs = len(payload_reqs)

    total_blocks = len(keys) * lt.blocks_per_chunk
    total_count = n_in * lt.values_per_block + n_maybe_val
    useful_bytes = len(sorted_blocks) * block_size

    return QueryResult(
        count=total_count,
        in_blocks=n_in,
        out_blocks=n_out,
        maybe_blocks=len(sorted_blocks),
        total_blocks=total_blocks,
        metadata_bytes_read=meta_bytes,
        payload_bytes_read=payload_bytes,
        metadata_requests=meta_requests,
        payload_requests=n_payload_reqs,
        useful_payload_bytes=useful_bytes,
        payload_overread_bytes=payload_bytes - useful_bytes,
        metadata_read_seconds=meta_read_sec,
        planning_seconds=plan_sec,
        payload_read_seconds=payload_read_sec,
        decode_seconds=decode_sec,
        total_seconds=time.perf_counter() - t0,
    )


def _below_f32(x: float) -> float:
    """Largest float32 strictly below x (-inf if there is none)."""
    f = np.float32(x)
    if float(f) >= x:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def range_bounds(
    lo: float | None,
    hi: float | None,
    lo_inclusive: bool = False,
    hi_inclusive: bool = True,
) -> tuple[float, float]:
    """Map a range on float32 values to the half-open (lo, hi] the planner uses.

    Decoded values are float32, so x >= a is exactly x > (largest float32 < a),
    and x < b is exactly x <= (largest float32 < b).

    Args:
        lo: Lower bound, or None for no lower bound.
        hi: Upper bound, or None for no upper bound.
        lo_inclusive: Use lo <= x instead of lo < x.
        hi_inclusive: Use x <= hi. Set it to False for x < hi.

    Returns:
        (a, b) such that a < x <= b selects the same float32 values. A
        missing bound becomes -inf or inf.

    Raises:
        ValueError: If lo or hi is NaN, or if lo > hi.
    """
    for name, v in (("lo", lo), ("hi", hi)):
        if v is not None and math.isnan(v):
            raise ValueError(f"{name} must not be NaN")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("lo must not exceed hi")
    if lo is None:
        a = -math.inf
    else:
        a = _below_f32(float(lo)) if lo_inclusive else float(lo)
    if hi is None:
        b = math.inf
    else:
        b = float(hi) if hi_inclusive else _below_f32(float(hi))
    return a, b


async def query_range_async(
    arr: zarr.Array,
    lo: float | None = None,
    hi: float | None = None,
    *,
    lo_inclusive: bool = False,
    hi_inclusive: bool = True,
    **kwargs: Any,
) -> QueryResult:
    """Async version of :func:`query_range` for an already opened array.

    Takes the same keyword arguments as :func:`query_range` except
    array_path and storage_options. Use it to run several queries
    concurrently.

    Raises:
        TypeError: If arr is not an opened zarr.Array.
    """
    a, b = range_bounds(lo, hi, lo_inclusive, hi_inclusive)
    return await _query_range_async(arr, a, b, **kwargs)


def query_range(
    source: Any,
    lo: float | None = None,
    hi: float | None = None,
    *,
    lo_inclusive: bool = False,
    hi_inclusive: bool = True,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    threads: int = 0,
    request_concurrency: int = 32,
    request_batch_size: int | None = None,
    merge_gap_blocks: int = 0,
    layer: int | None = None,
    max_chunk_requests: int = 1,
) -> QueryResult:
    """Count values in a range, by default lo < x <= hi.

    This function works like :func:`query_gt`. It uses the zone map to
    count or skip whole blocks, and decodes only the blocks that lie
    partly inside the range.

    The count equals what you get by decompressing the whole array and
    counting. It can differ slightly from counting the original data,
    because ZFP compression is lossy.

    Args:
        source: An opened zarr.Array, or a local path or URL to open.
        lo: Lower bound, or None for no lower bound.
        hi: Upper bound, or None for no upper bound.
        lo_inclusive: Use lo <= x instead of lo < x.
        hi_inclusive: Use x <= hi. Set it to False for x < hi.
        array_path, storage_options, threads, request_concurrency,
        request_batch_size, merge_gap_blocks, layer, max_chunk_requests:
            Same as in :func:`query_gt`.

    Returns:
        QueryResult, as in :func:`query_gt`.

    Raises:
        ValueError: If lo or hi is NaN, if lo > hi, or in any case where
            :func:`query_gt` raises it.

    Example:
        >>> query_range("data/t2m.zarr", 290.0, 295.0).count  # 290 < x <= 295
        >>> query_range("data/t2m.zarr", hi=273.15, hi_inclusive=False).count
    """
    arr = open_skipzfp(source, array_path=array_path, storage_options=storage_options)
    a, b = range_bounds(lo, hi, lo_inclusive, hi_inclusive)
    return sync(
        _query_range_async(
            arr,
            a,
            b,
            threads=threads,
            request_concurrency=request_concurrency,
            request_batch_size=request_batch_size,
            merge_gap_blocks=merge_gap_blocks,
            layer=layer,
            max_chunk_requests=max_chunk_requests,
        )
    )
