from __future__ import annotations

import asyncio
import collections
import ctypes
import itertools
import math
import os
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Zstd
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from ._native import check, load_library
from .codec import (
    META_ATTR,
    META_PACKING_ATTR,
    META_SOURCE_ATTR,
    META_T_CHUNK_ATTR,
    PLANAR_ZSTD,
    block_rank,
    check_source,
    find_codec,
    meta_size,
    read_plane,
    unpack_meta,
)
from .codec import native as codec_native


class _Aggregate(ctypes.Structure):
    """SzAggregate in csrc/query.h: running totals of a filtered aggregate."""

    _fields_ = [
        ("rec_count", ctypes.c_uint64),
        ("rec_sum", ctypes.c_double),
        ("sure_count", ctypes.c_uint64),
        ("sure_sum_lo", ctypes.c_double),
        ("sure_sum_hi", ctypes.c_double),
        ("maybe_count", ctypes.c_uint64),
        ("maybe_sum_neg", ctypes.c_double),
        ("maybe_sum_pos", ctypes.c_double),
        ("maybe_min", ctypes.c_double),
        ("maybe_max", ctypes.c_double),
        ("abs_total", ctypes.c_double),
    ]


class _Native:
    """ctypes bindings for the query functions in csrc/query.c."""

    def __init__(self) -> None:
        self.lib = load_library()

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

        plan_args = self.lib.szfp_plan_range_chunks.argtypes
        self.lib.szfp_plan_range_chunks_masked.argtypes = [
            *plan_args[:12],
            ctypes.POINTER(ctypes.c_ubyte),
            *plan_args[12:],
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_plan_range_chunks_masked.restype = ctypes.c_int

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

        self.lib.szfp_count_range_blocks_masked.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_uint64),
            *self.lib.szfp_count_range_blocks.argtypes[1:],
        ]
        self.lib.szfp_count_range_blocks_masked.restype = ctypes.c_int

        self.lib.szfp_merge_ranges.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_merge_ranges.restype = ctypes.c_int

        self.lib.szfp_count_offsets_range.argtypes = [
            ctypes.c_void_p,
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

        count_args = self.lib.szfp_count_offsets_range.argtypes
        self.lib.szfp_count_offsets_range_masked.argtypes = [
            *count_args[:3],
            ctypes.POINTER(ctypes.c_uint64),
            *count_args[3:],
        ]
        self.lib.szfp_count_offsets_range_masked.restype = ctypes.c_int

        self.lib.szfp_plan_read_chunks.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            *[ctypes.c_size_t] * 8,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_plan_read_chunks.restype = ctypes.c_int

        read_args = self.lib.szfp_plan_read_chunks.argtypes
        self.lib.szfp_plan_read_chunks_masked.argtypes = [
            *read_args[:12],
            ctypes.POINTER(ctypes.c_ubyte),
            *read_args[12:],
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_plan_read_chunks_masked.restype = ctypes.c_int

        self.lib.szfp_aggregate_offsets_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(_Aggregate),
        ]
        self.lib.szfp_aggregate_offsets_range.restype = ctypes.c_int

        agg_args = self.lib.szfp_aggregate_offsets_range.argtypes
        self.lib.szfp_aggregate_offsets_range_masked.argtypes = [
            *agg_args[:3],
            ctypes.POINTER(ctypes.c_uint64),
            *agg_args[3:],
        ]
        self.lib.szfp_aggregate_offsets_range_masked.restype = ctypes.c_int

    def plan_chunks(
        self,
        meta: np.ndarray,
        meta_size: int,
        unit_grid: tuple[int, int, int],
        subs: tuple[int, int, int],
        blocks_per_unit: int,
        th: float,
        threads: int,
        hi: float = math.inf,
        nvalid: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """Classify blocks for th < x <= hi and return the MAYBE blocks in chunk order.

        It returns the (chunk, block) ids of the MAYBE blocks, plus the IN
        and OUT counts. hi comes last so callers written for x > th still work.
        With nvalid, each block's number of valid values as in _Mask, a block
        with none is OUT, and the number of valid values in the IN blocks is
        returned last.
        """
        src = np.ascontiguousarray(meta, dtype=np.uint8)
        cap = int(np.prod(unit_grid)) * blocks_per_unit
        chunk_ids = np.empty(cap, dtype=np.uint32)
        block_ids = np.empty(cap, dtype=np.uint32)

        n_maybe = ctypes.c_size_t()
        n_in = ctypes.c_size_t()
        n_out = ctypes.c_size_t()
        in_values = ctypes.c_size_t()
        nv = None if nvalid is None else np.ascontiguousarray(nvalid, dtype=np.uint8)

        code = self.lib.szfp_plan_range_chunks_masked(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            meta_size,
            *unit_grid,
            *subs,
            blocks_per_unit,
            th,
            hi,
            threads,
            None if nv is None else nv.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            chunk_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            cap,
            ctypes.byref(n_maybe),
            ctypes.byref(n_in),
            ctypes.byref(n_out),
            ctypes.byref(in_values),
        )
        check(code, "szfp_plan_range_chunks")

        n = int(n_maybe.value)
        out = (chunk_ids[:n], block_ids[:n], int(n_in.value), int(n_out.value))
        return out if nvalid is None else (*out, int(in_values.value))

    def count(
        self,
        blocks: np.ndarray,
        n_block: int,
        block_size: int,
        rate: float,
        block_dim: int,
        th: float,
        threads: int,
        hi: float = math.inf,
        masks: np.ndarray | None = None,
    ) -> int:
        """Decode n_block packed blocks and count values with th < x <= hi.

        With masks, one per block as block_bits returns them, only the valid
        values count.
        """
        if n_block == 0:
            return 0

        src = np.ascontiguousarray(blocks, dtype=np.uint8)
        out = ctypes.c_size_t()
        m = None if masks is None else np.ascontiguousarray(masks, dtype=np.uint64)

        code = self.lib.szfp_count_range_blocks_masked(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            None if m is None else m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            n_block,
            block_size,
            rate,
            block_dim,
            th,
            hi,
            threads,
            ctypes.byref(out),
        )
        check(code, "szfp_count_range_blocks")
        return int(out.value)

    def merge(
        self,
        chunk_ids: np.ndarray,
        block_ids: np.ndarray,
        gap: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Group sorted (chunk, block) ids into runs that can share one read.

        A run may skip up to gap blocks between two ids. For each run it
        returns the chunk, the first and last block, and the index of its
        first id.
        """
        n = len(block_ids)
        ch = np.ascontiguousarray(chunk_ids, dtype=np.uint32)
        bl = np.ascontiguousarray(block_ids, dtype=np.uint32)
        out_chunk = np.empty(n, dtype=np.uint32)
        out_first = np.empty(n, dtype=np.uint64)
        out_last = np.empty(n, dtype=np.uint64)
        out_item = np.empty(n, dtype=np.uint64)
        m = ctypes.c_size_t()

        code = self.lib.szfp_merge_ranges(
            ch.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            bl.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            n,
            gap,
            out_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            out_first.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_last.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_item.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.byref(m),
        )
        check(code, "szfp_merge_ranges")
        k = int(m.value)
        return out_chunk[:k], out_first[:k], out_last[:k], out_item[:k]

    def count_offsets(
        self,
        data: Any,
        offsets: np.ndarray,
        block_size: int,
        rate: float,
        block_dim: int,
        th: float,
        hi: float = math.inf,
        masks: np.ndarray | None = None,
    ) -> int:
        """Decode the blocks at the given byte offsets and count th < x <= hi.

        data is any contiguous bytes-like object, read in place without a copy.
        With masks, one per offset as block_bits returns them, only the valid
        values count.
        """
        src = np.frombuffer(data, dtype=np.uint8)
        offs = np.ascontiguousarray(offsets, dtype=np.uint64)
        out = ctypes.c_size_t()
        m = None if masks is None else np.ascontiguousarray(masks, dtype=np.uint64)

        code = self.lib.szfp_count_offsets_range_masked(
            src.ctypes.data,
            src.size,
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            None if m is None else m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(offs),
            block_size,
            rate,
            block_dim,
            th,
            hi,
            ctypes.byref(out),
        )
        check(code, "szfp_count_offsets_range")
        return int(out.value)

    def plan_read(
        self,
        meta: np.ndarray,
        meta_size: int,
        unit_grid: tuple[int, int, int],
        subs: tuple[int, int, int],
        blocks_per_unit: int,
        lo: float,
        hi: float,
        threads: int,
        eps: bool = False,
        nvalid: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """Classify blocks for lo < x <= hi and return the IN and MAYBE blocks.

        It returns the (chunk, block) ids of the listed blocks in chunk order,
        the state of each (1 = IN, 2 = MAYBE), and the IN and OUT counts. With
        eps=True it also returns each listed block's error bound as float32,
        rounded up. With nvalid, as for plan_chunks, a block with no valid
        value is OUT, and the number of valid values in the IN blocks comes
        last.
        """
        src = np.ascontiguousarray(meta, dtype=np.uint8)
        cap = int(np.prod(unit_grid)) * blocks_per_unit
        chunk_ids = np.empty(cap, dtype=np.uint32)
        block_ids = np.empty(cap, dtype=np.uint32)
        states = np.empty(cap, dtype=np.uint8)
        bounds = np.empty(cap if eps else 0, dtype=np.float32)

        n_read = ctypes.c_size_t()
        n_in = ctypes.c_size_t()
        n_out = ctypes.c_size_t()
        in_values = ctypes.c_size_t()
        nv = None if nvalid is None else np.ascontiguousarray(nvalid, dtype=np.uint8)

        code = self.lib.szfp_plan_read_chunks_masked(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            meta_size,
            *unit_grid,
            *subs,
            blocks_per_unit,
            lo,
            hi,
            threads,
            None if nv is None else nv.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            chunk_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            states.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if eps else None,
            cap,
            ctypes.byref(n_read),
            ctypes.byref(n_in),
            ctypes.byref(n_out),
            ctypes.byref(in_values),
        )
        check(code, "szfp_plan_read_chunks")

        n = int(n_read.value)
        out = (
            chunk_ids[:n],
            block_ids[:n],
            states[:n],
            int(n_in.value),
            int(n_out.value),
        )
        if eps:
            out = (*out, bounds[:n])
        return out if nvalid is None else (*out, int(in_values.value))

    def aggregate_offsets(
        self,
        data: Any,
        offsets: np.ndarray,
        eps: np.ndarray,
        in_block: np.ndarray,
        block_size: int,
        rate: float,
        block_dim: int,
        lo: float,
        hi: float,
        masks: np.ndarray | None = None,
    ) -> _Aggregate:
        """Decode the blocks at the given byte offsets and aggregate lo < x <= hi.

        eps is each block's error bound and in_block marks the IN blocks. data
        is any contiguous bytes-like object, read in place without a copy.
        With masks, one per offset as block_bits returns them, only the valid
        values count.
        """
        src = np.frombuffer(data, dtype=np.uint8)
        offs = np.ascontiguousarray(offsets, dtype=np.uint64)
        e = np.ascontiguousarray(eps, dtype=np.float32)
        flag = np.ascontiguousarray(in_block, dtype=np.uint8)
        out = _Aggregate()
        m = None if masks is None else np.ascontiguousarray(masks, dtype=np.uint64)

        code = self.lib.szfp_aggregate_offsets_range_masked(
            src.ctypes.data,
            src.size,
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            None if m is None else m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            e.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            flag.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            len(offs),
            block_size,
            rate,
            block_dim,
            lo,
            hi,
            ctypes.byref(out),
        )
        check(code, "szfp_aggregate_offsets_range")
        return out


_NATIVE: _Native | None = None
_POOLS: dict[int, ThreadPoolExecutor] = {}
_POOLS_LOCK = threading.Lock()


def pool(threads: int = 0) -> ThreadPoolExecutor:
    """Return the shared pool with this many workers that decodes reads as they arrive.

    threads = 0 uses one worker per core.
    """
    n = threads if threads > 0 else (os.cpu_count() or 4)
    with _POOLS_LOCK:
        if n not in _POOLS:
            _POOLS[n] = ThreadPoolExecutor(n)
        return _POOLS[n]


def native() -> _Native:
    """Return the shared _Native, loading the C library on first use."""
    global _NATIVE

    if _NATIVE is None:
        _NATIVE = _Native()

    return _NATIVE


# Unneeded bytes a read may span to join its neighbor when merge_gap_blocks is
# None. On Google Cloud Storage, merging adjacent blocks only was up to 49x
# slower than this gap and a 64 KB gap up to 2.2x slower, while gaps of 850 KB
# and 1 MB changed query time by at most about 20%.
DEFAULT_MERGE_GAP_BYTES = 256_000


@dataclass(frozen=True)
class QueryOptions:
    """Tuning knobs shared by query_gt, query_lt and query_range.

    The defaults suit most queries, so most callers never pass this.

    Attributes:
        threads: Threads for planning and decoding, where 0 uses all cores.
        request_concurrency: Maximum number of reads running at a time.
        request_batch_size: Number of reads issued per batch when layer > 0,
            4 * request_concurrency by default.
        merge_gap_blocks: Merge two reads in the same chunk when at most this
            many unneeded blocks lie between them, so larger values mean fewer
            requests but more bytes read. None, the default, allows
            DEFAULT_MERGE_GAP_BYTES (256 KB) of them at the rate being read,
            and 0 merges adjacent blocks only.
        layer: Precision layer to read for arrays written with layers, where 0
            is the lowest rate and None reads the full rate.
        max_chunk_requests: When layer > 0, a chunk that would need more than
            this many reads is fetched in a single read instead.
        pipeline: Plan and read each zone map object as soon as it arrives,
            when the layout allows it. False runs the steps in turn, to
            measure what the overlap is worth.
        zero_copy: Decode each read in place. False copies it to bytes first,
            on the event loop, as earlier versions did, to measure the copy.
        max_inflight_bytes: Most payload bytes held at once, counted from
            when a read starts until its blocks are decoded, or None for no
            limit. A read larger than the limit still runs, alone. It bounds
            the payload buffers only, not the zone map or the planner's
            arrays.
        meta_by_key: Read the zone map objects directly by key. False opens
            the zone map as a Zarr array first, reading its zarr.json, as for
            arrays written before t_chunk was recorded, to measure what the
            direct read is worth. The pipeline needs the direct read, so False
            also runs the steps in turn.
        trace: Record when each zone map object arrived and when each payload
            read started, arrived and was decoded, in QueryResult.trace, to
            see where the time of a query goes.

    Raises:
        ValueError: If a value is out of range.

    Example:
        >>> opts = QueryOptions(merge_gap_blocks=0)  # adjacent blocks only
        >>> query_gt("data/t2m.zarr", 295.0, options=opts)
    """

    threads: int = 0
    request_concurrency: int = 32
    request_batch_size: int | None = None
    merge_gap_blocks: int | None = None
    layer: int | None = None
    max_chunk_requests: int = 1
    pipeline: bool = True
    zero_copy: bool = True
    max_inflight_bytes: int | None = None
    meta_by_key: bool = True
    trace: bool = False

    def __post_init__(self) -> None:
        if self.threads < 0:
            raise ValueError("threads must be non-negative")
        if self.request_concurrency <= 0:
            raise ValueError("request_concurrency must be greater than 0")
        if self.request_batch_size is not None and self.request_batch_size <= 0:
            raise ValueError("request_batch_size must be greater than 0")
        if self.merge_gap_blocks is not None and self.merge_gap_blocks < 0:
            raise ValueError("merge_gap_blocks must be non-negative")
        if self.max_chunk_requests <= 0:
            raise ValueError("max_chunk_requests must be greater than 0")
        if self.max_inflight_bytes is not None and self.max_inflight_bytes <= 0:
            raise ValueError("max_inflight_bytes must be greater than 0")


@dataclass(frozen=True)
class QueryResult:
    """Answer of a query, plus where its bytes and time went.

    Every block is exactly one of IN, OUT or MAYBE, so
    in_blocks + out_blocks + maybe_blocks == total_blocks. The metadata_*
    fields refer to the zone map, and the payload_* fields to the compressed
    blocks.

    Attributes:
        count: Number of matching values.
        in_blocks: Blocks whose values all match, counted from the zone map.
        out_blocks: Blocks with no matching value, skipped without reading.
        maybe_blocks: Blocks that were read and decoded.
        total_blocks: Blocks in the whole array.
        metadata_bytes_read: Bytes read from the zone map.
        payload_bytes_read: Bytes read from the compressed blocks.
        metadata_requests: Objects read from the zone map.
        payload_requests: Range reads issued for the compressed blocks.
        useful_payload_bytes: Bytes of the MAYBE blocks themselves.
        payload_overread_bytes: Extra bytes read because nearby reads were
            merged or a whole chunk prefix was fetched.
        metadata_read_seconds: Time to read the zone map.
        planning_seconds: Time to classify blocks as IN, OUT or MAYBE.
        payload_read_seconds: Time to read the compressed blocks, which with layer 0
            also includes decoding because the two overlap.
        decode_seconds: Time spent decoding, which with layer 0 is summed
            over reads running in parallel and can exceed the wall time.
        total_seconds: Wall time of the whole query.
        peak_inflight_bytes: Most payload bytes reserved at once, see
            QueryOptions.max_inflight_bytes. A read reserves its size when it
            starts, before its bytes arrive, and frees it once its blocks are
            decoded, so this bounds the payload buffers from above; with no
            limit, reads still downloading count too, so it can be far above
            the memory actually used. With layer > 0 every read is held until
            all have arrived, so it equals payload_bytes_read.
        trace: With QueryOptions.trace and layer 0, seconds since the query
            started: "meta" holds when each zone map object arrived (one
            time for the whole zone map unless pipelined) and "reads" holds
            (issued, arrived, decoded) for each payload read, where a read
            issued may still wait for a free request slot. None otherwise.
    """

    count: int
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
    peak_inflight_bytes: int = 0
    trace: dict[str, list[Any]] | None = None

    @property
    def bytes_read(self) -> int:
        """Total bytes read from the zone map and the compressed blocks."""
        return self.metadata_bytes_read + self.payload_bytes_read

    @property
    def range_requests(self) -> int:
        """Total read requests for the zone map and the compressed blocks."""
        return self.metadata_requests + self.payload_requests


@dataclass(frozen=True)
class _Mask:
    """Where the valid values of a masked array are, block by block.

    Both arrays have shape (units along y, units along x, blocks per unit),
    with the blocks of a unit in storage order. Every unit along the first
    axis shares them, since the mask is the same at every step.
    """

    nvalid: np.ndarray  # uint8, the number of valid values of each block
    bits: np.ndarray  # uint64, value x + 4y + 16z of a block at bit x + 4y + 16z


def build_mask(
    plane: np.ndarray,
    unit: tuple[int, int, int],
    unit_grid: tuple[int, int, int],
    block_order: tuple[int, ...],
) -> _Mask:
    """Turn the (y, x) mask of valid values into per-block counts and bits."""
    by, bx = plane.shape[0] // 4, plane.shape[1] // 4
    # the 16 (y, x) positions of each block column, y-major, as bits 4y + x
    cells = plane.reshape(by, 4, bx, 4).transpose(0, 2, 1, 3).reshape(by, bx, 16)
    f16 = (cells.astype(np.uint64) << np.arange(16, dtype=np.uint64)).sum(axis=-1)
    bits = f16.astype(np.uint64) * np.uint64(0x0001000100010001)  # every z step
    nvalid = 4 * cells.sum(axis=-1)
    b = tuple(u // 4 for u in unit)
    gy, gx = unit_grid[1], unit_grid[2]
    rank = block_rank(unit, block_order)

    def per_unit(a: np.ndarray) -> np.ndarray:
        a = a.reshape(gy, b[1], gx, b[2]).transpose(0, 2, 1, 3)
        a = np.broadcast_to(a[:, :, None], (gy, gx, *b)).reshape(gy, gx, -1)
        out = np.empty_like(a)
        out[..., rank] = a  # C order to storage order
        return out

    return _Mask(
        nvalid=np.ascontiguousarray(per_unit(nvalid).astype(np.uint8)),
        bits=np.ascontiguousarray(per_unit(bits)),
    )


def block_bits(
    lt: _Layout, chunk_ids: np.ndarray, block_ids: np.ndarray
) -> np.ndarray | None:
    """Return the 64-bit mask of valid values of each (chunk, block), or None."""
    if lt.mask is None:
        return None
    g = lt.grid_shape
    s = lt.unit_grid_per_chunk
    c = chunk_ids.astype(np.int64)
    sub, bid = np.divmod(block_ids.astype(np.int64), lt.blocks_per_unit)
    iy = (c // g[2]) % g[1] * s[1] + (sub // s[2]) % s[1]
    ix = c % g[2] * s[2] + sub % s[2]
    return lt.mask.bits[iy, ix, bid]


def need_mask(lt: _Layout) -> None:
    """Raise ValueError for a masked array whose mask write_meta has not recorded."""
    if lt.masked and lt.mask is None:
        raise ValueError("the masked array has no recorded mask; run write_meta")


@dataclass(frozen=True)
class _Layout:
    """Shapes and byte sizes of a SkipZFP array, computed once per query."""

    chunk_shape: tuple[int, int, int]
    grid_shape: tuple[int, int, int]  # chunks along each axis
    rate: float
    block_dim: int
    data_size: int  # payload bytes per chunk
    meta_size: int  # bytes per record in the zone map
    block_size: int  # bytes per block at the full rate
    blocks_per_chunk: int
    values_per_block: int
    layers: tuple[float, ...]
    layer_bytes: tuple[int, ...]
    layer_starts: tuple[int, ...]  # byte offset of each layer in a chunk
    unit_shape: tuple[int, int, int]
    unit_grid: tuple[int, int, int]  # units along each axis of the array
    blocks_per_unit: int
    masked: bool = False
    mask: _Mask | None = None  # for masked arrays once write_meta has run

    @property
    def unit_grid_per_chunk(self) -> tuple[int, int, int]:
        """Units along each axis of one chunk."""
        return tuple(c // u for c, u in zip(self.chunk_shape, self.unit_shape))


@dataclass(frozen=True)
class _Range:
    """Byte range [start, end) of one stored object."""

    key: str
    start: int
    end: int


@dataclass(frozen=True)
class _MergedRange:
    """One read covering a run of MAYBE blocks in a chunk."""

    key: str
    start: int
    end: int
    first_block: int  # block id at byte offset start
    item_start: int  # this read covers sorted MAYBE blocks
    item_end: int  # item_start to item_end - 1


def open_skipzfp(
    source: Any,
    *,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
) -> zarr.Array:
    """Open an array for reading, or return it as is if already opened.

    It does not check that the array uses the skipzfp codec, since the
    query functions do that. The skipzfp codec only exists for Zarr v3, so
    only zarr.json is read, skipping the two Zarr v2 files zarr would
    otherwise request alongside it.

    Args:
        source: An opened zarr.Array, or a local path, URL or store that
            zarr can open.
        array_path: Path of the array inside the store.
        storage_options: Options for a cloud store, such as credentials,
            passed to fsspec when source is a URL.

    Returns:
        The array, opened read-only.

    Raises:
        ValueError: If array_path or storage_options is given together
            with an opened array.
        RuntimeError: If source is a URL and its fsspec backend is missing.
        TypeError: If source opens to something other than an array.
    """
    if isinstance(source, zarr.Array):
        if array_path:
            raise ValueError("array_path cannot be used with an opened zarr.Array")
        if storage_options:
            raise ValueError("storage_options cannot be used with an opened zarr.Array")
        return source

    store = str(source) if isinstance(source, Path) else source
    opts: dict[str, Any] = {
        "store": store,
        "path": array_path,
        "mode": "r",
        "zarr_format": 3,
    }

    if storage_options:
        opts["storage_options"] = dict(storage_options)

    try:
        arr = zarr.open_array(**opts)
    except ImportError as exc:
        raise RuntimeError(
            "cloud backend is not installed; install fsspec and the matching store"
        ) from exc

    if not isinstance(arr, zarr.Array):
        raise TypeError("source did not resolve to a Zarr array")

    return arr


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_layout(arr: zarr.Array) -> _Layout:
    """Verify that arr can be queried and return its byte layout."""
    if len(arr.shape) != 3:
        raise ValueError("query_gt requires a 3D array")

    meta = arr.metadata
    chunk_shape = tuple(int(x) for x in meta.chunk_grid.chunk_shape)

    if len(chunk_shape) != 3:
        raise ValueError("query_gt requires 3D chunks")

    if any(int(size) % chunk != 0 for size, chunk in zip(arr.shape, chunk_shape)):
        raise ValueError("query_gt requires array dimensions divisible by chunks")

    codec = find_codec(arr)
    rate = float(codec.rate)
    block_dim = int(codec.block_dim)

    if rate <= 0:
        raise ValueError("rate must be greater than 0")
    if block_dim <= 0:
        raise ValueError("block_dim must be greater than 0")
    if any(size % block_dim != 0 for size in chunk_shape):
        raise ValueError("each chunk dimension must be divisible by block_dim")

    grid_shape = tuple(
        ceildiv(int(size), chunk) for size, chunk in zip(arr.shape, chunk_shape)
    )

    blocks_axis = tuple(size // block_dim for size in chunk_shape)
    blocks_per_chunk = int(np.prod(blocks_axis))
    values_per_block = block_dim**3

    data_size, _ = codec_native().layout(
        chunk_shape,
        rate,
        block_dim,
    )

    if blocks_per_chunk <= 0:
        raise ValueError("invalid block layout")
    if data_size <= 0:
        raise ValueError("invalid fixed-rate layout")
    if data_size % blocks_per_chunk != 0:
        raise ValueError("compressed payload is not divisible by blocks")

    block_size = data_size // blocks_per_chunk

    unit = codec.unit(chunk_shape)
    blocks_per_unit = int(np.prod([u // block_dim for u in unit]))
    unit_grid = tuple(g * (c // u) for g, c, u in zip(grid_shape, chunk_shape, unit))
    plane = read_plane(arr) if codec.masked else None

    return _Layout(
        chunk_shape=chunk_shape,
        grid_shape=grid_shape,
        rate=rate,
        block_dim=block_dim,
        data_size=data_size,
        meta_size=meta_size(codec, unit),
        block_size=block_size,
        blocks_per_chunk=blocks_per_chunk,
        values_per_block=values_per_block,
        layers=codec.layers,
        layer_bytes=codec.layer_bytes,
        layer_starts=tuple(
            int(x) for x in np.cumsum((0, *codec.layer_bytes[:-1])) * blocks_per_chunk
        ),
        unit_shape=unit,
        unit_grid=unit_grid,
        blocks_per_unit=blocks_per_unit,
        masked=codec.masked,
        mask=None
        if plane is None
        else build_mask(plane, unit, unit_grid, codec.block_order),
    )


def chunk_coords(
    grid_shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Return the coordinates of every chunk, in C order."""
    return list(itertools.product(*(range(n) for n in grid_shape)))


def full_key(arr: zarr.Array, chunk_key: str) -> str:
    """Return the store key of a chunk, including the array's path."""
    prefix = arr.store_path.path.strip("/")
    return f"{prefix}/{chunk_key}" if prefix else chunk_key


def batched(items: Iterable[_Range], size: int) -> Iterable[list[_Range]]:
    """Yield lists of up to size items.

    This is itertools.batched, written out because that needs Python 3.12
    and the package still supports 3.11.
    """
    it = iter(items)

    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


async def read_one(
    store: Any,
    req: _Range,
    sem: asyncio.Semaphore,
    copy: bool = False,
) -> np.ndarray:
    """Read one byte range, as a uint8 view of the returned buffer.

    The view avoids copying the bytes, which on a busy event loop costs about
    as much as reading them. It raises an error if the object is missing or
    fewer bytes come back than requested.
    """
    async with sem:
        buf = await store.get(
            req.key,
            prototype=default_buffer_prototype(),
            byte_range=RangeByteRequest(req.start, req.end),
        )

    if buf is None:
        raise FileNotFoundError(req.key)

    # copy=True is the older behaviour, kept so its cost can be measured
    data = np.frombuffer(buf.to_bytes(), np.uint8) if copy else buf.as_numpy_array()
    want = req.end - req.start

    if len(data) != want:
        raise OSError(f"short range read for {req.key}: got {len(data)}, want {want}")

    return data


async def read_ranges(
    store: Any,
    reqs: Iterable[_Range],
    *,
    concurrency: int,
    batch_size: int,
    copy: bool = False,
) -> list[np.ndarray]:
    """Read byte ranges in batches.

    At most `concurrency` reads run at a time, and the results come back in
    the same order as reqs.
    """
    if concurrency <= 0:
        raise ValueError("request_concurrency must be greater than 0")
    if batch_size <= 0:
        raise ValueError("request_batch_size must be greater than 0")

    sem = asyncio.Semaphore(concurrency)
    out: list[np.ndarray] = []

    for batch in batched(reqs, batch_size):
        parts = await asyncio.gather(
            *(read_one(store, req, sem, copy) for req in batch)
        )
        out.extend(parts)

    return out


@dataclass(frozen=True)
class _MetaObjects:
    """Where the zone map objects of an array are and how big each one is."""

    prefix: str
    t_chunk: int  # units along the first axis in each object
    rows: int  # bytes per unit along the first axis
    count: int
    packed: bool = False  # objects are pack_meta's layout compressed with zstd
    units: int = 0  # units along the first axis of the array

    @property
    def size(self) -> int:
        return self.t_chunk * self.rows


def meta_objects(arr: zarr.Array, lt: _Layout) -> _MetaObjects | None:
    """Locate the zone map objects, or return None if arr does not record t_chunk."""
    path = arr.attrs.get(META_ATTR)
    if not path:
        raise ValueError("array has no metadata; run codec.write_meta first")
    source = arr.attrs.get(META_SOURCE_ATTR)
    if source:  # zone maps written before the source was recorded are trusted
        check_source(arr, source)

    t_chunk = arr.attrs.get(META_T_CHUNK_ATTR)
    if t_chunk is None:
        return None

    return _MetaObjects(
        prefix=path.strip("/"),
        t_chunk=int(t_chunk),
        rows=int(np.prod(lt.unit_grid[1:])) * lt.meta_size,
        count=ceildiv(lt.unit_grid[0], int(t_chunk)),
        packed=meta_packed(arr),
        units=lt.unit_grid[0],
    )


# decompresses packed zone map objects, which zarr's ZstdCodec wrote
_zstd = Zstd()


def meta_packed(arr: zarr.Array) -> bool:
    """Return whether the zone map is packed, or raise ValueError if unknown."""
    packing = arr.attrs.get(META_PACKING_ATTR)
    if packing not in (None, PLANAR_ZSTD):
        raise ValueError(f"unknown zone map packing {packing!r}")
    return packing == PLANAR_ZSTD


def unpack_records(data: np.ndarray, lt: _Layout, n_records: int) -> np.ndarray:
    """Undo pack_meta on the first n_records records of a packed object."""
    size = n_records * lt.meta_size
    out = data.copy()
    out[:size] = unpack_meta(
        data[:size], lt.meta_size, 8 + 4 * len(lt.layers), lt.blocks_per_unit
    ).reshape(-1)
    return out


async def read_meta_object(
    store: Any,
    mo: _MetaObjects,
    i: int,
    sem: asyncio.Semaphore,
    lt: _Layout | None = None,
    sizes: list[int] | None = None,
) -> np.ndarray:
    """Read zone map object i, as a read-only view of the returned bytes.

    A packed object is decompressed and unpacked, in a worker thread, which
    needs lt. sizes, if given, gets the object's stored size appended.
    """
    key = f"{mo.prefix}/c/{i}/0/0/0"
    async with sem:
        buf = await store.get(key, prototype=default_buffer_prototype())
    # write_meta writes every object, so a missing one is never all zeros
    if buf is None:
        raise FileNotFoundError(key)
    data = buf.as_numpy_array()
    if sizes is not None:
        sizes.append(len(data))
    if mo.packed:
        n = min(mo.t_chunk, mo.units - i * mo.t_chunk) * mo.rows // lt.meta_size
        data = await asyncio.to_thread(
            lambda: unpack_records(np.frombuffer(_zstd.decode(data), np.uint8), lt, n)
        )
    if len(data) != mo.size:
        raise ValueError(
            f"metadata object {i} has {len(data)} bytes, which does not match "
            f"the array layout ({mo.size} bytes)"
        )
    return data


async def read_meta(
    arr: zarr.Array, lt: _Layout, concurrency: int = 32, by_key: bool = True
) -> tuple[np.ndarray, int, int]:
    """Read the whole zone map, with its size in bytes and number of objects.

    The zone map's shape follows from the layout, so when arr records its
    t_chunk the objects are fetched directly by key, skipping the round trip
    to open the zone map's zarr.json. Arrays written before t_chunk was
    recorded fall back to opening it as a Zarr array, and so does
    by_key=False.
    """
    mo = meta_objects(arr, lt)
    if mo is None or not by_key:
        return await read_meta_zarr(arr, lt, arr.attrs[META_ATTR])

    store = arr.store_path.store
    sem = asyncio.Semaphore(concurrency)
    # each object is copied straight into its slot, the only copy made
    out = np.empty(mo.count * mo.size, dtype=np.uint8)

    sizes: list[int] = []

    async def one(i: int) -> None:
        out[i * mo.size : (i + 1) * mo.size] = await read_meta_object(
            store, mo, i, sem, lt, sizes
        )

    await asyncio.gather(*(one(i) for i in range(mo.count)))
    # the last object is padded to a full t_chunk, so cut it back to the grid
    return out[: lt.unit_grid[0] * mo.rows], sum(sizes), mo.count


async def read_meta_zarr(
    arr: zarr.Array, lt: _Layout, path: str
) -> tuple[np.ndarray, int, int]:
    """Read the zone map by opening it as a Zarr array, as :func:`read_meta` returns."""
    meta = await zarr.api.asynchronous.open_array(
        store=arr.store_path.store,
        path=path,
        mode="r",
        zarr_format=3,
    )

    if tuple(meta.shape) != (*lt.unit_grid, lt.meta_size):
        raise ValueError(f"metadata shape {meta.shape} does not match the array layout")

    data = np.ascontiguousarray(await meta.getitem(...), dtype=np.uint8).reshape(-1)
    n_obj = int(np.prod([ceildiv(s, c) for s, c in zip(meta.shape, meta.chunks)]))
    nbytes = data.nbytes
    if meta_packed(arr):
        nbytes = await meta.nbytes_stored()
        step = meta.chunks[0] * int(np.prod(lt.unit_grid[1:])) * lt.meta_size
        for s in range(0, len(data), step):
            part = data[s : s + step]
            part[...] = unpack_records(part, lt, len(part) // lt.meta_size)
    return data, nbytes, n_obj


def plan_meta(meta: np.ndarray, lt: _Layout, layer: int) -> np.ndarray:
    """Keep only one layer's error bound in each record of the zone map.

    The C planner reads records with a single error bound, so the bounds of
    the other layers are dropped. With a single layer the records already
    have that form, so meta is returned without a copy.
    """
    if len(lt.layers) == 1:
        return meta
    m = meta.reshape(-1, lt.meta_size)
    n_layer = len(lt.layers)
    eps = m[:, 8 + 4 * layer : 12 + 4 * layer]
    return np.ascontiguousarray(
        np.concatenate([m[:, :8], eps, m[:, 8 + 4 * n_layer :]], axis=1)
    )


class ByteBudget:
    """Limit the payload bytes held at once, and record the most ever held.

    A read reserves its size before it starts and gives it back once its
    blocks are decoded, so the bytes waiting for a decoder count too, not just
    the downloads in flight. With limit None it only records the peak.
    """

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.used = 0
        self.peak = 0
        self._cond = asyncio.Condition()

    async def acquire(self, n: int) -> None:
        async with self._cond:
            if self.limit is not None:
                # a read larger than the limit runs once nothing else is held
                await self._cond.wait_for(
                    lambda: self.used == 0 or self.used + n <= self.limit
                )
            self.used += n
            self.peak = max(self.peak, self.used)

    async def release(self, n: int) -> None:
        async with self._cond:
            self.used -= n
            self._cond.notify_all()


async def stream_count(
    store: Any,
    merged: Sequence[_MergedRange],
    blocks: np.ndarray,
    block_size: int,
    rate: float,
    block_dim: int,
    lo: float,
    hi: float,
    concurrency: int,
    threads: int,
    sem: asyncio.Semaphore | None = None,
    budget: ByteBudget | None = None,
    copy: bool = False,
    trace: list[tuple[float, float, float]] | None = None,
    t0: float = 0.0,
    masks: np.ndarray | None = None,
) -> tuple[int, int, float]:
    """Read each merged range and count lo < x <= hi as soon as it arrives.

    It returns the count, the bytes read and the decode seconds summed over
    all reads. Pass sem to share the concurrency limit with other reads, and
    budget to share a limit on the bytes held between reading and decoding.
    copy=True copies each read before decoding, as earlier versions did. With
    trace, each read appends when it started, arrived and was decoded, in
    seconds since t0. With masks, one per entry of blocks, only the valid
    values count.
    """
    loop = asyncio.get_running_loop()
    sem = sem or asyncio.Semaphore(concurrency)
    budget = budget or ByteBudget()

    def count(
        data: np.ndarray, offs: np.ndarray, m: np.ndarray | None
    ) -> tuple[int, float]:
        t0 = time.perf_counter()
        n = native().count_offsets(
            data, offs, block_size, rate, block_dim, lo, hi, masks=m
        )
        return n, time.perf_counter() - t0

    async def one(req: _MergedRange) -> tuple[int, int, float]:
        size = req.end - req.start
        await budget.acquire(size)
        try:
            start = time.perf_counter()
            data = await read_one(
                store, _Range(key=req.key, start=req.start, end=req.end), sem, copy
            )
            arrived = time.perf_counter()
            rows = (
                blocks[req.item_start : req.item_end].astype(np.int64) - req.first_block
            )
            m = None if masks is None else masks[req.item_start : req.item_end]
            n, sec = await loop.run_in_executor(
                pool(threads), count, data, (rows * block_size).astype(np.uint64), m
            )
            del data  # drop the buffer before giving its bytes back
        finally:
            await budget.release(size)
        if trace is not None:
            trace.append((start - t0, arrived - t0, time.perf_counter() - t0))
        return n, size, sec

    parts = await asyncio.gather(*(one(r) for r in merged))
    return sum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[2] for p in parts)


def gap_blocks(opts: QueryOptions, block_bytes: int) -> int:
    """Merge gap of opts in blocks, for reads of block_bytes per block."""
    if opts.merge_gap_blocks is not None:
        return opts.merge_gap_blocks
    return DEFAULT_MERGE_GAP_BYTES // block_bytes


def merge_ranges(
    keys: Sequence[str],
    chunk_ids: np.ndarray,
    block_ids: np.ndarray,
    block_size: int,
    merge_gap_blocks: int,
    base: int = 0,
) -> tuple[list[_MergedRange], np.ndarray]:
    """Turn MAYBE blocks into merged byte-range reads for one layer.

    base is the byte offset of the layer in each chunk. It returns the reads
    and the block ids sorted in chunk order, which the reads index into.
    """
    if merge_gap_blocks < 0:
        raise ValueError("merge_gap_blocks must be non-negative")

    n = len(block_ids)
    if n == 0:
        return [], np.empty(0, dtype=np.uint32)

    # the C planner already returns chunk order, so the sort is usually skipped
    key = (chunk_ids.astype(np.uint64) << np.uint64(32)) | block_ids.astype(np.uint64)
    if np.all(key[1:] > key[:-1]):
        chunks, blocks = chunk_ids, block_ids
    else:
        order = np.lexsort((block_ids, chunk_ids))
        chunks = chunk_ids[order]
        blocks = block_ids[order]

    out_chunk, out_first, out_last, out_item = native().merge(
        chunks, blocks, merge_gap_blocks
    )
    item_end = np.r_[out_item[1:], n]
    out = [
        _MergedRange(
            key=keys[int(c)],
            start=base + int(f) * block_size,
            end=base + (int(last) + 1) * block_size,
            first_block=int(f),
            item_start=int(i),
            item_end=int(e),
        )
        for c, f, last, i, e in zip(out_chunk, out_first, out_last, out_item, item_end)
    ]

    return out, blocks


async def _query_async(
    arr: zarr.Array,
    lo: float,
    hi: float,
    options: QueryOptions | None = None,
) -> QueryResult:
    """Count values with lo < x <= hi. Either bound may be infinite.

    This is the shared core of :func:`query_gt` and :func:`query_range`.
    """
    if not isinstance(arr, zarr.Array):
        raise TypeError("the query expects an opened zarr.Array")

    opts = options or QueryOptions()
    batch_size = opts.request_batch_size or 4 * opts.request_concurrency

    t0 = time.perf_counter()

    lt = get_layout(arr)
    need_mask(lt)
    layer = len(lt.layers) - 1 if opts.layer is None else opts.layer
    if not 0 <= layer < len(lt.layers):
        raise ValueError(f"layer must be in [0, {len(lt.layers) - 1}]")
    rate = lt.layers[layer]
    block_size = int(8 * rate)
    coords = chunk_coords(lt.grid_shape)
    keys = [full_key(arr, arr.metadata.encode_chunk_key(coord)) for coord in coords]
    store = arr.store_path.store

    # With layer 0 and zone map objects that cover whole chunks along the
    # first axis, each object is planned and its blocks read as soon as it
    # arrives, overlapping the three steps below instead of running them in turn.
    if layer > 0 and opts.max_inflight_bytes is not None:
        raise ValueError(
            "max_inflight_bytes needs layer 0: other layers hold every read "
            "until all have arrived"
        )
    mo = meta_objects(arr, lt)
    if (
        opts.pipeline
        and opts.meta_by_key
        and layer == 0
        and mo is not None
        and mo.t_chunk % lt.unit_grid_per_chunk[0] == 0
    ):
        return await _query_pipelined(arr, lt, mo, lo, hi, opts, keys, t0)
    budget = ByteBudget(opts.max_inflight_bytes)
    trace: dict[str, list[Any]] | None = None

    # 1. read the zone map: per-unit value range and error bounds, plus
    # per-block offsets
    t1 = time.perf_counter()
    meta, meta_bytes, meta_requests = await read_meta(
        arr, lt, opts.request_concurrency, opts.meta_by_key
    )
    meta_read_sec = time.perf_counter() - t1
    meta_done = time.perf_counter() - t0

    # 2. plan: classify every block as IN, OUT or MAYBE. plan_meta keeps only
    # the chosen layer's error bound, so the C planner sees a single-layer
    # record that is 4 bytes per extra layer shorter.
    t1 = time.perf_counter()
    chunk_ids, block_ids, n_in, n_out, *in_values = await asyncio.to_thread(
        native().plan_chunks,
        plan_meta(meta, lt, layer),
        lt.meta_size - 4 * (len(lt.layers) - 1),
        lt.unit_grid,
        lt.unit_grid_per_chunk,
        lt.blocks_per_unit,
        lo,
        opts.threads,
        hi,
        None if lt.mask is None else lt.mask.nvalid,
    )
    plan_sec = time.perf_counter() - t1
    # every valid value of an IN block matches, so IN blocks are counted unread
    in_count = in_values[0] if in_values else n_in * lt.values_per_block

    # 3. turn MAYBE blocks into byte-range reads. Each layer is stored as its
    # own contiguous section of the chunk, so every layer up to `layer` needs
    # its own set of reads. They share one gap in blocks, from the bytes a
    # block takes over all of them.
    gap = gap_blocks(opts, sum(lt.layer_bytes[: layer + 1]))
    per_layer = [
        merge_ranges(
            keys,
            chunk_ids,
            block_ids,
            lt.layer_bytes[j],
            gap,
            lt.layer_starts[j],
        )
        for j in range(layer + 1)
    ]
    sorted_blocks = per_layer[0][1]
    sorted_chunks = chunk_ids
    if sorted_blocks is not block_ids:
        sorted_chunks = chunk_ids[np.lexsort((block_ids, chunk_ids))]
    sorted_masks = block_bits(lt, sorted_chunks, sorted_blocks)
    cols = np.cumsum((0, *lt.layer_bytes[: layer + 1]))

    # A chunk that would need many small reads is usually cheaper to fetch as
    # one read of its whole layer prefix, since per-request latency dominates.
    dense: set[str] = set()
    if layer > 0:
        per_key = collections.Counter(
            req.key for merged, _ in per_layer for req in merged
        )
        dense = {k for k, n in per_key.items() if n > opts.max_chunk_requests}
    prefix_end = int(cols[-1]) * lt.blocks_per_chunk

    # 4. read and decode the MAYBE blocks.
    if layer == 0:
        # Layer 0 alone is a complete low-rate encoding of each block, so each
        # read is decoded as soon as it arrives, overlapping with the reads
        # still in flight. Hence
        # payload_read_sec includes decoding, and decode_sec sums the decode
        # time of every read.
        merged0, sorted0 = per_layer[0]
        if opts.trace:
            trace = {"meta": [meta_done], "reads": []}
        t1 = time.perf_counter()
        n_maybe_val, payload_bytes, decode_sec = await stream_count(
            store,
            merged0,
            sorted0,
            lt.layer_bytes[0],
            rate,
            lt.block_dim,
            lo,
            hi,
            opts.request_concurrency,
            opts.threads,
            budget=budget,
            copy=not opts.zero_copy,
            trace=trace["reads"] if trace else None,
            t0=t0,
            masks=sorted_masks,
        )
        payload_read_sec = time.perf_counter() - t1
        n_payload_reqs = len(merged0)
        peak = budget.peak
    else:
        # A block's bits are split across layers, so all reads must finish
        # before any block can be decoded. Read everything, assemble one row
        # per block, then decode in a single multithreaded C call.
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
            concurrency=opts.request_concurrency,
            batch_size=batch_size,
            copy=not opts.zero_copy,
        )
        payload_read_sec = time.perf_counter() - t1

        # row i = block i's layer-0 bytes, then layer-1 bytes, ... which is a
        # valid ZFP stream at the combined rate
        out = np.empty((len(sorted_blocks), int(cols[-1])), dtype=np.uint8)
        for buf, (j, item) in zip(payload_parts, fills):
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
            native().count,
            blocks,
            len(sorted_blocks),
            block_size,
            rate,
            lt.block_dim,
            lo,
            opts.threads,
            hi,
            sorted_masks,
        )
        decode_sec = time.perf_counter() - t1
        payload_bytes = sum(len(x) for x in payload_parts)
        n_payload_reqs = len(payload_reqs)
        peak = payload_bytes

    total_blocks = len(keys) * lt.blocks_per_chunk
    total_count = in_count + n_maybe_val
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
        peak_inflight_bytes=peak,
        trace=trace,
    )


async def _query_pipelined(
    arr: zarr.Array,
    lt: _Layout,
    mo: _MetaObjects,
    lo: float,
    hi: float,
    opts: QueryOptions,
    keys: Sequence[str],
    t0: float,
) -> QueryResult:
    """Run a layer 0 query one zone map object at a time.

    Each object covers whole chunks along the first axis, so it can be planned
    alone, and its MAYBE blocks are read while later objects are still
    arriving. The objects and the block reads share one concurrency limit.
    Since the steps overlap, metadata_read_seconds is the time until the last
    object arrived, planning_seconds sums the planning of every object, and
    payload_read_seconds runs from the first block read to the end.
    """
    store = arr.store_path.store
    sem = asyncio.Semaphore(opts.request_concurrency)
    budget = ByteBudget(opts.max_inflight_bytes)
    subs = lt.unit_grid_per_chunk
    chunks_per_step = lt.grid_shape[1] * lt.grid_shape[2]
    meta_size = lt.meta_size - 4 * (len(lt.layers) - 1)
    meta_sizes: list[int] = []  # stored size of each zone map object
    last_meta = t0
    first_read = math.inf
    trace: dict[str, list[Any]] | None = (
        {"meta": [], "reads": []} if opts.trace else None
    )

    def plan(slab: np.ndarray, n_units: int) -> tuple[Any, ...]:
        t1 = time.perf_counter()
        out = native().plan_chunks(
            plan_meta(slab, lt, 0),
            meta_size,
            (n_units, *lt.unit_grid[1:]),
            subs,
            lt.blocks_per_unit,
            lo,
            1,  # objects are planned in parallel, one thread each
            hi,
            None if lt.mask is None else lt.mask.nvalid,
        )
        if lt.mask is None:  # every value of an IN block matches
            out = (*out, out[2] * lt.values_per_block)
        return (*out, time.perf_counter() - t1)

    async def one(i: int) -> tuple[int, ...]:
        nonlocal last_meta, first_read
        data = await read_meta_object(store, mo, i, sem, lt, meta_sizes)
        arrived = time.perf_counter()
        last_meta = max(last_meta, arrived)
        if trace is not None:
            trace["meta"].append(arrived - t0)
        u0 = i * mo.t_chunk
        n_units = min(mo.t_chunk, lt.unit_grid[0] - u0)
        chunk_ids, block_ids, n_in, n_out, in_count, sec = await asyncio.to_thread(
            plan, data[: n_units * mo.rows], n_units
        )
        chunk_ids += np.uint32(u0 // subs[0] * chunks_per_step)
        merged, blocks = merge_ranges(
            keys,
            chunk_ids,
            block_ids,
            lt.layer_bytes[0],
            gap_blocks(opts, lt.layer_bytes[0]),
        )
        if blocks is not block_ids:
            chunk_ids = chunk_ids[np.lexsort((block_ids, chunk_ids))]
        if merged:
            first_read = min(first_read, time.perf_counter())
        n, nbytes, dec = await stream_count(
            store,
            merged,
            blocks,
            lt.layer_bytes[0],
            lt.layers[0],
            lt.block_dim,
            lo,
            hi,
            opts.request_concurrency,
            opts.threads,
            sem,
            budget,
            not opts.zero_copy,
            trace["reads"] if trace is not None else None,
            t0,
            block_bits(lt, chunk_ids, blocks),
        )
        return n_in, n_out, in_count, len(blocks), n, nbytes, len(merged), sec, dec

    parts = np.array(
        await asyncio.gather(*(one(i) for i in range(mo.count))), dtype=np.float64
    ).reshape(-1, 9)
    n_in, n_out, in_count, n_maybe, n_maybe_val, payload_bytes, n_reqs = (
        int(x) for x in parts[:, :7].sum(axis=0)
    )
    plan_sec, decode_sec = (float(x) for x in parts[:, 7:].sum(axis=0))
    end = time.perf_counter()
    useful_bytes = n_maybe * lt.layer_bytes[0]

    return QueryResult(
        count=in_count + n_maybe_val,
        in_blocks=n_in,
        out_blocks=n_out,
        maybe_blocks=n_maybe,
        total_blocks=len(keys) * lt.blocks_per_chunk,
        metadata_bytes_read=sum(meta_sizes),
        payload_bytes_read=payload_bytes,
        metadata_requests=mo.count,
        payload_requests=n_reqs,
        useful_payload_bytes=useful_bytes,
        payload_overread_bytes=payload_bytes - useful_bytes,
        metadata_read_seconds=last_meta - t0,
        planning_seconds=plan_sec,
        payload_read_seconds=end - first_read if n_reqs else 0.0,
        decode_seconds=decode_sec,
        total_seconds=end - t0,
        peak_inflight_bytes=budget.peak,
        trace=trace,
    )


def _run(
    source: Any,
    lo: float,
    hi: float,
    array_path: str,
    storage_options: Mapping[str, Any] | None,
    options: QueryOptions | None,
) -> QueryResult:
    """Open source and count lo < x <= hi on it."""
    arr = open_skipzfp(source, array_path=array_path, storage_options=storage_options)
    return sync(_query_async(arr, lo, hi, options))


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
        hi_inclusive: Use x <= hi, or x < hi when False.

    Returns:
        (a, b) such that a < x <= b selects the same float32 values, with
        a missing bound mapped to -inf or inf.

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


async def query_gt_async(
    arr: zarr.Array,
    threshold: float,
    *,
    inclusive: bool = False,
    options: QueryOptions | None = None,
) -> QueryResult:
    """Async version of :func:`query_gt` for an already opened array.

    It takes the same keyword arguments as :func:`query_gt` except
    array_path and storage_options, and is useful for running several
    queries concurrently.

    Raises:
        TypeError: If arr is not an opened zarr.Array.
    """
    if not isinstance(arr, zarr.Array):
        raise TypeError("query_gt_async expects an opened zarr.Array")
    lo, hi = range_bounds(threshold, None, lo_inclusive=inclusive)
    return await _query_async(
        arr,
        lo,
        hi,
        options,
    )


def query_gt(
    source: Any,
    threshold: float,
    *,
    inclusive: bool = False,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    options: QueryOptions | None = None,
) -> QueryResult:
    """Count values greater than threshold in a SkipZFP array.

    The zone map shows which blocks lie entirely above or below the
    threshold, so those blocks are counted or skipped without being read,
    and only the remaining blocks are downloaded and decoded.

    The count equals what you get by decompressing the whole array and
    counting. It can differ slightly from counting the original data,
    because ZFP compression is lossy.

    Args:
        source: An opened zarr.Array, or a local path or URL to open.
        threshold: Count values strictly greater than this.
        inclusive: Use x >= threshold instead of x > threshold.
        array_path: Path of the array inside the store, when source is
            not an opened array.
        storage_options: Options for a cloud store, such as credentials,
            passed to fsspec when source is a URL.
        options: Tuning knobs such as threads and read merging, see
            :class:`QueryOptions`. The defaults suit most queries.

    Returns:
        QueryResult with the count, the number of IN, OUT and MAYBE blocks,
        and the bytes, requests and seconds spent in each stage.

    Raises:
        ValueError: If threshold is NaN, the array's shape or chunking is not
            supported, it does not use the skipzfp codec, it has no zone map (see
            :func:`write_meta`), or options.layer is out of range.
        RuntimeError: If source is a URL and its fsspec backend is missing.
        FileNotFoundError: If a chunk the query needs is missing.

    Example:
        >>> r = query_gt("data/t2m.zarr", 295.0)
        >>> print(r.count, r.bytes_read)
    """

    lo, hi = range_bounds(threshold, None, lo_inclusive=inclusive)
    return _run(
        source,
        lo,
        hi,
        array_path,
        storage_options,
        options,
    )


async def query_lt_async(
    arr: zarr.Array,
    threshold: float,
    *,
    inclusive: bool = False,
    options: QueryOptions | None = None,
) -> QueryResult:
    """Async version of :func:`query_lt` for an already opened array.

    It takes the same keyword arguments as :func:`query_lt` except
    array_path and storage_options, and is useful for running several
    queries concurrently.

    Raises:
        TypeError: If arr is not an opened zarr.Array.
    """
    if not isinstance(arr, zarr.Array):
        raise TypeError("query_lt_async expects an opened zarr.Array")
    lo, hi = range_bounds(None, threshold, hi_inclusive=inclusive)
    return await _query_async(
        arr,
        lo,
        hi,
        options,
    )


def query_lt(
    source: Any,
    threshold: float,
    *,
    inclusive: bool = False,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    options: QueryOptions | None = None,
) -> QueryResult:
    """Count values less than threshold in a SkipZFP array.

    It is the mirror image of :func:`query_gt` and takes the same arguments.

    Args:
        source: An opened zarr.Array, or a local path or URL to open.
        threshold: Count values strictly less than this.
        inclusive: Use x <= threshold instead of x < threshold.
        array_path, storage_options, options: Same as in :func:`query_gt`.

    Returns:
        QueryResult, as in :func:`query_gt`.

    Raises:
        ValueError: In any case where :func:`query_gt` raises it.

    Example:
        >>> query_lt("data/t2m.zarr", 273.15).count  # x < 273.15
    """
    lo, hi = range_bounds(None, threshold, hi_inclusive=inclusive)
    return _run(
        source,
        lo,
        hi,
        array_path,
        storage_options,
        options,
    )


async def query_range_async(
    arr: zarr.Array,
    lo: float,
    hi: float,
    *,
    lo_inclusive: bool = False,
    hi_inclusive: bool = True,
    options: QueryOptions | None = None,
) -> QueryResult:
    """Async version of :func:`query_range` for an already opened array.

    It takes the same keyword arguments as :func:`query_range` except
    array_path and storage_options, and is useful for running several
    queries concurrently.

    Raises:
        TypeError: If arr is not an opened zarr.Array.
    """
    if lo is None or hi is None:
        raise TypeError(
            "query_range needs both bounds; use query_gt or query_lt for one side"
        )
    a, b = range_bounds(lo, hi, lo_inclusive, hi_inclusive)
    return await _query_async(
        arr,
        a,
        b,
        options,
    )


def query_range(
    source: Any,
    lo: float,
    hi: float,
    *,
    lo_inclusive: bool = False,
    hi_inclusive: bool = True,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    options: QueryOptions | None = None,
) -> QueryResult:
    """Count values between two bounds, by default lo < x <= hi.

    This function works like :func:`query_gt`. It uses the zone map to
    count or skip whole blocks, and decodes only the blocks that lie
    partly inside the range. For a one-sided bound, use :func:`query_gt`
    or :func:`query_lt`.

    The count equals what you get by decompressing the whole array and
    counting. It can differ slightly from counting the original data,
    because ZFP compression is lossy.

    Args:
        source: An opened zarr.Array, or a local path or URL to open.
        lo: Lower bound.
        hi: Upper bound.
        lo_inclusive: Use lo <= x instead of lo < x.
        hi_inclusive: Use x <= hi, or x < hi when False.
        array_path, storage_options, options: Same as in :func:`query_gt`.

    Returns:
        QueryResult, as in :func:`query_gt`.

    Raises:
        TypeError: If lo or hi is None.
        ValueError: If lo or hi is NaN, if lo > hi, or in any case where
            :func:`query_gt` raises it.

    Example:
        >>> query_range("data/t2m.zarr", 290.0, 295.0).count  # 290 < x <= 295
    """
    if lo is None or hi is None:
        raise TypeError(
            "query_range needs both bounds; use query_gt or query_lt for one side"
        )
    a, b = range_bounds(lo, hi, lo_inclusive, hi_inclusive)
    return _run(
        source,
        a,
        b,
        array_path,
        storage_options,
        options,
    )
