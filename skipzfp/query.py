from __future__ import annotations

import asyncio
import collections
import ctypes
import itertools
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from ._native import load_library
from .codec import META_ATTR, meta_size

# plan in chunk order in C; False keeps the older unit-order plan + numpy remap
FUSED_PLAN = True


class _Native:
    """ctypes bindings for the query functions in csrc/query.c."""

    def __init__(self) -> None:
        self.lib = load_library()

        self.lib.szfp_layout.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_layout.restype = ctypes.c_int

        self.lib.szfp_plan_gt.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_plan_gt.restype = ctypes.c_int

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
            raise RuntimeError(f"{name} failed: {code}")

    def layout(
        self,
        shape: tuple[int, int, int],
        rate: float,
        block_dim: int,
    ) -> tuple[int, int]:
        """Return the byte sizes of a chunk's payload and of its zone map record."""
        data_size = ctypes.c_size_t()
        meta_size = ctypes.c_size_t()

        code = self.lib.szfp_layout(
            shape[0],
            shape[1],
            shape[2],
            rate,
            block_dim,
            ctypes.byref(data_size),
            ctypes.byref(meta_size),
        )
        self.check(code, "szfp_layout")

        return int(data_size.value), int(meta_size.value)

    def plan(
        self,
        meta: np.ndarray,
        n_chunk: int,
        meta_size: int,
        n_block: int,
        th: float,
        threads: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Classify blocks with the older unit-order planner.

        It returns the MAYBE blocks as (unit, rank) pairs, plus the IN and
        OUT counts. :func:`unit_to_chunk` turns the pairs into chunk order.
        """
        src = np.ascontiguousarray(meta, dtype=np.uint8)
        cap = n_chunk * n_block
        chunk_ids = np.empty(cap, dtype=np.uint32)
        block_ids = np.empty(cap, dtype=np.uint32)

        n_maybe = ctypes.c_size_t()
        n_in = ctypes.c_size_t()
        n_out = ctypes.c_size_t()

        code = self.lib.szfp_plan_gt(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            n_chunk,
            meta_size,
            n_block,
            th,
            threads,
            chunk_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            cap,
            ctypes.byref(n_maybe),
            ctypes.byref(n_in),
            ctypes.byref(n_out),
        )
        self.check(code, "szfp_plan_gt")

        n = int(n_maybe.value)
        return (
            chunk_ids[:n].copy(),
            block_ids[:n].copy(),
            int(n_in.value),
            int(n_out.value),
        )

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
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Classify blocks for th < x <= hi and return the MAYBE blocks in chunk order.

        It returns the (chunk, block) ids of the MAYBE blocks, plus the IN
        and OUT counts. hi comes last so callers written for x > th still work.
        """
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
            th,
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
        th: float,
        threads: int,
        hi: float = math.inf,
    ) -> int:
        """Decode n_block packed blocks and count values with th < x <= hi."""
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
            th,
            hi,
            threads,
            ctypes.byref(out),
        )
        self.check(code, "szfp_count_range_blocks")
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
        self.check(code, "szfp_merge_ranges")
        k = int(m.value)
        return out_chunk[:k], out_first[:k], out_last[:k], out_item[:k]

    def count_offsets(
        self,
        data: bytes,
        offsets: np.ndarray,
        block_size: int,
        rate: float,
        block_dim: int,
        th: float,
        hi: float = math.inf,
    ) -> int:
        """Decode the blocks at the given byte offsets and count th < x <= hi."""
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
            th,
            hi,
            ctypes.byref(out),
        )
        self.check(code, "szfp_count_offsets_range")
        return int(out.value)


_NATIVE: _Native | None = None
_POOL: ThreadPoolExecutor | None = None


def pool() -> ThreadPoolExecutor:
    """Return the shared thread pool that decodes reads as they arrive."""
    global _POOL

    if _POOL is None:
        _POOL = ThreadPoolExecutor(os.cpu_count() or 4)

    return _POOL


def native() -> _Native:
    """Return the shared _Native, loading the C library on first use."""
    global _NATIVE

    if _NATIVE is None:
        _NATIVE = _Native()

    return _NATIVE


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

    @property
    def bytes_read(self) -> int:
        """Total bytes read from the zone map and the compressed blocks."""
        return self.metadata_bytes_read + self.payload_bytes_read

    @property
    def range_requests(self) -> int:
        """Total read requests for the zone map and the compressed blocks."""
        return self.metadata_requests + self.payload_requests


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
    query functions do that.

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


def find_codec(arr: zarr.Array) -> Any:
    """Return the skipzfp codec of arr, matching by name rather than by class."""
    for codec in arr.metadata.codecs:
        if getattr(codec, "codec_name", None) == "skipzfp":
            return codec

        to_dict = getattr(codec, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, dict) and data.get("name") == "skipzfp":
                return codec

    raise ValueError("array does not use the skipzfp codec")


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

    data_size, _ = native().layout(
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
        unit_grid=tuple(g * (c // u) for g, c, u in zip(grid_shape, chunk_shape, unit)),
        blocks_per_unit=blocks_per_unit,
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
) -> bytes:
    """Read one byte range.

    It raises an error if the object is missing or fewer bytes come back
    than requested.
    """
    async with sem:
        buf = await store.get(
            req.key,
            prototype=default_buffer_prototype(),
            byte_range=RangeByteRequest(req.start, req.end),
        )

    if buf is None:
        raise FileNotFoundError(req.key)

    data = buf.to_bytes()
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
) -> list[bytes]:
    """Read byte ranges in batches.

    At most `concurrency` reads run at a time, and the results come back in
    the same order as reqs.
    """
    if concurrency <= 0:
        raise ValueError("request_concurrency must be greater than 0")
    if batch_size <= 0:
        raise ValueError("request_batch_size must be greater than 0")

    sem = asyncio.Semaphore(concurrency)
    out: list[bytes] = []

    for batch in batched(reqs, batch_size):
        parts = await asyncio.gather(*(read_one(store, req, sem) for req in batch))
        out.extend(parts)

    return out


async def read_meta(arr: zarr.Array, lt: _Layout) -> tuple[np.ndarray, int, int]:
    """Read the whole zone map, with its size in bytes and number of objects."""
    path = arr.attrs.get(META_ATTR)
    if not path:
        raise ValueError("array has no metadata; run codec.write_meta first")

    meta = await zarr.api.asynchronous.open_array(
        store=arr.store_path.store,
        path=path,
        mode="r",
    )

    if tuple(meta.shape) != (*lt.unit_grid, lt.meta_size):
        raise ValueError(f"metadata shape {meta.shape} does not match the array layout")

    data = np.ascontiguousarray(await meta.getitem(...), dtype=np.uint8).reshape(-1)
    n_obj = int(np.prod([ceildiv(s, c) for s, c in zip(meta.shape, meta.chunks)]))
    return data, data.nbytes, n_obj


def unit_to_chunk(
    lt: _Layout, unit_ids: np.ndarray, ranks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the older planner's (unit, rank) pairs into chunk-order ids.

    It is only used when FUSED_PLAN is False.
    """
    subs = tuple(c // u for c, u in zip(lt.chunk_shape, lt.unit_shape))
    n_unit = int(np.prod(lt.unit_grid))
    u = np.unravel_index(np.arange(n_unit), lt.unit_grid)
    u_chunk = np.ravel_multi_index(
        tuple(x // s for x, s in zip(u, subs)), lt.grid_shape
    )
    u_sub = np.ravel_multi_index(tuple(x % s for x, s in zip(u, subs)), subs)

    counts = np.bincount(unit_ids, minlength=n_unit)
    starts = np.cumsum(counts) - counts
    u_order = np.lexsort((u_sub, u_chunk))
    seg = counts[u_order]
    perm = np.repeat(starts[u_order] - (np.cumsum(seg) - seg), seg) + np.arange(
        len(unit_ids)
    )

    uid = unit_ids[perm]
    blocks = u_sub[uid] * lt.blocks_per_unit + ranks[perm].astype(np.int64)
    return u_chunk[uid].astype(np.uint32), blocks.astype(np.uint32)


def plan_meta(meta: np.ndarray, lt: _Layout, layer: int) -> np.ndarray:
    """Keep only one layer's error bound in each record of the zone map.

    The C planner reads records with a single error bound, so the bounds of
    the other layers are dropped.
    """
    m = meta.reshape(-1, lt.meta_size)
    n_layer = len(lt.layers)
    eps = m[:, 8 + 4 * layer : 12 + 4 * layer]
    return np.ascontiguousarray(
        np.concatenate([m[:, :8], eps, m[:, 8 + 4 * n_layer :]], axis=1)
    )


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
) -> tuple[int, int, float]:
    """Read each merged range and count lo < x <= hi as soon as it arrives.

    It returns the count, the bytes read and the decode seconds summed over
    all reads.
    """
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    def count(data: bytes, offs: np.ndarray) -> tuple[int, float]:
        t0 = time.perf_counter()
        n = native().count_offsets(data, offs, block_size, rate, block_dim, lo, hi)
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
    *,
    threads: int = 0,
    request_concurrency: int = 32,
    request_batch_size: int | None = None,
    merge_gap_blocks: int = 0,
    layer: int | None = None,
    max_chunk_requests: int = 1,
) -> QueryResult:
    """Count values with lo < x <= hi. Either bound may be infinite.

    This is the shared core of :func:`query_gt` and :func:`query_range`.
    """
    if not isinstance(arr, zarr.Array):
        raise TypeError("the query expects an opened zarr.Array")

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

    # 1. read the zone map: per-unit value range and error bounds, plus
    # per-block offsets
    t1 = time.perf_counter()
    meta, meta_bytes, meta_requests = await read_meta(arr, lt)
    meta_read_sec = time.perf_counter() - t1

    # 2. plan: classify every block as IN, OUT or MAYBE. plan_meta keeps only
    # the chosen layer's error bound, so the C planner sees a single-layer
    # record that is 4 bytes per extra layer shorter.
    t1 = time.perf_counter()
    # the older unit-order planner only handles an open upper end
    if FUSED_PLAN or hi != math.inf:
        chunk_ids, block_ids, n_in, n_out = await asyncio.to_thread(
            native().plan_chunks,
            plan_meta(meta, lt, layer),
            lt.meta_size - 4 * (len(lt.layers) - 1),
            lt.unit_grid,
            tuple(c // u for c, u in zip(lt.chunk_shape, lt.unit_shape)),
            lt.blocks_per_unit,
            lo,
            threads,
            hi,
        )
    else:
        unit_ids, ranks, n_in, n_out = await asyncio.to_thread(
            native().plan,
            plan_meta(meta, lt, layer),
            int(np.prod(lt.unit_grid)),
            lt.meta_size - 4 * (len(lt.layers) - 1),
            lt.blocks_per_unit,
            lo,
            threads,
        )
        chunk_ids, block_ids = unit_to_chunk(lt, unit_ids, ranks)
    plan_sec = time.perf_counter() - t1

    # 3. turn MAYBE blocks into byte-range reads. Each layer is stored as its
    # own contiguous section of the chunk, so every layer up to `layer` needs
    # its own set of reads.
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

    # A chunk that would need many small reads is usually cheaper to fetch as
    # one read of its whole layer prefix, since per-request latency dominates.
    dense: set[str] = set()
    if layer > 0:
        per_key = collections.Counter(
            req.key for merged, _ in per_layer for req in merged
        )
        dense = {k for k, n in per_key.items() if n > max_chunk_requests}
    prefix_end = int(cols[-1]) * lt.blocks_per_chunk

    # 4. read and decode the MAYBE blocks.
    if layer == 0:
        # Layer 0 alone is a complete low-rate encoding of each block, so each
        # read is decoded as soon as it arrives, overlapping with the reads
        # still in flight. Hence
        # payload_read_sec includes decoding, and decode_sec sums the decode
        # time of every read.
        merged0, sorted0 = per_layer[0]
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
            request_concurrency,
        )
        payload_read_sec = time.perf_counter() - t1
        n_payload_reqs = len(merged0)
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
            concurrency=request_concurrency,
            batch_size=request_batch_size,
        )
        payload_read_sec = time.perf_counter() - t1

        # row i = block i's layer-0 bytes, then layer-1 bytes, ... which is a
        # valid ZFP stream at the combined rate
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
            native().count,
            blocks,
            len(sorted_blocks),
            block_size,
            rate,
            lt.block_dim,
            lo,
            threads,
            hi,
        )
        decode_sec = time.perf_counter() - t1
        payload_bytes = sum(len(x) for x in payload_parts)
        n_payload_reqs = len(payload_reqs)

    total_blocks = len(keys) * lt.blocks_per_chunk
    # every value of an IN block matches, so IN blocks are counted unread
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


async def query_gt_async(
    arr: zarr.Array,
    threshold: float,
    *,
    threads: int = 0,
    request_concurrency: int = 32,
    request_batch_size: int | None = None,
    merge_gap_blocks: int = 0,
    layer: int | None = None,
    max_chunk_requests: int = 1,
) -> QueryResult:
    """Async version of query_gt for an already opened array.

    It takes the same keyword arguments as :func:`query_gt` except
    array_path and storage_options, and is useful for running several
    queries concurrently.

    Raises:
        TypeError: If arr is not an opened zarr.Array.
    """
    if not isinstance(arr, zarr.Array):
        raise TypeError("query_gt_async expects an opened zarr.Array")
    return await _query_async(
        arr,
        float(threshold),
        math.inf,
        threads=threads,
        request_concurrency=request_concurrency,
        request_batch_size=request_batch_size,
        merge_gap_blocks=merge_gap_blocks,
        layer=layer,
        max_chunk_requests=max_chunk_requests,
    )


def query_gt(
    source: Any,
    threshold: float,
    *,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
    threads: int = 0,
    request_concurrency: int = 32,
    request_batch_size: int | None = None,
    merge_gap_blocks: int = 0,
    layer: int | None = None,
    max_chunk_requests: int = 1,
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
        array_path: Path of the array inside the store, when source is
            not an opened array.
        storage_options: Options for a cloud store, such as credentials,
            passed to fsspec when source is a URL.
        threads: C threads for planning, and for decoding when layer > 0,
            where 0 uses all cores.
        request_concurrency: Maximum number of reads running at a time.
        request_batch_size: Number of reads issued per batch when
            layer > 0, 4 * request_concurrency by default.
        merge_gap_blocks: Merge two reads in the same chunk when at most
            this many unneeded blocks lie between them, so larger values
            mean fewer requests but more bytes read.
        layer: Precision layer to read for arrays written with layers,
            where 0 is the lowest rate and None reads the full rate.
        max_chunk_requests: When layer > 0, a chunk that would need more
            than this many reads is fetched in a single read instead.

    Returns:
        QueryResult with the count, the number of IN, OUT and MAYBE blocks,
        and the bytes, requests and seconds spent in each stage.

    Raises:
        ValueError: If the array's shape or chunking is not supported, it
            does not use the skipzfp codec, it has no zone map (see
            :func:`write_meta`), or an argument is out of range.
        RuntimeError: If source is a URL and its fsspec backend is missing.
        FileNotFoundError: If a chunk the query needs is missing.

    Example:
        >>> r = query_gt("data/t2m.zarr", 295.0)
        >>> print(r.count, r.bytes_read)
    """
    arr = open_skipzfp(
        source,
        array_path=array_path,
        storage_options=storage_options,
    )

    return sync(
        _query_async(
            arr,
            float(threshold),
            math.inf,
            threads=threads,
            request_concurrency=request_concurrency,
            request_batch_size=request_batch_size,
            merge_gap_blocks=merge_gap_blocks,
            layer=layer,
            max_chunk_requests=max_chunk_requests,
        )
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

    It takes the same keyword arguments as :func:`query_range` except
    array_path and storage_options, and is useful for running several
    queries concurrently.

    Raises:
        TypeError: If arr is not an opened zarr.Array.
    """
    a, b = range_bounds(lo, hi, lo_inclusive, hi_inclusive)
    return await _query_async(arr, a, b, **kwargs)


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
        hi_inclusive: Use x <= hi, or x < hi when False.
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
        _query_async(
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
