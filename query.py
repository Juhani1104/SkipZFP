from __future__ import annotations

import asyncio
import ctypes
import itertools
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from codec import META_ATTR
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync


class _Native:
    def __init__(self) -> None:
        path = Path(__file__).with_name("core.so")
        if not path.exists():
            raise FileNotFoundError(f"could not find {path}; compile core.so first")

        self.lib = ctypes.CDLL(str(path))

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

        self.lib.szfp_count_gt_blocks.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_count_gt_blocks.restype = ctypes.c_int

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

    def count(
        self,
        blocks: np.ndarray,
        n_block: int,
        block_size: int,
        rate: float,
        block_dim: int,
        th: float,
        threads: int,
    ) -> int:
        if n_block == 0:
            return 0

        src = np.ascontiguousarray(blocks, dtype=np.uint8)
        out = ctypes.c_size_t()

        code = self.lib.szfp_count_gt_blocks(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            n_block,
            block_size,
            rate,
            block_dim,
            th,
            threads,
            ctypes.byref(out),
        )
        self.check(code, "szfp_count_gt_blocks")
        return int(out.value)


_NATIVE: _Native | None = None


def native() -> _Native:
    global _NATIVE

    if _NATIVE is None:
        _NATIVE = _Native()

    return _NATIVE


@dataclass(frozen=True)
class QueryResult:
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
        return self.metadata_bytes_read + self.payload_bytes_read

    @property
    def range_requests(self) -> int:
        return self.metadata_requests + self.payload_requests


@dataclass(frozen=True)
class _Layout:
    chunk_shape: tuple[int, int, int]
    grid_shape: tuple[int, int, int]
    rate: float
    block_dim: int
    data_size: int
    meta_size: int
    block_size: int
    blocks_per_chunk: int
    values_per_block: int


@dataclass(frozen=True)
class _Range:
    key: str
    start: int
    end: int


@dataclass(frozen=True)
class _MergedRange:
    key: str
    start: int
    end: int
    first_block: int
    item_start: int
    item_end: int


def open_skipzfp(
    source: Any,
    *,
    array_path: str = "",
    storage_options: Mapping[str, Any] | None = None,
) -> zarr.Array:
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

    data_size, meta_size = native().layout(
        chunk_shape,
        rate,
        block_dim,
    )

    if blocks_per_chunk <= 0:
        raise ValueError("invalid block layout")
    if data_size <= 0 or meta_size <= 0:
        raise ValueError("invalid fixed-rate layout")
    if data_size % blocks_per_chunk != 0:
        raise ValueError("compressed payload is not divisible by blocks")

    block_size = data_size // blocks_per_chunk

    return _Layout(
        chunk_shape=chunk_shape,
        grid_shape=grid_shape,
        rate=rate,
        block_dim=block_dim,
        data_size=data_size,
        meta_size=meta_size,
        block_size=block_size,
        blocks_per_chunk=blocks_per_chunk,
        values_per_block=values_per_block,
    )


def chunk_coords(
    grid_shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    return list(itertools.product(*(range(n) for n in grid_shape)))


def full_key(arr: zarr.Array, chunk_key: str) -> str:
    prefix = arr.store_path.path.strip("/")
    return f"{prefix}/{chunk_key}" if prefix else chunk_key


def batched(items: Iterable[_Range], size: int) -> Iterable[list[_Range]]:
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
    path = arr.attrs.get(META_ATTR)
    if not path:
        raise ValueError("array has no metadata; run codec.write_meta first")

    meta = await zarr.api.asynchronous.open_array(
        store=arr.store_path.store,
        path=path,
        mode="r",
    )

    if tuple(meta.shape) != (*lt.grid_shape, lt.meta_size):
        raise ValueError(f"metadata shape {meta.shape} does not match the array layout")

    data = np.ascontiguousarray(await meta.getitem(...), dtype=np.uint8).reshape(-1)
    n_obj = int(np.prod([ceildiv(s, c) for s, c in zip(meta.shape, meta.chunks)]))
    return data, data.nbytes, n_obj


def merge_ranges(
    keys: Sequence[str],
    chunk_ids: np.ndarray,
    block_ids: np.ndarray,
    block_size: int,
    merge_gap_blocks: int,
) -> tuple[list[_MergedRange], np.ndarray]:
    if merge_gap_blocks < 0:
        raise ValueError("merge_gap_blocks must be non-negative")

    n = len(block_ids)
    if n == 0:
        return [], np.empty(0, dtype=np.uint32)

    order = np.lexsort((block_ids, chunk_ids))
    chunks = chunk_ids[order]
    blocks = block_ids[order]

    out: list[_MergedRange] = []
    item_start = 0
    chunk = int(chunks[0])
    first = int(blocks[0])
    last = first

    for i in range(1, n):
        next_chunk = int(chunks[i])
        next_block = int(blocks[i])

        same_range = next_chunk == chunk and next_block <= last + merge_gap_blocks + 1

        if same_range:
            last = next_block
            continue

        out.append(
            _MergedRange(
                key=keys[chunk],
                start=first * block_size,
                end=(last + 1) * block_size,
                first_block=first,
                item_start=item_start,
                item_end=i,
            )
        )

        item_start = i
        chunk = next_chunk
        first = next_block
        last = next_block

    out.append(
        _MergedRange(
            key=keys[chunk],
            start=first * block_size,
            end=(last + 1) * block_size,
            first_block=first,
            item_start=item_start,
            item_end=n,
        )
    )

    return out, blocks


def extract_blocks(
    parts: Sequence[bytes],
    merged: Sequence[_MergedRange],
    blocks: np.ndarray,
    block_size: int,
) -> np.ndarray:
    out = np.empty(len(blocks) * block_size, dtype=np.uint8)
    cur = 0

    for data, req in zip(parts, merged, strict=True):
        n_block = (req.end - req.start) // block_size
        mat = np.frombuffer(data, dtype=np.uint8).reshape(n_block, block_size)

        target = blocks[req.item_start : req.item_end]
        rows = target.astype(np.int64) - req.first_block
        picked = mat[rows]

        n_pick = len(target)
        start = cur * block_size
        end = (cur + n_pick) * block_size
        out[start:end] = picked.reshape(-1)
        cur += n_pick

    if cur != len(blocks):
        raise RuntimeError("failed to assemble MAYBE blocks")

    return out


async def query_gt_async(
    arr: zarr.Array,
    threshold: float,
    *,
    threads: int = 0,
    request_concurrency: int = 32,
    request_batch_size: int | None = None,
    merge_gap_blocks: int = 0,
) -> QueryResult:
    if not isinstance(arr, zarr.Array):
        raise TypeError("query_gt_async expects an opened zarr.Array")

    if request_batch_size is None:
        request_batch_size = max(request_concurrency, request_concurrency * 4)

    t0 = time.perf_counter()

    lt = get_layout(arr)
    coords = chunk_coords(lt.grid_shape)
    keys = [full_key(arr, arr.metadata.encode_chunk_key(coord)) for coord in coords]
    store = arr.store_path.store

    t1 = time.perf_counter()
    meta, meta_bytes, meta_requests = await read_meta(arr, lt)
    meta_read_sec = time.perf_counter() - t1

    t1 = time.perf_counter()
    chunk_ids, block_ids, n_in, n_out = await asyncio.to_thread(
        native().plan,
        meta,
        len(keys),
        lt.meta_size,
        lt.blocks_per_chunk,
        float(threshold),
        threads,
    )
    plan_sec = time.perf_counter() - t1

    merged, sorted_blocks = merge_ranges(
        keys,
        chunk_ids,
        block_ids,
        lt.block_size,
        merge_gap_blocks,
    )

    payload_reqs = [
        _Range(
            key=req.key,
            start=req.start,
            end=req.end,
        )
        for req in merged
    ]

    t1 = time.perf_counter()
    payload_parts = await read_ranges(
        store,
        payload_reqs,
        concurrency=request_concurrency,
        batch_size=request_batch_size,
    )
    payload_read_sec = time.perf_counter() - t1

    blocks = extract_blocks(
        payload_parts,
        merged,
        sorted_blocks,
        lt.block_size,
    )

    t1 = time.perf_counter()
    n_maybe_val = await asyncio.to_thread(
        native().count,
        blocks,
        len(sorted_blocks),
        lt.block_size,
        lt.rate,
        lt.block_dim,
        float(threshold),
        threads,
    )
    decode_sec = time.perf_counter() - t1

    total_blocks = len(keys) * lt.blocks_per_chunk
    total_count = n_in * lt.values_per_block + n_maybe_val
    payload_bytes = sum(len(x) for x in payload_parts)
    useful_bytes = len(sorted_blocks) * lt.block_size

    return QueryResult(
        count=total_count,
        in_blocks=n_in,
        out_blocks=n_out,
        maybe_blocks=len(sorted_blocks),
        total_blocks=total_blocks,
        metadata_bytes_read=meta_bytes,
        payload_bytes_read=payload_bytes,
        metadata_requests=meta_requests,
        payload_requests=len(payload_reqs),
        useful_payload_bytes=useful_bytes,
        payload_overread_bytes=payload_bytes - useful_bytes,
        metadata_read_seconds=meta_read_sec,
        planning_seconds=plan_sec,
        payload_read_seconds=payload_read_sec,
        decode_seconds=decode_sec,
        total_seconds=time.perf_counter() - t0,
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
) -> QueryResult:
    arr = open_skipzfp(
        source,
        array_path=array_path,
        storage_options=storage_options,
    )

    return sync(
        query_gt_async(
            arr,
            threshold,
            threads=threads,
            request_concurrency=request_concurrency,
            request_batch_size=request_batch_size,
            merge_gap_blocks=merge_gap_blocks,
        )
    )
