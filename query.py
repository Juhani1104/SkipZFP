from __future__ import annotations

import asyncio
import collections
import ctypes
import os
import itertools
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from codec import META_ATTR, meta_size
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

        self.lib.szfp_count_offsets.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_count_offsets.restype = ctypes.c_int

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


    def merge(
        self,
        chunk_ids: np.ndarray,
        block_ids: np.ndarray,
        gap: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    ) -> int:
        offs = np.ascontiguousarray(offsets, dtype=np.uint64)
        out = ctypes.c_size_t()

        code = self.lib.szfp_count_offsets(
            data,
            len(data),
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(offs),
            block_size,
            rate,
            block_dim,
            th,
            ctypes.byref(out),
        )
        self.check(code, "szfp_count_offsets")
        return int(out.value)


_NATIVE: _Native | None = None
_POOL: ThreadPoolExecutor | None = None


def pool() -> ThreadPoolExecutor:
    global _POOL

    if _POOL is None:
        _POOL = ThreadPoolExecutor(os.cpu_count() or 4)

    return _POOL


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
    layers: tuple[float, ...]
    layer_bytes: tuple[int, ...]
    layer_starts: tuple[int, ...]
    unit_shape: tuple[int, int, int]
    unit_grid: tuple[int, int, int]
    blocks_per_unit: int


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

    if tuple(meta.shape) != (*lt.unit_grid, lt.meta_size):
        raise ValueError(f"metadata shape {meta.shape} does not match the array layout")

    data = np.ascontiguousarray(await meta.getitem(...), dtype=np.uint8).reshape(-1)
    n_obj = int(np.prod([ceildiv(s, c) for s, c in zip(meta.shape, meta.chunks)]))
    return data, data.nbytes, n_obj


def unit_to_chunk(lt: _Layout, unit_ids: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sub_chunk 編號 + sub_chunk 內的 block 位置 → Zarr chunk 編號 + chunk 內的 block 位置。"""
    subs = tuple(c // u for c, u in zip(lt.chunk_shape, lt.unit_shape))
    u = np.unravel_index(unit_ids.astype(np.int64), lt.unit_grid)
    chunk = np.ravel_multi_index(tuple(x // s for x, s in zip(u, subs)), lt.grid_shape)
    sub_idx = np.ravel_multi_index(tuple(x % s for x, s in zip(u, subs)), subs)
    blocks = sub_idx * lt.blocks_per_unit + ranks.astype(np.int64)
    return chunk.astype(np.uint32), blocks.astype(np.uint32)


def plan_meta(meta: np.ndarray, lt: _Layout, layer: int) -> np.ndarray:
    """把新格式（每層一個 eps）轉成 planner 用的 (cmin, cmax, 該層 eps, offsets)。"""
    m = meta.reshape(-1, lt.meta_size)
    n_layer = len(lt.layers)
    eps = m[:, 8 + 4 * layer : 12 + 4 * layer]
    return np.ascontiguousarray(np.concatenate([m[:, :8], eps, m[:, 8 + 4 * n_layer :]], axis=1))


async def stream_count(
    store: Any,
    merged: Sequence[_MergedRange],
    blocks: np.ndarray,
    block_size: int,
    rate: float,
    block_dim: int,
    threshold: float,
    concurrency: int,
) -> tuple[int, int, float]:
    """每個 range 一下載完就交給 C 原地解碼計數，讓解碼跟下載重疊。"""
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    def count(data: bytes, offs: np.ndarray) -> tuple[int, float]:
        t0 = time.perf_counter()
        n = native().count_offsets(data, offs, block_size, rate, block_dim, threshold)
        return n, time.perf_counter() - t0

    async def one(req: _MergedRange) -> tuple[int, int, float]:
        data = await read_one(store, _Range(key=req.key, start=req.start, end=req.end), sem)
        rows = blocks[req.item_start : req.item_end].astype(np.int64) - req.first_block
        n, sec = await loop.run_in_executor(pool(), count, data, (rows * block_size).astype(np.uint64))
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
    if merge_gap_blocks < 0:
        raise ValueError("merge_gap_blocks must be non-negative")

    n = len(block_ids)
    if n == 0:
        return [], np.empty(0, dtype=np.uint32)

    order = np.lexsort((block_ids, chunk_ids))
    chunks = chunk_ids[order]
    blocks = block_ids[order]

    out_chunk, out_first, out_last, out_item = native().merge(chunks, blocks, merge_gap_blocks)
    item_end = np.r_[out_item[1:], n]
    out = [
        _MergedRange(
            key=keys[int(c)],
            start=base + int(f) * block_size,
            end=base + (int(l) + 1) * block_size,
            first_block=int(f),
            item_start=int(i),
            item_end=int(e),
        )
        for c, f, l, i, e in zip(out_chunk, out_first, out_last, out_item, item_end)
    ]

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
    layer: int | None = None,
    max_chunk_requests: int = 1,
) -> QueryResult:
    if not isinstance(arr, zarr.Array):
        raise TypeError("query_gt_async expects an opened zarr.Array")

    if request_batch_size is None:
        request_batch_size = max(request_concurrency, request_concurrency * 4)

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
    unit_ids, ranks, n_in, n_out = await asyncio.to_thread(
        native().plan,
        plan_meta(meta, lt, layer),
        int(np.prod(lt.unit_grid)),
        lt.meta_size - 4 * (len(lt.layers) - 1),
        lt.blocks_per_unit,
        float(threshold),
        threads,
    )
    chunk_ids, block_ids = unit_to_chunk(lt, unit_ids, ranks)
    plan_sec = time.perf_counter() - t1

    per_layer = [
        merge_ranges(keys, chunk_ids, block_ids, lt.layer_bytes[j], merge_gap_blocks, lt.layer_starts[j])
        for j in range(layer + 1)
    ]
    sorted_blocks = per_layer[0][1]
    order = np.lexsort((block_ids, chunk_ids))
    sorted_chunks = chunk_ids[order]
    cols = np.cumsum((0, *lt.layer_bytes[: layer + 1]))

    # 分層時，逐層讀會在同一個 chunk 發多個 request；超過門檻就改讀 chunk 開頭的連續一段
    dense: set[str] = set()
    if layer > 0:
        per_key = collections.Counter(req.key for merged, _ in per_layer for req in merged)
        dense = {k for k, n in per_key.items() if n > max_chunk_requests}
    prefix_end = int(cols[-1]) * lt.blocks_per_chunk

    if layer == 0:
        merged0, sorted0 = per_layer[0]
        t1 = time.perf_counter()
        n_maybe_val, payload_bytes, decode_sec = await stream_count(
            store,
            merged0,
            sorted0,
            lt.layer_bytes[0],
            rate,
            lt.block_dim,
            float(threshold),
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
                    payload_reqs.append(_Range(key=req.key, start=req.start, end=req.end))
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
                rows = sorted_blocks[item.item_start : item.item_end].astype(np.int64) - item.first_block
                out[item.item_start : item.item_end, cols[j] : cols[j + 1]] = (
                    buf.reshape(-1, lt.layer_bytes[j])[rows]
                )
            else:
                c = chunk_of[item]
                lo, hi = np.searchsorted(sorted_chunks, [c, c + 1])
                ids = sorted_blocks[lo:hi].astype(np.int64)
                for jj in range(layer + 1):
                    seg = buf[lt.layer_starts[jj] : lt.layer_starts[jj] + lt.blocks_per_chunk * lt.layer_bytes[jj]]
                    out[lo:hi, cols[jj] : cols[jj + 1]] = seg.reshape(-1, lt.layer_bytes[jj])[ids]
        blocks = out.reshape(-1)

        t1 = time.perf_counter()
        n_maybe_val = await asyncio.to_thread(
            native().count,
            blocks,
            len(sorted_blocks),
            block_size,
            rate,
            lt.block_dim,
            float(threshold),
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
            layer=layer,
            max_chunk_requests=max_chunk_requests,
        )
    )
