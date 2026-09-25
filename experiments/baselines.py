"""Baseline threshold queries: count(x > T), streamed and decoded in C.

chunk_zm   chunk-level min/max; skip whole chunks, read the rest (zstd, zfpacc)
inner_zm   Zarr sharding: inner-chunk min/max + a stored shard index;
           read only MAYBE inner chunks, merging neighbours into one request
full_scan  no metadata; read and decode everything (zstd, zfpacc, skipzfp)
"""

import asyncio
import ctypes
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs.blosc
import numpy as np
import zarr
import zfpy
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype

from common import EMPTY, GAP_BYTES, query

HERE = Path(__file__).resolve().parent
POOL = ThreadPoolExecutor(os.cpu_count() or 4)


class _C:
    def __init__(self):
        src = HERE / "blosc_count.c"
        so = src.with_suffix(".so")
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            cmd = ["gcc", "-O3", "-fPIC", "-shared", str(src), "-o", str(so)]
            subprocess.run(cmd, check=True)
        self.lib = ctypes.CDLL(str(so))
        self.lib.bc_count_gt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.bc_count_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        self.lib.bc_count_f32.restype = ctypes.c_size_t
        self.lib.bc_decode_many.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        blosc = ctypes.CDLL(numcodecs.blosc.__file__)
        self.dec = ctypes.cast(blosc.blosc_decompress_ctx, ctypes.c_void_p).value

    def blosc(self, data, offs, sizes, raw_nbytes, th):
        offs = np.ascontiguousarray(offs, np.uint64)
        sizes = np.ascontiguousarray(sizes, np.uint64)
        out = ctypes.c_size_t()
        code = self.lib.bc_count_gt(
            self.dec,
            data,
            len(data),
            offs.ctypes.data,
            sizes.ctypes.data,
            len(offs),
            raw_nbytes,
            th,
            ctypes.byref(out),
        )
        if code:
            raise RuntimeError(f"bc_count_gt failed: {code}")
        return int(out.value)

    def decode_many(self, data, offs, sizes, raw_nbytes):
        offs = np.ascontiguousarray(offs, np.uint64)
        sizes = np.ascontiguousarray(sizes, np.uint64)
        out = np.empty(len(offs) * raw_nbytes // 4, np.float32)
        code = self.lib.bc_decode_many(
            self.dec,
            data,
            len(data),
            offs.ctypes.data,
            sizes.ctypes.data,
            len(offs),
            raw_nbytes,
            out.ctypes.data,
        )
        if code:
            raise RuntimeError(f"bc_decode_many failed: {code}")
        return out

    def f32(self, a, th):
        a = np.ascontiguousarray(a, np.float32)
        return int(self.lib.bc_count_f32(a.ctypes.data, a.size, th))


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = _C()
    return _LIB


async def open_side(arr, name):
    return await zarr.api.asynchronous.open_array(
        store=arr.store_path.store, path=name, mode="r"
    )


async def read_full(store, keys, conc):
    sem = asyncio.Semaphore(conc)

    async def one(k):
        async with sem:
            buf = await store.get(k, prototype=default_buffer_prototype())
        return buf.to_bytes()

    return await asyncio.gather(*(one(k) for k in keys))


async def stream(store, reqs, conc, work):
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(conc)

    def timed(data, item):
        t0 = time.perf_counter()
        n = work(data, item)
        return n, time.perf_counter() - t0

    async def one(key, rng, item):
        async with sem:
            buf = await store.get(
                key,
                prototype=default_buffer_prototype(),
                byte_range=RangeByteRequest(*rng) if rng else None,
            )
        if buf is None:
            raise FileNotFoundError(key)
        return await loop.run_in_executor(POOL, timed, buf.to_bytes(), item)

    parts = await asyncio.gather(*(one(*r) for r in reqs))
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def chunk_counter(kind, raw_nbytes, th):
    c = lib()
    if kind == "zstd":
        return lambda data, _: c.blosc(data, [0], [len(data)], raw_nbytes, th)
    return lambda data, _: c.f32(zfpy.decompress_numpy(data), th)


def _keys(arr, coords):
    return [
        query.full_key(arr, arr.metadata.encode_chunk_key(tuple(int(x) for x in c)))
        for c in coords
    ]


async def chunk_zm(arr, th, conc, kind):
    t0 = time.perf_counter()
    stats_arr = await open_side(arr, "data_stats")
    stats = np.asarray(await stats_arr.getitem(...), np.float64)
    eps = float(stats_arr.attrs.get("eps") or 0.0)
    t_meta = time.perf_counter() - t0

    lo, hi = stats[..., 0] - eps, stats[..., 1] + eps
    is_in = lo > th
    maybe = ~is_in & (hi > th)
    n_val = int(np.prod(arr.metadata.chunk_grid.chunk_shape))
    count = int(is_in.sum()) * n_val

    keys = _keys(arr, np.argwhere(maybe))
    t1 = time.perf_counter()
    n, t_dec = await stream(
        arr.store_path.store,
        [(k, None, None) for k in keys],
        conc,
        chunk_counter(kind, 4 * n_val, th),
    )
    return dict(
        seconds=time.perf_counter() - t0,
        count=count + n,
        in_blocks=int(is_in.sum()),
        out_blocks=int((~is_in & ~maybe).sum()),
        maybe_blocks=len(keys),
        metadata_seconds=t_meta,
        payload_seconds=time.perf_counter() - t1,
        decode_seconds=t_dec,
    )


async def full_scan(arr, th, conc, kind):
    t0 = time.perf_counter()
    shape = tuple(int(x) for x in arr.metadata.chunk_grid.chunk_shape)
    coords = np.argwhere(np.ones([s // c for s, c in zip(arr.shape, shape)], bool))
    keys = _keys(arr, coords)
    if kind == "skipzfp":
        lt = query.get_layout(arr)
        offs = np.arange(lt.blocks_per_chunk, dtype=np.uint64) * np.uint64(
            lt.block_size
        )
        nat = query.native()

        def work(data, _):
            return nat.count_offsets(
                data, offs, lt.block_size, lt.rate, lt.block_dim, th
            )

    else:
        work = chunk_counter(kind, 4 * int(np.prod(shape)), th)
    n, t_dec = await stream(
        arr.store_path.store, [(k, None, None) for k in keys], conc, work
    )
    sec = time.perf_counter() - t0
    return dict(
        seconds=sec,
        count=n,
        maybe_blocks=len(keys),
        metadata_seconds=0.0,
        payload_seconds=sec,
        decode_seconds=t_dec,
    )


async def inner_zm(arr, th, conc):
    t0 = time.perf_counter()
    stats_arr, index_arr = await asyncio.gather(
        open_side(arr, "data_stats"), open_side(arr, "data_index")
    )
    stats, index = await asyncio.gather(stats_arr.getitem(...), index_arr.getitem(...))
    grid = stats.shape[:-2]
    n_inner = stats.shape[-2]
    stats = np.asarray(stats, np.float64).reshape(-1, n_inner, 2)
    index = np.asarray(index, np.uint64).reshape(-1, n_inner, 2)
    t_meta = time.perf_counter() - t0

    t1 = time.perf_counter()
    lo, hi = stats[..., 0], stats[..., 1]
    is_in = lo > th
    maybe = ~is_in & (hi > th)
    n_val = int(np.prod(arr.chunks))
    count = int(is_in.sum()) * n_val

    shard, inner = np.nonzero(maybe)
    offs, sizes = index[shard, inner, 0], index[shard, inner, 1]
    empty = offs == EMPTY
    count += int(empty.sum()) * n_val * (float(arr.metadata.fill_value) > th)
    shard, offs, sizes = shard[~empty], offs[~empty], sizes[~empty]

    order = np.lexsort((offs, shard))
    shard, offs, sizes = shard[order], offs[order], sizes[order]
    ends = offs + sizes
    brk = np.ones(len(offs), bool)
    if len(offs) > 1:
        run_end = np.maximum.accumulate(ends)
        brk[1:] = (shard[1:] != shard[:-1]) | (
            offs[1:].astype(np.int64) - run_end[:-1].astype(np.int64) > GAP_BYTES
        )
    starts = np.flatnonzero(brk)
    stops = np.r_[starts[1:], len(offs)]
    shard_keys = (
        _keys(arr, np.stack(np.unravel_index(shard[starts], grid), 1))
        if len(starts)
        else []
    )
    reqs = []
    for k, a, b in zip(shard_keys, starts, stops):
        s = int(offs[a])
        e = int(ends[a:b].max())
        reqs.append((k, (s, e), (offs[a:b] - np.uint64(s), sizes[a:b])))
    t_plan = time.perf_counter() - t1

    c = lib()
    t1 = time.perf_counter()
    n, t_dec = await stream(
        arr.store_path.store,
        reqs,
        conc,
        lambda data, it: c.blosc(data, it[0], it[1], 4 * n_val, th),
    )
    return dict(
        seconds=time.perf_counter() - t0,
        count=count + n,
        in_blocks=int(is_in.sum()),
        out_blocks=int((~is_in & ~maybe).sum()),
        maybe_blocks=int(maybe.sum()),
        metadata_seconds=t_meta,
        planning_seconds=t_plan,
        payload_seconds=time.perf_counter() - t1,
        decode_seconds=t_dec,
    )
