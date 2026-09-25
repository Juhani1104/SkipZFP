"""Experiment 1: point queries, the full time series of grid points (one year of ERA5).

usage: python exp1_point_query.py run <root> <out.jsonl> [--reps 3]
       python exp1_point_query.py check <raw.zarr> <root> [--nt N]
       python exp1_point_query.py one <root> <var> <method> <size> <mode> <n> <seed>
                                      [conc]

<root> holds the stores built by `prepare_stores.py --suite point`.
batch       random / cluster: N points in one query, each needed unit read once
latency     N single-point queries one after another
throughput  N single-point queries issued together
SkipZFP computes block offsets instead of reading metadata; zstd and ZFP read whole
chunks; Zarr sharding reads inner chunks, either reading the shard index first (cold)
or with the index already cached (warm).
"""

import argparse
import asyncio
import ctypes
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
import zfpy
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype

import baselines
from common import (
    GAP_BYTES,
    SHARD,
    codec,
    load_era5,
    open_counted,
    query,
    store_url,
)

P = default_buffer_prototype()
POOL = ThreadPoolExecutor(os.cpu_count() or 4)
GAP = GAP_BYTES
METHODS = [
    ("skipzfp", "k8"),
    ("skipzfp", "k8t"),
    ("skipzfp", "k64t"),
    ("zstd", "64x16x32"),
    ("zfpacc", "64x16x32"),
    ("zshard_cold", "16x4x4"),
    ("zshard_cold", "16x16x16"),
    ("zshard_cold", "64x16x32"),
    ("zshard_warm", "16x4x4"),
    ("zshard_warm", "16x16x16"),
    ("zshard_warm", "64x16x32"),
]


def pick(n, mode, seed, ny=128, nx=256):
    rng = np.random.default_rng(seed)
    if mode == "random":
        flat = rng.choice(ny * nx, size=n, replace=False)
        return np.stack(np.unravel_index(flat, (ny, nx)), 1)
    cy, cx = rng.integers(0, ny), rng.integers(0, nx)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    flat = np.argsort(d.reshape(-1), kind="stable")[:n]
    return np.stack(np.unravel_index(flat, (ny, nx)), 1)


async def q_skipzfp(arr, pts, conc, sem=None):
    lt = query.get_layout(arr)
    cdc = query.find_codec(arr)
    unit = lt.unit_shape
    subs = tuple(c // u for c, u in zip(lt.chunk_shape, unit))
    ub = tuple(u // 4 for u in unit)
    rank = codec.block_rank(unit, cdc.block_order, 4)
    n_t = lt.grid_shape[0]
    sy, sx = (
        pts[:, 0] % lt.chunk_shape[1] // unit[1],
        pts[:, 1] % lt.chunk_shape[2] // unit[2],
    )
    by, bx = pts[:, 0] % unit[1] // 4, pts[:, 1] % unit[2] // 4
    bt = np.arange(ub[0])
    native_b = (bt[None, :] * ub[1] + by[:, None]) * ub[2] + bx[:, None]
    pos = np.concatenate(
        [
            np.ravel_multi_index((np.full_like(sy, st), sy, sx), subs)[:, None]
            * lt.blocks_per_unit
            + rank[native_b]
            for st in range(subs[0])
        ],
        axis=1,
    )
    need = np.unique(pos)
    keys = [
        query.full_key(arr, arr.metadata.encode_chunk_key((t, 0, 0)))
        for t in range(n_t)
    ]
    chunk_ids = np.repeat(np.arange(n_t, dtype=np.uint32), len(need))
    block_ids = np.tile(need.astype(np.uint32), n_t)
    merged, sorted_blocks = query.merge_ranges(
        keys, chunk_ids, block_ids, lt.block_size, GAP // lt.block_size
    )
    col = {int(b): i for i, b in enumerate(need)}
    store = arr.store_path.store
    sem = sem or asyncio.Semaphore(conc)
    loop = asyncio.get_running_loop()
    nat = query.native()
    nat.lib.szfp_decode_blocks.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    out = np.empty((n_t, len(need), 64), np.float32)

    def decode(data, req):
        rows = (
            sorted_blocks[req.item_start : req.item_end].astype(np.int64)
            - req.first_block
        )
        buf = np.frombuffer(data, np.uint8).reshape(-1, lt.block_size)[rows]
        vals = np.empty((len(rows), 64), np.float32)
        code = nat.lib.szfp_decode_blocks(
            np.ascontiguousarray(buf).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            len(rows),
            lt.block_size,
            lt.rate,
            4,
            1,
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        assert code == 0, code
        t = keys.index(req.key)
        for j, b in enumerate(sorted_blocks[req.item_start : req.item_end]):
            out[t, col[int(b)]] = vals[j]

    async def one(req):
        data = await query.read_one(
            store, query._Range(req.key, req.start, req.end), sem
        )
        await loop.run_in_executor(POOL, decode, data, req)

    await asyncio.gather(*(one(r) for r in merged))
    ly, lx = pts[:, 0] % 4, pts[:, 1] % 4
    res = np.empty((len(pts), n_t * lt.chunk_shape[0]), np.float32)
    for k in range(len(pts)):
        cols = [col[int(b)] for b in pos[k]]
        v = out[:, cols, :].reshape(n_t, len(cols), 4, 4, 4)
        res[k] = v[:, :, :, ly[k], lx[k]].reshape(-1)
    return res


async def q_chunks(arr, pts, conc, kind, sem=None):
    ch = tuple(int(x) for x in arr.metadata.chunk_grid.chunk_shape)
    n_t = arr.shape[0] // ch[0]
    cyx = np.unique(np.stack([pts[:, 0] // ch[1], pts[:, 1] // ch[2]], 1), axis=0)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cyx)}
    out = np.empty((n_t, len(cyx), *ch), np.float32)
    store = arr.store_path.store
    sem = sem or asyncio.Semaphore(conc)
    loop = asyncio.get_running_loop()
    c = baselines.lib()
    raw = 4 * int(np.prod(ch))

    def decode(data, t, i):
        if kind == "zstd":
            out[t, i] = c.decode_many(data, [0], [len(data)], raw).reshape(ch)
        else:
            out[t, i] = zfpy.decompress_numpy(data).reshape(ch)

    async def one(t, i, cy, cx):
        key = query.full_key(arr, arr.metadata.encode_chunk_key((t, cy, cx)))
        async with sem:
            buf = await store.get(key, prototype=P)
        await loop.run_in_executor(POOL, decode, buf.to_bytes(), t, i)

    await asyncio.gather(
        *(
            one(t, i, int(cy), int(cx))
            for t in range(n_t)
            for i, (cy, cx) in enumerate(cyx)
        )
    )
    res = np.empty((len(pts), n_t * ch[0]), np.float32)
    for k, (y, x) in enumerate(pts):
        res[k] = out[:, idx[(y // ch[1], x // ch[2])], :, y % ch[1], x % ch[2]].reshape(
            -1
        )
    return res


async def q_zshard(arr, pts, conc, index_cache, sem=None):
    inner = tuple(int(x) for x in arr.chunks)
    k = tuple(s // i for s, i in zip(SHARD, inner))
    n_inner = int(np.prod(k))
    n_t = arr.shape[0] // SHARD[0]
    iy, ix = pts[:, 0] // inner[1], pts[:, 1] // inner[2]
    it = np.arange(k[0])
    need = np.unique(
        ((it[None, :] * k[1] + iy[:, None]) * k[2] + ix[:, None]).reshape(-1)
    )
    col = {int(b): i for i, b in enumerate(need)}
    store = arr.store_path.store
    sem = sem or asyncio.Semaphore(conc)
    loop = asyncio.get_running_loop()
    c = baselines.lib()
    raw = 4 * int(np.prod(inner))
    out = np.empty((n_t, len(need), *inner), np.float32)

    async def shard(t):
        key = query.full_key(arr, arr.metadata.encode_chunk_key((t, 0, 0)))
        if index_cache is None:
            async with sem:
                tail = (
                    await store.get(
                        key, prototype=P, byte_range=SuffixByteRequest(n_inner * 16 + 4)
                    )
                ).to_bytes()
            index = np.frombuffer(tail[:-4], "<u8").reshape(n_inner, 2)
        else:
            index = index_cache[t]
        offs, sizes = index[need, 0], index[need, 1]
        order = np.argsort(offs)
        groups, cur = [], [order[0]]
        for a, b in zip(order[:-1], order[1:]):
            if int(offs[b]) - int(offs[a] + sizes[a]) <= GAP:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)

        async def one(g):
            s, e = int(offs[g[0]]), int(offs[g[-1]] + sizes[g[-1]])
            async with sem:
                data = (
                    await store.get(key, prototype=P, byte_range=RangeByteRequest(s, e))
                ).to_bytes()

            def decode():
                v = c.decode_many(data, offs[g] - np.uint64(s), sizes[g], raw).reshape(
                    len(g), *inner
                )
                out[t, g] = v

            await loop.run_in_executor(POOL, decode)

        await asyncio.gather(*(one(g) for g in groups))

    await asyncio.gather(*(shard(t) for t in range(n_t)))
    res = np.empty((len(pts), n_t * SHARD[0]), np.float32)
    for kk, (y, x) in enumerate(pts):
        cols = [col[int((i * k[1] + y // inner[1]) * k[2] + x // inner[2])] for i in it]
        res[kk] = out[:, cols][:, :, :, y % inner[1], x % inner[2]].reshape(-1)
    return res


def load_index(arr):
    idx = zarr.open_array(store=arr.store_path.store, path="data_index", mode="r")
    return np.asarray(idx[...], np.uint64).reshape(idx.shape[0], -1, 2)


def runner(arr, method, conc):
    if method == "skipzfp":
        return lambda pts, sem=None: q_skipzfp(arr, pts, conc, sem)
    if method in ("zstd", "zfpacc"):
        return lambda pts, sem=None: q_chunks(arr, pts, conc, method, sem)
    cache = load_index(arr) if method == "zshard_warm" else None
    return lambda pts, sem=None: q_zshard(arr, pts, conc, cache, sem)


def one(root, var, method, size, mode, n, seed, conc=128):
    zarr.config.set({"async.concurrency": conc})
    arr, store = open_counted(store_url(root, var, method, size))
    zarr.core.sync.sync(store.exists("zarr.json"))
    q = runner(arr, method, conc)
    store.requests = store.bytes = 0
    if mode in ("random", "cluster"):
        pts = pick(n, mode, seed)
        t0 = time.perf_counter()
        zarr.core.sync.sync(q(pts))
        out = dict(seconds=time.perf_counter() - t0)
    elif mode == "latency":
        pts = pick(n, "random", seed)
        lat = []
        for p in pts:
            t0 = time.perf_counter()
            zarr.core.sync.sync(q(p[None]))
            lat.append(time.perf_counter() - t0)
        out = dict(
            seconds=float(np.median(lat)), p90=float(np.quantile(lat, 0.9)), lat=lat
        )
    else:
        pts = pick(n, "random", seed)

        async def all_q():
            sem = asyncio.Semaphore(conc)
            return await asyncio.gather(*(q(p[None], sem) for p in pts))

        t0 = time.perf_counter()
        zarr.core.sync.sync(all_q())
        sec = time.perf_counter() - t0
        out = dict(seconds=sec, qps=n / sec)
    out.update(requests=store.requests, bytes=store.bytes)
    print(json.dumps(out))


def check(raw, root, nt=None):
    """Every method must return exactly the stored values of the requested points."""
    a = load_era5(raw, "t2m", nt)
    for method, size in METHODS:
        arr, _ = open_counted(store_url(root, "t2m", method, size))
        lossless = method.startswith("zshard") or method == "zstd"
        ref = a if lossless else np.asarray(arr[:])
        q = runner(arr, method, 64)
        for mode, n in (
            ("random", 1),
            ("random", 37),
            ("cluster", 50),
            ("random", 1000),
        ):
            pts = pick(n, mode, 3)
            got = zarr.core.sync.sync(q(pts))
            want = ref[:, pts[:, 0], pts[:, 1]].T
            assert np.array_equal(got, want), (method, size, mode, n)
        print("ok", method, size, flush=True)


def jobs_for(reps):
    jobs = []
    for method, size in METHODS:
        for r in range(reps):
            for mode in ("random", "cluster"):
                for n in (1, 10, 100, 330, 1000):
                    jobs.append((method, size, mode, n, r))
            jobs.append((method, size, "latency", 20, r))
            jobs.append((method, size, "throughput", 200, r))
    random.Random(1).shuffle(jobs)
    return jobs


def run(root, out_path, reps=3):
    jobs = jobs_for(reps)
    keys = ("method", "size", "mode", "n", "rep")
    try:
        done = {tuple(json.loads(line)[k] for k in keys) for line in open(out_path)}
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, (method, size, mode, n, r) in enumerate(todo, 1):
        cmd = [sys.executable, str(Path(__file__)), "one", root, "t2m", method, size]
        cmd += [mode, str(n), str(100 + r)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print("failed", method, size, mode, n, p.stderr[-1500:], flush=True)
            continue
        out = json.loads(p.stdout.strip().splitlines()[-1])
        if mode != "latency":
            out.pop("lat", None)
        rec = dict(method=method, size=size, mode=mode, n=n, rep=r, **out)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if i % 30 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        a = sys.argv[2:]
        conc = int(a[7]) if len(a) > 7 else 128
        one(a[0], a[1], a[2], a[3], a[4], int(a[5]), int(a[6]), conc)
        sys.exit()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("root")
    r.add_argument("out")
    r.add_argument("--reps", type=int, default=3)
    c = sub.add_parser("check")
    c.add_argument("raw")
    c.add_argument("root")
    c.add_argument("--nt", type=int)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.root, args.out, args.reps)
    else:
        check(args.raw, args.root, args.nt)
