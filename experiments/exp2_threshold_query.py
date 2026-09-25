"""Experiment 2: threshold query count(x > T) on ten years of ERA5 (t2m, wind speed).

usage: python exp2_threshold_query.py run <raw.zarr> <root> <out.jsonl> [--reps 5]
                                          [--layouts k8t,k8] [--nt N]
       python exp2_threshold_query.py one <url> <method> <threshold>

<root> holds the stores built by `prepare_stores.py --suite threshold`. Every
measurement runs in a fresh process; jobs are shuffled so methods interleave, and a
rerun skips jobs already in <out.jsonl>.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zarr

import baselines
from common import (
    CONCURRENCY,
    GAP_BYTES,
    VARS,
    load_era5,
    open_counted,
    query,
    store_url,
)

SELECTIVITIES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
FULL_SCAN_SELECTIVITY = 0.01
BASELINES = [
    ("zstd_zm", "64x64x128"),
    ("zstd_zm", "64x128x256"),
    ("zfpacc_zm", "64x64x128"),
    ("zfpacc_zm", "64x128x256"),
    ("zshard_izm", "64x16x32"),
    ("zshard_izm", "16x16x16"),
]
FULL_SCANS = [
    ("fullscan_zstd", "64x128x256"),
    ("fullscan_zfpacc", "64x128x256"),
    ("fullscan_skipzfp", "k8t"),
]


def one(url, method, th):
    zarr.config.set({"async.concurrency": CONCURRENCY})
    arr, store = open_counted(url)
    zarr.core.sync.sync(store.exists("zarr.json"))
    store.requests = store.bytes = 0
    sync = zarr.core.sync.sync
    if method == "skipzfp":
        r = query.query_gt(
            arr, th, request_concurrency=CONCURRENCY, merge_gap_blocks=GAP_BYTES // 64
        )
        out = dict(
            seconds=r.total_seconds,
            count=r.count,
            in_blocks=r.in_blocks,
            out_blocks=r.out_blocks,
            maybe_blocks=r.maybe_blocks,
            metadata_seconds=r.metadata_read_seconds,
            planning_seconds=r.planning_seconds,
            payload_seconds=r.payload_read_seconds,
            decode_seconds=r.decode_seconds,
        )
    elif method in ("zstd_zm", "zfpacc_zm"):
        out = sync(baselines.chunk_zm(arr, th, CONCURRENCY, method[:-3]))
    elif method == "zshard_izm":
        out = sync(baselines.inner_zm(arr, th, CONCURRENCY))
    else:
        kind = method.removeprefix("fullscan_")
        out = sync(baselines.full_scan(arr, th, CONCURRENCY, kind))
    out.update(requests=store.requests, bytes=store.bytes)
    print(json.dumps(out))


def thresholds(raw, nt=None, step=64 * 16):
    ths, truth = {}, {}
    for var in VARS:
        a = load_era5(raw, var, nt)
        ths[var] = [float(x) for x in np.quantile(a, [1 - s for s in SELECTIVITIES])]
        truth[var] = [0] * len(SELECTIVITIES)
        for s in range(0, a.shape[0], step):
            blk = a[s : s + step].astype(np.float64)
            for i, t in enumerate(ths[var]):
                truth[var][i] += int((blk > t).sum())
        del a
        print(var, "thresholds", ths[var], flush=True)
    return ths, truth


def jobs_for(layouts, reps):
    methods = [("skipzfp", lay) for lay in layouts] + BASELINES
    jobs = [
        dict(var=v, method=m, size=sz, sel=s, rep=r)
        for v in VARS
        for m, sz in methods
        for s in SELECTIVITIES
        for r in range(reps)
    ]
    jobs += [
        dict(var=v, method=m, size=sz, sel=FULL_SCAN_SELECTIVITY, rep=r)
        for v in VARS
        for m, sz in FULL_SCANS
        for r in range(reps)
    ]
    random.Random(10).shuffle(jobs)
    return jobs


def run(raw, root, out_path, reps, layouts, nt=None):
    ths, truth = thresholds(raw, nt)
    jobs = jobs_for(layouts, reps)

    def key(j):
        return (j["var"], j["method"], j["size"], j["sel"], j["rep"])

    try:
        done = {key(json.loads(line)) for line in open(out_path)}
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, j in enumerate(todo, 1):
        idx = SELECTIVITIES.index(j["sel"])
        th = ths[j["var"]][idx]
        url = store_url(root, j["var"], j["method"], j["size"])
        cmd = [sys.executable, str(Path(__file__)), "one", url, j["method"], repr(th)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print("failed", j, p.stderr[-2000:], flush=True)
            continue
        result = json.loads(p.stdout.strip().splitlines()[-1])
        r = dict(j, threshold=th, truth=truth[j["var"]][idx], **result)
        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if i % 25 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("raw")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--layouts", default="k8t,k8")
    ap.add_argument("--nt", type=int)
    args = ap.parse_args()
    run(args.raw, args.root, args.out, args.reps, args.layouts.split(","), args.nt)
