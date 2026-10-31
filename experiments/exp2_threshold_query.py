"""Experiment 2: threshold query count(x > T) on ten years of ERA5 (t2m, wind speed).

usage: python exp2_threshold_query.py run <raw.zarr> <root> <out.jsonl> [--reps 5]
                                          [--layouts k8t,k8] [--nt N] [--vars t2m,ws]
                                          [--tight] [--chunk-root ROOT]
                                          [--gap-sweep] [--thresholds-from F]
                                          [--concurrency-sweep 64,128] [--sels 0.01,0.5]
       python exp2_threshold_query.py one <url> <method> <threshold> [<options>]

<options> is a JSON object for SkipZFP: "op" ("count" or "avg"), "fast_decode" (a
szfp_set_fast_decode mode; 1, everything, by default) plus any QueryOptions field,
such as {"pipeline": false}; for zshard_izm, "gap_bytes", the
merge gap of its reads (default 850 KB). Each row records its run_id;
<out.jsonl>.runs.json holds the code, machine, packages and data of each run.

<root> holds the stores built by `prepare_stores.py --suite threshold`; --vars runs
other variables, such as the CMIP6 precipitation pr of fetch_cmip6.py. --tight also
measures ZFP fixed-accuracy at the tolerance of `prepare_stores.py --suite tight`,
whose counts are as accurate as SkipZFP's, as size <chunk>-tight. --chunk-root
reads the chunked baselines (zstd and ZFP with chunk zone maps, and their full
scans) from another root, such as the threshold suite next to an objects suite.
--gap-sweep also measures Zarr sharding at three more merge gaps and SkipZFP k8t at
1 MB, as sizes such as 64x16x32@256K and k8t@1M (see GAP_OPTS).
--concurrency-sweep measures SkipZFP k8t@1M, Zarr sharding 64x16x32@2M and zstd
64x128x256 at each number of requests in flight, every method with the same one,
at the selectivities of --sels.
--thresholds-from takes the thresholds and exact counts from earlier exp2 results
on the same data instead of computing them, so <raw.zarr> is not read.
Every
measurement runs in a fresh process; jobs are shuffled so methods interleave, and a
rerun skips jobs already in <out.jsonl>.
"""

import argparse
import dataclasses
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zarr

from skipzfp import aggregate_gt
from skipzfp._native import load_library

import baselines
from common import (
    CONCURRENCY,
    SKIPZFP_GAP_BYTES,
    VARS,
    load_era5,
    open_counted,
    quantiles,
    query,
    read_slabs,
    start_run,
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
# the baselines whose storage object is their chunk, so the object size of SkipZFP
# and Zarr sharding leaves them as they are
CHUNKED = {"zstd_zm", "zfpacc_zm", "fullscan_zstd", "fullscan_zfpacc"}
# ZFP fixed-accuracy at the tolerance whose counts are as accurate as SkipZFP's
TIGHT_BASELINES = [
    ("zfpacc_zm", "64x64x128-tight"),
    ("zfpacc_zm", "64x128x256-tight"),
]
# what --gap-sweep adds to exp2's methods, as size@gap: options. Zarr sharding at
# three more merge gaps (bytes; its default is 850 KB) and SkipZFP k8t also at 1 MB
# (64-byte blocks; 4000 is its default 256 KB)
GAP_OPTS = {
    ("skipzfp", "k8t@1M"): {"merge_gap_blocks": 15625},
    **{
        ("zshard_izm", f"{inner}@{tag}"): {"gap_bytes": gap}
        for inner in ("64x16x32", "16x16x16")
        for tag, gap in (("0", 0), ("256K", 256_000), ("2M", 2_000_000))
    },
}
# what --concurrency-sweep measures, with the merge gaps of GAP_OPTS
CONC_SWEEP = [
    ("skipzfp", "k8t@1M"),
    ("zshard_izm", "64x16x32@2M"),
    ("zstd_zm", "64x128x256"),
]
FULL_SCANS = [
    ("fullscan_zstd", "64x128x256"),
    ("fullscan_zfpacc", "64x128x256"),
    ("fullscan_skipzfp", "k8t"),
]


def peak_rss_kb():
    """Peak resident memory of this process, from VmHWM in /proc/self/status.

    Not ru_maxrss, which Linux carries over from the parent across fork and exec,
    so a job started by a large parent would report the parent's peak.
    """
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return None


def cgroup_memory():
    """Peak and current memory of this process's cgroup, page cache included.

    memory.peak counts page cache as well as anonymous memory, unlike ru_maxrss,
    so the file and anon split at the end says how much was cache. Returns {}
    outside a cgroup v2 memory controller.
    """
    try:
        path = open("/proc/self/cgroup").read().strip().split("::")[-1]
        d = Path("/sys/fs/cgroup" + path)
        stat = dict(line.split() for line in open(d / "memory.stat"))
        return dict(
            cgroup_peak_bytes=int((d / "memory.peak").read_text()),
            cgroup_max=(d / "memory.max").read_text().strip(),
            cgroup_file_bytes=int(stat["file"]),
            cgroup_anon_bytes=int(stat["anon"]),
        )
    except (OSError, KeyError, ValueError):
        return {}


def skipzfp_options(opts):
    """QueryOptions for exp2: the defaults below, overridden by opts."""
    base = dict(
        request_concurrency=CONCURRENCY, merge_gap_blocks=SKIPZFP_GAP_BYTES // 64
    )
    return query.QueryOptions(**{**base, **opts})


def one(url, method, th, opts=None):
    """Measure one query in this fresh process and print one JSON line.

    Opening the array, and a baseline's zone map and shard index arrays, is
    timed and counted on its own, then a request for zarr.json opens a
    connection, and only then the query is measured.
    """
    opts = dict(opts or {})
    op = opts.pop("op", "count")
    fast = opts.pop("fast_decode", None)
    if fast is not None:
        load_library().szfp_set_fast_decode(int(fast))
    zarr.config.set({"async.concurrency": CONCURRENCY})
    t0 = time.perf_counter()
    arr, store = open_counted(url)
    side = zarr.core.sync.sync(baselines.open_sides(arr, method))
    open_out = dict(
        open_seconds=time.perf_counter() - t0,
        open_requests=store.requests,
        open_bytes=store.bytes,
    )
    zarr.core.sync.sync(store.exists("zarr.json"))
    store.requests = store.bytes = 0
    sync = zarr.core.sync.sync
    if method == "skipzfp":
        options = skipzfp_options(opts)
        run = aggregate_gt if op == "avg" else query.query_gt
        r = run(arr, th, options=options)
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
            payload_bytes=r.payload_bytes_read,
            useful_payload_bytes=r.useful_payload_bytes,
            metadata_bytes=r.metadata_bytes_read,
            peak_inflight_bytes=r.peak_inflight_bytes,
            op=op,
            fast_decode=1 if fast is None else int(fast),
            options=dataclasses.asdict(options),
        )
        if getattr(r, "trace", None) is not None:
            out["trace"] = r.trace
        if op == "avg":
            out.update(
                sum=r.sum,
                mean=r.mean,
                original_count=list(r.original_count),
                original_mean=r.original_mean and list(r.original_mean),
            )
    elif method == "zshard_izm" and set(opts) <= {"gap_bytes"} and op == "count":
        gap = opts.get("gap_bytes", baselines.GAP_BYTES)
        out = sync(baselines.inner_zm(arr, side, th, CONCURRENCY, gap))
        out["options"] = dict(gap_bytes=gap)
    elif opts or op != "count":
        raise ValueError("options and op apply to skipzfp, and gap_bytes to zshard_izm")
    elif method in ("zstd_zm", "zfpacc_zm"):
        out = sync(baselines.chunk_zm(arr, side, th, CONCURRENCY, method[:-3]))
    else:
        kind = method.removeprefix("fullscan_")
        out = sync(baselines.full_scan(arr, th, CONCURRENCY, kind))
    out.update(
        requests=store.requests,
        bytes=store.bytes,
        peak_rss_kb=peak_rss_kb(),
        **open_out,
        **cgroup_memory(),
    )
    print(json.dumps(out))


def thresholds(
    raw, nt=None, step=64 * 16, cache=None, selectivities=SELECTIVITIES, variables=VARS
):
    """Thresholds for each selectivity and the exact count above each.

    With cache, each variable is read through a memory map on disk, so it never has
    to fit in memory, and the result is kept there as thresholds_<var>.json for a
    rerun on the same data.
    """
    ths, truth = {}, {}
    for var in variables:
        # a rerun on the same data reuses what an earlier run found
        saved = Path(cache, f"thresholds_{var}.json") if cache else None
        key = dict(raw=str(raw), var=var, nt=nt, selectivities=list(selectivities))
        if saved and saved.exists():
            got = json.loads(saved.read_text())
            if got["key"] == key:
                ths[var], truth[var] = got["thresholds"], got["truth"]
                print(var, "thresholds (saved)", ths[var], flush=True)
                continue
        a = load_era5(raw, var, nt, cache=cache)
        ths[var] = [float(x) for x in quantiles(a, [1 - s for s in selectivities])]
        truth[var] = [0] * len(selectivities)
        for _, _, x in read_slabs(a, step):
            blk = x.astype(np.float64)
            for i, t in enumerate(ths[var]):
                truth[var][i] += int((blk > t).sum())
        del a
        if cache:
            Path(cache, f"{var}.npy").unlink()
            saved.write_text(
                json.dumps(dict(key=key, thresholds=ths[var], truth=truth[var]))
            )
        print(var, "thresholds", ths[var], flush=True)
    return ths, truth


def thresholds_from(path, variables=VARS, selectivities=SELECTIVITIES):
    """The thresholds and exact counts recorded in exp2 results, as thresholds returns."""
    seen = {}
    for line in open(path):
        r = json.loads(line)
        th = seen.setdefault((r["var"], r["sel"]), (r["threshold"], r["truth"]))
        if th != (r["threshold"], r["truth"]):
            raise ValueError(f"{path}: two thresholds for {r['var']} at {r['sel']}")
    ths = {v: [seen[(v, s)][0] for s in selectivities] for v in variables}
    truth = {v: [seen[(v, s)][1] for s in selectivities] for v in variables}
    return ths, truth


def jobs_for(layouts, reps, variables=VARS, tight=False, gap_sweep=False):
    methods = [("skipzfp", lay) for lay in layouts] + BASELINES
    if tight:
        methods += TIGHT_BASELINES
    if gap_sweep:
        methods += list(GAP_OPTS)
    jobs = [
        dict(var=v, method=m, size=sz, sel=s, rep=r)
        for v in variables
        for m, sz in methods
        for s in SELECTIVITIES
        for r in range(reps)
    ]
    jobs += [
        dict(var=v, method=m, size=sz, sel=FULL_SCAN_SELECTIVITY, rep=r)
        for v in variables
        for m, sz in FULL_SCANS
        for r in range(reps)
    ]
    random.Random(10).shuffle(jobs)
    return jobs


def run(
    raw,
    root,
    out_path,
    reps,
    layouts,
    nt=None,
    cache=None,
    variables=VARS,
    tight=False,
    chunk_root=None,
    gap_sweep=False,
    from_exp2=None,
    conc_sweep=None,
    sels=None,
):
    run_id = start_run(
        out_path,
        None if from_exp2 else raw,
        script="exp2",
        reps=reps,
        layouts=layouts,
        nt=nt,
        vars=variables,
        tight=tight,
        chunk_root=chunk_root,
        gap_sweep=gap_sweep,
        thresholds_from=from_exp2,
        concurrency_sweep=conc_sweep,
    )
    if from_exp2:
        ths, truth = thresholds_from(from_exp2, variables)
    else:
        ths, truth = thresholds(raw, nt, cache=cache, variables=variables)
    jobs = jobs_for(layouts, reps, variables, tight, gap_sweep)
    if conc_sweep:
        jobs = [
            dict(var=v, method=m, size=sz, sel=sel, rep=r, concurrency=c)
            for v in variables
            for m, sz in CONC_SWEEP
            for sel in sels or SELECTIVITIES
            for c in conc_sweep
            for r in range(reps)
        ]
        random.Random(11).shuffle(jobs)

    def key(j):
        return (
            j["var"],
            j["method"],
            j["size"],
            j["sel"],
            j["rep"],
            j.get("concurrency"),
        )

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
        chunked = j["method"] in CHUNKED and chunk_root
        size = j["size"].partition("@")[0]  # a merge gap of --gap-sweep is not a store
        url = store_url(chunk_root if chunked else root, j["var"], j["method"], size)
        cmd = [sys.executable, str(Path(__file__)), "one", url, j["method"], repr(th)]
        opts = (
            GAP_OPTS.get((j["method"], j["size"])) if gap_sweep or conc_sweep else None
        )
        if opts:
            cmd.append(json.dumps(opts))
        env = dict(os.environ)
        if j.get("concurrency"):
            env["SKIPZFP_CONCURRENCY"] = str(j["concurrency"])
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            print("failed", j, p.stderr[-2000:], flush=True)
            continue
        result = json.loads(p.stdout.strip().splitlines()[-1])
        r = dict(j, threshold=th, truth=truth[j["var"]][idx], run_id=run_id, **result)
        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if i % 25 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        opts = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
        one(sys.argv[2], sys.argv[3], float(sys.argv[4]), opts)
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("raw")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--layouts", default="k8t,k8")
    ap.add_argument("--nt", type=int)
    ap.add_argument("--cache", help="directory for on-disk copies of the raw data")
    ap.add_argument("--vars", default=",".join(VARS))
    ap.add_argument("--tight", action="store_true")
    ap.add_argument("--chunk-root", help="where the chunked baselines are")
    ap.add_argument("--gap-sweep", action="store_true")
    ap.add_argument("--thresholds-from", help="earlier exp2 results on the same data")
    ap.add_argument(
        "--concurrency-sweep", help="requests in flight, such as 64,128,256"
    )
    ap.add_argument("--sels", help="selectivities, such as 0.01,0.5")
    args = ap.parse_args()
    if args.cache:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
    run(
        args.raw,
        args.root,
        args.out,
        args.reps,
        args.layouts.split(","),
        args.nt,
        args.cache,
        args.vars.split(","),
        args.tight,
        args.chunk_root,
        args.gap_sweep,
        args.thresholds_from,
        [int(c) for c in args.concurrency_sweep.split(",")]
        if args.concurrency_sweep
        else None,
        [float(x) for x in args.sels.split(",")] if args.sels else None,
    )
