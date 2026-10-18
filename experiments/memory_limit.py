"""Memory: run each method's threshold query under the same cgroup memory limit.

usage: python memory_limit.py run <raw.zarr> <root> <out.jsonl>
                                  [--limits 1G,2G,4G,none] [--reps 3] [--nt N] [--cache DIR]

<root> holds the threshold suite from prepare_stores.py. Every job runs
`exp2_threshold_query.py one` in a fresh systemd scope with MemoryMax set to the
limit and swap off, so every method gets the same limit. A job killed for using
too much memory exits with 137 and is recorded as oom. Each row has the wall time
seen from outside, ru_maxrss of the measuring process (resident memory, no page
cache) and, from the cgroup, memory.peak (page cache included) with the file and
anon split at the end. SkipZFP also runs with a 128 MB limit on in-flight payload
bytes, which bounds what it holds between reading and decoding. Each row records
its run_id; <out.jsonl>.runs.json holds the code, machine, packages and data.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from exp2_threshold_query import thresholds

from common import VARS, start_run, store_url

SELECTIVITIES = [0.01, 0.5]
INFLIGHT_CAP = 128 * 2**20
# name: (method, size, options for SkipZFP)
METHODS = {
    "skipzfp": ("skipzfp", "k8t", {}),
    "skipzfp_cap": ("skipzfp", "k8t", {"max_inflight_bytes": INFLIGHT_CAP}),
    "skipzfp_avg": ("skipzfp", "k8t", {"op": "avg"}),
    "zstd_zm": ("zstd_zm", "64x128x256", {}),
    "zfpacc_zm": ("zfpacc_zm", "64x128x256", {}),
    "zshard_izm": ("zshard_izm", "16x16x16", {}),
    "fullscan_zstd": ("fullscan_zstd", "64x128x256", {}),
}


def scope_prefix(limit):
    """The command prefix that runs a job in its own scope with this memory limit.

    It uses sudo when it can, since a job left running after logging out of a VM
    loses the user's systemd manager, and the user's manager otherwise, as on a
    shared machine without sudo.
    """
    props = ["-p", "MemorySwapMax=0"]
    if limit != "none":
        props += ["-p", f"MemoryMax={limit}"]
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode:
        return ["systemd-run", "--user", "--scope", "--quiet", *props, "--"]
    return [
        "sudo",
        "systemd-run",
        "--scope",
        "--quiet",
        f"--uid={os.getuid()}",
        f"--gid={os.getgid()}",
        *props,
        "--",
    ]


def run(raw, root, out_path, limits, reps, nt=None, cache=None):
    run_id = start_run(
        out_path,
        raw,
        script="memory_limit",
        limits=limits,
        reps=reps,
        methods={k: list(v) for k, v in METHODS.items()},
    )
    ths, truth = thresholds(raw, nt, cache=cache, selectivities=SELECTIVITIES)
    jobs = [
        dict(var=v, name=m, sel=s, limit=lim, rep=r)
        for v in VARS
        for m in METHODS
        for s in SELECTIVITIES
        for lim in limits
        for r in range(reps)
    ]
    random.Random(12).shuffle(jobs)

    def key(j):
        return (j["var"], j["name"], j["sel"], j["limit"], j["rep"])

    try:
        done = {key(json.loads(line)) for line in open(out_path)}
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    exp2 = Path(__file__).with_name("exp2_threshold_query.py")
    prefixes = {lim: scope_prefix(lim) for lim in limits}
    t0 = time.time()
    for i, j in enumerate(todo, 1):
        method, size, opts = METHODS[j["name"]]
        idx = SELECTIVITIES.index(j["sel"])
        th = ths[j["var"]][idx]
        url = store_url(root, j["var"], method, size)
        cmd = [*prefixes[j["limit"]], sys.executable, str(exp2), "one", url, method]
        cmd.append(repr(th))
        if opts:
            cmd.append(json.dumps(opts))
        t1 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t1
        row = dict(
            j,
            method=method,
            size=size,
            threshold=th,
            truth=truth[j["var"]][idx],
            run_id=run_id,
            exit_code=p.returncode,
            oom=p.returncode in (137, -9),
            outside_seconds=wall,
        )
        if p.returncode == 0:
            row.update(json.loads(p.stdout.strip().splitlines()[-1]))
        elif not row["oom"]:
            print("failed", j, p.stderr[-2000:], flush=True)
            continue
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if i % 25 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("raw")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--limits", default="1G,2G,4G,none")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nt", type=int)
    ap.add_argument("--cache", help="directory for on-disk copies of the raw data")
    args = ap.parse_args()
    if args.cache:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
    limits = args.limits.split(",")
    run(args.raw, args.root, args.out, limits, args.reps, args.nt, args.cache)
