"""Sensitivity to object size: threshold queries on k64t (512-step objects) vs k8.

usage: python sensitivity_object_size.py run <raw.zarr> <root> <out.jsonl> [--reps 5]
       python sensitivity_object_size.py one <url> <threshold> <split_bytes>

Uses the one-year stores from `prepare_stores.py --suite point`. Larger objects give far
higher point-query throughput (see exp1) but fewer, larger ranges for threshold
queries. For k64t, merged ranges longer than <split_bytes> are cut at block boundaries
(4 MB here) to restore parallelism; this splitting is an experiment-only patch of
skipzfp.query.merge_ranges and is not part of the library.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import VARS, load_era5, query

SELECTIVITIES = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
CONFIGS = {"k8": ("k8", 0), "k64t_split4": ("k64t", 4 << 20)}


def patch_split(split):
    orig = query.merge_ranges

    def merge_split(keys, chunk_ids, block_ids, block_size, gap, base=0):
        merged, blocks = orig(keys, chunk_ids, block_ids, block_size, gap, base)
        if split <= 0:
            return merged, blocks
        out = []
        for r in merged:
            if r.end - r.start <= split:
                out.append(r)
                continue
            items = blocks[r.item_start : r.item_end].astype(np.int64)
            i = 0
            while i < len(items):
                first = int(items[i])
                j = int(
                    np.searchsorted(items, first + split // block_size, side="left")
                )
                j = max(j, i + 1)
                last = int(items[j - 1])
                out.append(
                    query._MergedRange(
                        key=r.key,
                        start=base + first * block_size,
                        end=base + (last + 1) * block_size,
                        first_block=first,
                        item_start=r.item_start + i,
                        item_end=r.item_start + j,
                    )
                )
                i = j
        return out, blocks

    query.merge_ranges = merge_split


def one(url, th, split):
    import exp2_threshold_query

    patch_split(split)
    exp2_threshold_query.one(url, "skipzfp", th)


def run(raw, root, out_path, reps):
    ths = {}
    for var in VARS:
        a = load_era5(raw, var)
        ths[var] = [float(np.quantile(a, 1 - s)) for s in SELECTIVITIES]
        del a
    jobs = [
        (v, cfg, s, r)
        for v in VARS
        for cfg in CONFIGS
        for s in SELECTIVITIES
        for r in range(reps)
    ]
    random.Random(21).shuffle(jobs)
    try:
        done = {
            tuple(json.loads(line)[k] for k in ("var", "cfg", "sel", "rep"))
            for line in open(out_path)
        }
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, (v, cfg, s, r) in enumerate(todo, 1):
        layout, split = CONFIGS[cfg]
        th = ths[v][SELECTIVITIES.index(s)]
        url = f"{root}/stores_sub/{v}/{layout}.zarr"
        cmd = [sys.executable, str(Path(__file__)), "one", url, repr(th), str(split)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            print("failed", v, cfg, s, p.stderr[-800:], flush=True)
            continue
        result = json.loads(p.stdout.strip().splitlines()[-1])
        rec = dict(var=v, cfg=cfg, sel=s, rep=r, threshold=th, **result)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if i % 35 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("raw")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    run(args.raw, args.root, args.out, args.reps)
