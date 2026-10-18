"""Ablation: change one thing at a time in a SkipZFP threshold query on ten years of ERA5.

usage: python ablation.py run <raw.zarr> <root> <out.jsonl> [--reps 5] [--tail-reps 30]
                              [--nt N] [--cache DIR]

<root> holds the threshold and bits suites from prepare_stores.py. Every config is
the exp2 SkipZFP query (k8t, pipeline on, zero copy, 256 KB merge gap, no memory
limit, COUNT) with exactly one factor changed, so the difference to its reference
(the default, unless REFERENCE names another) is that factor's effect: the pipeline, the copy before decoding, the merge gap, the
block order, 4-bit per-block error bounds, block pruning itself (a full scan with the
same decoder), reading the zone map through zarr instead of by key, and AVG instead
of COUNT, with and without 4-bit bounds and the pipeline. Two more configs record
the timeline of every read with the pipeline on and off. Every job runs in a fresh process through
`exp2_threshold_query.py one`; jobs are shuffled so configs interleave in time, and
a rerun skips jobs already in <out.jsonl>. The default and the pipeline-off configs
are also repeated --tail-reps times at two selectivities, for p95 and p99. The
merge gap of 0 runs at the lowest selectivity only (see ONLY_AT): at higher ones it
needs millions of requests per query.
Each row records its run_id; <out.jsonl>.runs.json holds the code, machine,
packages and data of each run.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from exp2_threshold_query import thresholds

from common import VARS, start_run, store_url

SELECTIVITIES = [0.001, 0.01, 0.1, 0.5]
TAIL_SELECTIVITIES = [0.01, 0.5]

# name: (factor, method, layout, options); options are QueryOptions fields plus "op"
CONFIGS = {
    "default": ("none", "skipzfp", "k8t", {}),
    "pipeline_off": ("pipeline", "skipzfp", "k8t", {"pipeline": False}),
    "zero_copy_off": ("zero_copy", "skipzfp", "k8t", {"zero_copy": False}),
    "gap_0": ("merge_gap", "skipzfp", "k8t", {"merge_gap_blocks": 0}),
    "gap_64k": ("merge_gap", "skipzfp", "k8t", {"merge_gap_blocks": 64_000 // 64}),
    "gap_850k": ("merge_gap", "skipzfp", "k8t", {"merge_gap_blocks": 850_000 // 64}),
    "order_c": ("block_order", "skipzfp", "k8", {}),
    "bits4": ("summary_precision", "skipzfp", "k8t-b4", {}),
    "no_pruning": ("pruning", "fullscan_skipzfp", "k8t", {}),
    "avg": ("operator", "skipzfp", "k8t", {"op": "avg"}),
    "avg_bits4": ("summary_precision", "skipzfp", "k8t-b4", {"op": "avg"}),
    "avg_pipeline_off": (
        "pipeline",
        "skipzfp",
        "k8t",
        {"op": "avg", "pipeline": False},
    ),
    # the pipeline needs the zone map read by key, so this one is run in turn and
    # compared with pipeline_off
    "meta_zarr": (
        "zone_map_read",
        "skipzfp",
        "k8t",
        {"pipeline": False, "meta_by_key": False},
    ),
    # the timeline of every read, with the pipeline on and off, to see where the
    # pipeline's time goes; kept apart so the other rows stay small
    "trace_pipeline_on": ("timeline", "skipzfp", "k8t", {"trace": True}),
    "trace_pipeline_off": (
        "timeline",
        "skipzfp",
        "k8t",
        {"trace": True, "pipeline": False},
    ),
}
# The Zarr sharding baseline's merge gap (its default is 850 KB): a gap of 0 still
# merges inner chunks that sit next to each other, as SkipZFP's gap_0 merges
# adjacent blocks. Each is compared with the same inner chunks at 850 KB.
for _inner in ("16x16x16", "64x16x32"):
    _tag = "zshard" + _inner.split("x")[0]
    for _gap, _name in ((0, "0"), (256_000, "256k"), (850_000, "850k")):
        CONFIGS[f"{_tag}_gap_{_name}"] = (
            "baseline_merge_gap",
            "zshard_izm",
            _inner,
            {"gap_bytes": _gap},
        )
# the config each one is compared with; the rest are compared with default
REFERENCE = {
    c: c.rsplit("_gap_", 1)[0] + "_gap_850k" for c in CONFIGS if c.startswith("zshard")
}
REFERENCE.update(
    avg_bits4="avg",
    avg_pipeline_off="avg",
    meta_zarr="pipeline_off",
    trace_pipeline_off="trace_pipeline_on",
)
TAIL_CONFIGS = ["default", "pipeline_off"]
# Without merging, every block is its own request: ten years of ws at 50% would
# take about 4.7 million requests, over an hour per query at the ~1000 requests a
# second zarr reaches on GCS. gap_0 therefore runs at the lowest selectivity only,
# which already shows what merging saves; gap_64k and gap_850k cover the rest.
ONLY_AT = {"gap_0": [0.001]}
# the traced configs only need a few runs each
ONLY_REPS = {"trace_pipeline_on": 5, "trace_pipeline_off": 5}


def jobs_for(reps, tail_reps):
    jobs = [
        dict(var=v, config=c, sel=s, rep=r)
        for v in VARS
        for c in CONFIGS
        for s in ONLY_AT.get(c, SELECTIVITIES)
        for r in range(min(reps, ONLY_REPS.get(c, reps)))
    ]
    jobs += [
        dict(var=v, config=c, sel=s, rep=r)
        for v in VARS
        for c in TAIL_CONFIGS
        for s in TAIL_SELECTIVITIES
        for r in range(reps, tail_reps)
    ]
    random.Random(11).shuffle(jobs)
    return jobs


def run(raw, root, out_path, reps, tail_reps, nt=None, cache=None):
    run_id = start_run(
        out_path,
        raw,
        script="ablation",
        reps=reps,
        tail_reps=tail_reps,
        configs={k: list(v) for k, v in CONFIGS.items()},
        reference=REFERENCE,
    )
    sels = sorted(set(SELECTIVITIES) | set(TAIL_SELECTIVITIES))
    ths, truth = thresholds(raw, nt, cache=cache, selectivities=sels)
    jobs = jobs_for(reps, tail_reps)

    def key(j):
        return (j["var"], j["config"], j["sel"], j["rep"])

    try:
        done = {key(json.loads(line)) for line in open(out_path)}
    except FileNotFoundError:
        done = set()
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    exp2 = Path(__file__).with_name("exp2_threshold_query.py")
    t0 = time.time()
    for i, j in enumerate(todo, 1):
        factor, method, layout, opts = CONFIGS[j["config"]]
        idx = sels.index(j["sel"])
        th = ths[j["var"]][idx]
        url = store_url(root, j["var"], method, layout)
        cmd = [sys.executable, str(exp2), "one", url, method, repr(th)]
        if opts:
            cmd.append(json.dumps(opts))
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print("failed", j, p.stderr[-2000:], flush=True)
            continue
        result = json.loads(p.stdout.strip().splitlines()[-1])
        row = dict(
            j,
            factor=factor,
            reference=REFERENCE.get(j["config"], "default"),
            method=method,
            layout=layout,
            threshold=th,
            truth=truth[j["var"]][idx],
            run_id=run_id,
            **result,
        )
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
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tail-reps", type=int, default=30)
    ap.add_argument("--nt", type=int)
    ap.add_argument("--cache", help="directory for on-disk copies of the raw data")
    args = ap.parse_args()
    if args.cache:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
    run(args.raw, args.root, args.out, args.reps, args.tail_reps, args.nt, args.cache)
