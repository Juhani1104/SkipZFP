"""A/B test SkipZFP threshold queries across code versions on the same store.

usage: python ab_versions.py <root> <exp2.jsonl> <out.jsonl> NAME=PYTHON:PYTHONPATH ...
       [--sels 0.001,0.01,0.1] [--conc 128,256,512] [--reps 3] [--var t2m]
       [--size k8t[,T128,...]] [--gap 256000]

<exp2.jsonl> is an exp2_threshold_query.py result, used for its thresholds. Each NAME
runs its queries with that Python and PYTHONPATH, for example a checkout of another
branch.

Versions run in a shuffled order inside every repetition, each query in a fresh
process, so slow drift in the network hits every version alike.
"""

import argparse
import json
import os
import random
import subprocess

ONE = r"""
import json, sys
sys.path.insert(0, sys.argv[5])
from common import open_counted
from skipzfp import query
import zarr
arr, store = open_counted(sys.argv[1])
zarr.core.sync.sync(store.exists("zarr.json"))
store.requests = store.bytes = 0
r = query.query_gt(arr, float(sys.argv[2]), options=query.QueryOptions(
    request_concurrency=int(sys.argv[3]), merge_gap_blocks=int(sys.argv[4]) // 64))
print(json.dumps(dict(seconds=r.total_seconds, count=r.count, maybe_blocks=r.maybe_blocks,
    metadata_seconds=r.metadata_read_seconds, planning_seconds=r.planning_seconds,
    payload_seconds=r.payload_read_seconds, requests=store.requests, bytes=store.bytes)))
"""

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("exp2")
ap.add_argument("out")
ap.add_argument("versions", nargs="+")
ap.add_argument("--sels", default="0.001,0.01,0.1")
ap.add_argument("--conc", default="128,256,512")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--var", default="t2m")
ap.add_argument("--size", default="k8t")  # comma list
ap.add_argument("--gap", type=int, default=256_000)
ap.add_argument("--exp", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()

ths = {}
for line in open(a.exp2):
    r = json.loads(line)
    if r["var"] == a.var:
        ths[r["sel"]] = (r["threshold"], r["truth"])
versions = dict(v.split("=", 1) for v in a.versions)
jobs = [
    (s, c, z)
    for s in map(float, a.sels.split(","))
    for c in map(int, a.conc.split(","))
    for z in a.size.split(",")
]
for rep in range(a.reps):
    random.shuffle(jobs)
    for sel, conc, size in jobs:
        url = f"{a.root}/stores_sub/{a.var}/{size}.zarr"
        order = list(versions)
        random.shuffle(order)
        for name in order:
            py, path = versions[name].split(":", 1)
            env = dict(os.environ, PYTHONPATH=path)
            th, truth = ths[sel]
            p = subprocess.run(
                [py, "-c", ONE, url, repr(th), str(conc), str(a.gap), a.exp],
                capture_output=True,
                text=True,
                env=env,
                cwd=a.exp,
            )
            if p.returncode:
                print("failed", name, sel, conc, p.stderr[-1500:], flush=True)
                continue
            res = json.loads(p.stdout.strip().splitlines()[-1])
            row = dict(
                version=name,
                var=a.var,
                size=size,
                sel=sel,
                conc=conc,
                rep=rep,
                truth=truth,
                **res,
            )
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
