"""What it costs to build a SkipZFP store: encoding, the zone map, memory and bytes.

usage: python build_cost.py run <raw.zarr> <out.jsonl> [--layouts k8t,k8t-b4]
                                [--tmp DIR] [--reps 2] [--nt N]
       python build_cost.py one <raw.zarr> <var> <layout> <tmp> [<nt>]

Each measurement runs in a fresh process on a local disk under --tmp: it writes the
payload (encoding every chunk), then builds the zone map with write_meta, and
records the seconds of each step, the process's peak resident memory (VmHWM), and
the bytes of the payload and of the zone map. The raw variable is first copied to a
memory map by the parent, which is not measured. Memory is sampled every 20 ms
while building: RssAnon is what the build allocates, RssFile the pages of the
memory-mapped input it has touched. Nothing is uploaded. How many queries repay the
zone map follows from these seconds and the query times of exp2 and the ablation.
Each row records its run_id; <out.jsonl>.runs.json holds the code, machine,
packages and data.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from exp2_threshold_query import peak_rss_kb

from common import SHARD, VARS, codec, load_era5, skipzfp_codec, start_run, write_slabs


class Sampler:
    """Track the peak RssAnon and RssFile of this process from a thread."""

    def __init__(self, every=0.02):
        import threading

        self.peak = dict(RssAnon=0, RssFile=0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(every,), daemon=True)

    def _run(self, every):
        while not self._stop.is_set():
            for line in open("/proc/self/status"):
                name = line.split(":")[0]
                if name in self.peak:
                    self.peak[name] = max(self.peak[name], int(line.split()[1]))
            self._stop.wait(every)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def one(raw, var, layout, tmp, nt=None):
    import zarr

    tmp = Path(tmp)
    a = load_era5(raw, var, nt, cache=tmp / "cache")  # made by run(), only opened
    path = tmp / f"{var}_{layout}.zarr"
    shutil.rmtree(path, ignore_errors=True)
    g = zarr.open_group(str(path), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHARD,
        dtype="float32",
        serializer=skipzfp_codec(layout),
        compressors=None,
        filters=None,
    )
    with Sampler() as s_data:
        t0 = time.perf_counter()
        write_slabs(z, a)
        t_data = time.perf_counter() - t0
    with Sampler() as s_meta:
        t0 = time.perf_counter()
        codec.write_meta(z, a)
        t_meta = time.perf_counter() - t0
    out = dict(
        shape=list(a.shape),
        values=int(a.size),
        payload_seconds=t_data,
        meta_seconds=t_meta,
        payload_bytes=dir_bytes(path / "data"),
        meta_bytes=dir_bytes(path / "data_meta"),
        payload_peak_anon_kb=s_data.peak["RssAnon"],
        payload_peak_file_kb=s_data.peak["RssFile"],
        meta_peak_anon_kb=s_meta.peak["RssAnon"],
        meta_peak_file_kb=s_meta.peak["RssFile"],
        peak_rss_kb=peak_rss_kb(),
    )
    shutil.rmtree(path)
    print(json.dumps(out))


def run(raw, out_path, layouts, tmp, reps, nt=None):
    run_id = start_run(out_path, raw, script="build_cost", layouts=layouts, nt=nt)
    Path(tmp, "cache").mkdir(parents=True, exist_ok=True)
    for var in VARS:  # the memory-mapped copies the measurements open
        load_era5(raw, var, nt, cache=Path(tmp, "cache"))
    jobs = [(v, lay, r) for r in range(reps) for v in VARS for lay in layouts]
    for var, layout, rep in jobs:
        cmd = [sys.executable, __file__, "one", str(raw), var, layout, str(tmp)]
        if nt:
            cmd.append(str(nt))
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print("failed", var, layout, p.stderr[-2000:], flush=True)
            continue
        row = dict(var=var, layout=layout, rep=rep, run_id=run_id)
        row.update(json.loads(p.stdout.strip().splitlines()[-1]))
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(var, layout, rep, f"meta {row['meta_seconds']:.1f}s", flush=True)
    shutil.rmtree(Path(tmp, "cache"), ignore_errors=True)
    print("finished", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        nt = int(sys.argv[6]) if len(sys.argv) > 6 else None
        one(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], nt)
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--layouts", default="k8t,k8t-b4")
    ap.add_argument("--tmp", default="build_tmp")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--nt", type=int)
    args = ap.parse_args()
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    run(args.raw, args.out, args.layouts.split(","), args.tmp, args.reps, args.nt)
