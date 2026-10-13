"""Write SkipZFP k8t stores whose shards span T time steps, keeping 64x16x32 units.

usage: python prep_shards.py <raw.zarr> <dest> <tmp> [--steps 64,128,256,512] [--vars t2m,ws]
                              [--nt N]

Every store is trimmed to a whole number of the largest shard, so all sizes hold the
same data. Each one is uploaded to <dest>/stores_sub/<var>/T<steps>.zarr.
"""

import argparse
import shutil
from pathlib import Path

import zarr

from skipzfp import codec

from common import SHARD, SUB_CHUNK, load_era5, upload

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("dest")
ap.add_argument("tmp")
ap.add_argument("--steps", default="64,128,256,512")
ap.add_argument("--vars", default="t2m,ws")
ap.add_argument("--nt", type=int)
a = ap.parse_args()
steps = [int(s) for s in a.steps.split(",")]
for var in a.vars.split(","):
    x = load_era5(a.raw, var, a.nt)
    x = x[: x.shape[0] // max(steps) * max(steps)]
    print(var, x.shape, flush=True)
    for t in steps:
        path = Path(a.tmp, f"T{t}.zarr")
        shutil.rmtree(path, ignore_errors=True)
        g = zarr.open_group(str(path), mode="w")
        z = g.create_array(
            "data",
            shape=x.shape,
            chunks=(t, *SHARD[1:]),
            dtype="float32",
            serializer=codec.SkipZFPCodec(
                rate=8.0, sub_chunk=SUB_CHUNK, block_order=(1, 2, 0)
            ),
            compressors=None,
            filters=None,
        )
        z[:] = x
        codec.write_meta(z, x, t_chunk=16)
        upload(path, f"{a.dest}/stores_sub/{var}")
        shutil.rmtree(path)
        print(var, f"T{t}", "done", flush=True)
