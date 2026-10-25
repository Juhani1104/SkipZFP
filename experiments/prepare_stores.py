"""Build the stores the experiments read and copy them to a bucket or local root.

usage: python prepare_stores.py <raw.zarr> <dest> --suite threshold|bits|zmeta|objects|tight|point
                                [--tmp DIR]
                                [--nt N] [--vars t2m,ws]

threshold  exp2 (ten years): SkipZFP k8, k8t; zstd and ZFP fixed-accuracy with chunk
           zone maps at 64x64x128 and 64x128x256; Zarr sharding with inner chunks
           64x16x32 and 16x16x16.
bits       the ablation: SkipZFP k8t-b4, which is k8t with a 4-bit error bound
           per block; build it next to the threshold suite.
zmeta      SkipZFP k8-z and k8t-z, which are k8 and k8t with the zone map
           compressed; build it next to the threshold suite.
objects    SkipZFP k8, k8t and Zarr sharding as in the threshold suite, at the object
           size SKIPZFP_OBJECT sets (default 64x128x256); exp2 reads the chunked
           baselines, whose objects are their chunks, from the threshold suite.
tight      ZFP fixed-accuracy with chunk zone maps at 64x64x128 and 64x128x256, at
           the largest tolerance / 2**k whose counts at every exp2 threshold are
           as accurate as SkipZFP's (see tight_tolerance); stored as
           <chunk>-tight next to the threshold suite.
point      exp1 and the sensitivity study (one year): SkipZFP k8, k8t, k64t; zstd and
           ZFP fixed-accuracy at 64x16x32; Zarr sharding with inner chunks 16x4x4,
           16x16x16 and 64x16x32.

The ZFP fixed-accuracy tolerance is the largest error SkipZFP 8 bpv makes on the same
data, so both lossy formats give the same error bound.
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from common import (
    POINT_STEP_MULTIPLE,
    SHARD,
    VARS,
    chunk_stats,
    codec,
    format_kwargs,
    inner_stats,
    load_era5,
    load_point,
    lossless,
    put,
    quantiles,
    read_slabs,
    shape_of,
    shard_index,
    skipzfp_codec,
    upload,
    write_slabs,
)

SUITES = {
    "threshold": dict(
        skipzfp=["k8", "k8t"],
        chunks=["64x64x128", "64x128x256"],
        inners=["64x16x32", "16x16x16"],
    ),
    # the ablation's 4-bit store, the same layout as exp2's k8t plus per-block bounds
    "bits": dict(skipzfp=["k8t-b4"], chunks=[], inners=[]),
    # the threshold suite's stores whose object is SHARD, for another object size
    # set by SKIPZFP_OBJECT; the chunked baselines do not depend on it
    "objects": dict(skipzfp=["k8", "k8t"], chunks=[], inners=["64x16x32", "16x16x16"]),
    # exp2's k8 and k8t with the zone map compressed by codec.pack_meta and zstd
    "zmeta": dict(skipzfp=["k8-z", "k8t-z"], chunks=[], inners=[]),
    # ZFP fixed-accuracy tightened until its counts are as accurate as SkipZFP's
    "tight": dict(skipzfp=[], chunks=[], inners=[], tight=["64x64x128", "64x128x256"]),
    "point": dict(
        skipzfp=["k8", "k8t", "k64t"],
        chunks=["64x16x32"],
        inners=["16x4x4", "16x16x16", "64x16x32"],
    ),
}
K64T_STEPS = POINT_STEP_MULTIPLE


def new_group(path):
    return zarr.open_group(str(path), mode="w")


def write_skipzfp(path, a, layout):
    g = new_group(path)
    if layout == "k64t":
        a = a[: a.shape[0] // K64T_STEPS * K64T_STEPS]
        cdc = codec.SkipZFPCodec(
            rate=8.0, sub_chunk=(K64T_STEPS, 16, 32), block_order=(1, 2, 0)
        )
        chunks = (K64T_STEPS, *SHARD[1:])
    else:
        cdc, chunks = skipzfp_codec(layout), SHARD
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=chunks,
        dtype="float32",
        serializer=cdc,
        compressors=None,
        filters=None,
    )
    write_slabs(z, a)
    codec.write_meta(
        z, a, t_chunk=1 if layout == "k64t" else None, compress=layout.endswith("-z")
    )
    return z


def max_error(z, a, step=64 * 16):
    err = 0.0
    for s in range(0, a.shape[0], step):
        d = z[s : s + step].astype(np.float64) - a[s : s + step]
        err = max(err, float(np.abs(d).max()))
    return err


# exp2's selectivities, whose thresholds tight_tolerance checks
SELECTIVITIES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
TILE = (64, 128, 256)


def tight_tolerance(a, z, tol, selectivities=SELECTIVITIES, halvings=16):
    """The largest tol / 2**k at which ZFP fixed-accuracy counts are as accurate as SkipZFP's.

    At the same largest error as SkipZFP, ZFP fixed-accuracy can count far worse,
    since it spends that error on every block, so a value near a low threshold
    can cross it. For each exp2 threshold this compares the count above it, in
    float64, of the original values a, of SkipZFP's decoded values z, and of ZFP
    fixed-accuracy at tol / 2**k for k = 0, 1, ...; the first k at which ZFP's
    count error is at most SkipZFP's at every threshold wins. ZFP codes each
    4x4x4 block on its own, so 64 x 128 x 256 tiles give the counts of every chunk
    shape. Returns the tolerance and what each step measured.
    """
    import zfpy

    ths = [float(x) for x in quantiles(a, [1 - s for s in selectivities])]

    def above(x):
        x = x.astype(np.float64)
        return np.array([int((x > t).sum()) for t in ths], np.int64)

    def zfp_above(x, t):
        x = np.ascontiguousarray(x)
        y = zfpy.decompress_numpy(zfpy.compress_numpy(x, tolerance=t))
        return above(y)

    truth = np.zeros(len(ths), np.int64)
    ours = np.zeros(len(ths), np.int64)
    for s, e, x in read_slabs(a):
        truth += above(x)
        ours += above(np.asarray(z[s:e]))
    ours_err = np.abs(ours - truth) / truth
    tiles = [
        (slice(t, t + TILE[0]), slice(i, i + TILE[1]), slice(j, j + TILE[2]))
        for t in range(0, a.shape[0], TILE[0])
        for i in range(0, a.shape[1], TILE[1])
        for j in range(0, a.shape[2], TILE[2])
    ]
    tried = []
    with ThreadPoolExecutor(16) as ex:
        for k in range(halvings + 1):
            t = tol / 2**k
            got = sum(ex.map(lambda sl, t=t: zfp_above(a[sl], t), tiles))
            err = np.abs(got - truth) / truth
            ok = bool((err <= ours_err).all())
            tried.append(dict(tolerance=t, count_error=err.tolist(), ok=ok))
            print("tolerance", t, "count error", err.tolist(), ok, flush=True)
            if ok:
                break
        else:
            raise RuntimeError(
                "no tolerance down to tol / 2**halvings is accurate enough"
            )
    info = dict(
        thresholds=ths,
        selectivities=list(selectivities),
        skipzfp_count_error=ours_err.tolist(),
        max_error=tol,
        tried=tried,
    )
    return t, info


def write_chunked(path, a, fmt, chunk, tol):
    g = new_group(path)
    z = g.create_array(
        "data", shape=a.shape, dtype="float32", **format_kwargs(fmt, tol, chunk)
    )
    write_slabs(z, a)
    s = chunk_stats(a, chunk)
    st = g.create_array(
        "data_stats",
        shape=s.shape,
        dtype=s.dtype,
        compressors=None,
        filters=None,
        chunks=s.shape,
    )
    st[...] = s
    st.update_attributes({"eps": tol if fmt == "zfpacc" else 0.0})


def write_sharded(path, a, inner):
    g = new_group(path)
    z = g.create_array(
        "data",
        shape=a.shape,
        dtype="float32",
        shards=SHARD,
        chunks=inner,
        compressors=lossless(),
    )
    write_slabs(z, a)
    grid = tuple(s // c for s, c in zip(a.shape, SHARD))
    n_inner = int(np.prod([s // i for s, i in zip(SHARD, inner)]))
    put(g, "data_stats", inner_stats(a, SHARD, inner))
    put(g, "data_index", shard_index(path / "data", grid, n_inner))


def prepare(raw, dest, suite, tmp, nt=None, variables=VARS):
    zarr.config.set({"async.concurrency": 32})
    spec = SUITES[suite]
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    tols = {}

    stores = {}  # every store's shape, chunks and codecs, uploaded at the end

    def done(path, sub):
        z = zarr.open_array(str(path), path="data", mode="r", zarr_format=3)
        stores[f"{sub}/{path.name}"] = dict(
            shape=list(z.shape),
            chunks=list(z.chunks),
            shards=list(z.shards) if z.shards else None,
            codecs=[c.to_dict() for c in z.metadata.codecs],
        )
        upload(path, f"{dest}/{sub}")
        shutil.rmtree(path)
        print(sub, path.name, "done", flush=True)

    for var in variables:
        # a memory map on disk, so a variable never has to fit in memory
        if suite == "point":  # every point store holds the same 512-step range
            a = load_point(raw, var, cache=tmp)
        else:
            a = load_era5(raw, var, nt, cache=tmp)
        print(var, a.shape, flush=True)
        for layout in spec["skipzfp"]:
            path = tmp / f"{layout}.zarr"
            z = write_skipzfp(path, a, layout)
            if layout == "k8":
                tols[var] = max_error(z, a)
                print(var, "tolerance", tols[var], flush=True)
            done(path, f"stores_sub/{var}")
        for tag in spec["chunks"]:
            for fmt in ("zstd", "zfpacc"):
                path = tmp / f"{fmt}.zarr"
                write_chunked(path, a, fmt, shape_of(tag), tols[var])
                done(path, f"stores_chunks/{var}/{tag}")
        for tag in spec["inners"]:
            path = tmp / f"{tag}.zarr"
            write_sharded(path, a, shape_of(tag))
            done(path, f"stores_zshard/{var}")
        if spec.get("tight"):
            # SkipZFP's own errors are the target, so build k8 just to measure them
            path = tmp / "k8.zarr"
            z = write_skipzfp(path, a, "k8")
            tol, info = tight_tolerance(a, z, max_error(z, a))
            shutil.rmtree(path)
            tols[var] = tol
            Path(tmp, f"tight_{var}.json").write_text(json.dumps(info, indent=1))
            upload(Path(tmp, f"tight_{var}.json"), dest)
            for tag in spec["tight"]:
                path = tmp / "zfpacc.zarr"
                write_chunked(path, a, "zfpacc", shape_of(tag), tol)
                done(path, f"stores_chunks/{var}/{tag}-tight")
        del a
        Path(tmp, f"{var}.npy").unlink()
    Path(tmp, f"tolerances_{suite}.json").write_text(json.dumps(tols))
    upload(Path(tmp, f"tolerances_{suite}.json"), dest)
    Path(tmp, f"stores_{suite}.json").write_text(json.dumps(stores, indent=1))
    upload(Path(tmp, f"stores_{suite}.json"), dest)
    print("all stores ready", tols, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("dest")
    ap.add_argument("--suite", choices=SUITES, required=True)
    ap.add_argument("--tmp", default="prepare_tmp")
    ap.add_argument("--nt", type=int)
    ap.add_argument("--vars", default=",".join(VARS))
    args = ap.parse_args()
    prepare(args.raw, args.dest, args.suite, args.tmp, args.nt, args.vars.split(","))
