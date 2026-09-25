"""Experiment 5: metadata overhead, metadata bytes / ZFP payload bytes.

usage: python exp5_overhead.py <out.json>

The 3D numbers are measured from a written array (payload = chunk object size,
metadata = one row of the metadata array) and checked against the formula; the 2D
column is the formula for a chunk with the same number of values.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr

from common import codec

RATES = [2, 4, 8, 12, 16]
CHUNK_3D = (64, 16, 32)
CHUNK_2D = (128, 256)
HEADER = 12  # cmin, cmax, eps
PER_BLOCK = 2  # uint8 lower / upper offset


def formula(chunk, rate):
    nblk = int(np.prod([c // 4 for c in chunk]))
    payload = nblk * (4 ** len(chunk)) * rate / 8
    meta = HEADER + PER_BLOCK * nblk
    return nblk, payload, meta, meta / payload


def measured_3d(rate, tmp):
    shape = tuple(2 * c for c in CHUNK_3D)
    a = np.random.default_rng(0).standard_normal(shape).astype(np.float32) + 280
    g = zarr.open_group(str(Path(tmp) / f"r{rate}.zarr"), mode="w")
    z = g.create_array(
        "data",
        shape=shape,
        chunks=CHUNK_3D,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=float(rate)),
        compressors=None,
        filters=None,
    )
    z[:] = a
    meta = codec.write_meta(z, a)
    files = [p for p in (Path(tmp) / f"r{rate}.zarr" / "data" / "c").rglob("*")]
    sizes = {p.stat().st_size for p in files if p.is_file()}
    assert len(sizes) == 1, sizes
    return sizes.pop(), meta.shape[-1]


def main(out):
    rows = []
    tmp = tempfile.mkdtemp()
    try:
        for rate in RATES:
            nblk, pay, meta, ov = formula(CHUNK_3D, rate)
            assert measured_3d(rate, tmp) == (pay, meta), rate
            nblk2, pay2, meta2, ov2 = formula(CHUNK_2D, rate)
            rows.append(
                dict(
                    rate=rate,
                    blocks_3d=nblk,
                    payload_3d=int(pay),
                    meta_3d=meta,
                    overhead_3d=ov,
                    blocks_2d=nblk2,
                    payload_2d=int(pay2),
                    meta_2d=meta2,
                    overhead_2d=ov2,
                )
            )
    finally:
        shutil.rmtree(tmp)
    for r in rows:
        print(
            f"{r['rate']:>3} bpv | 3D {r['meta_3d']} / {r['payload_3d']} = "
            f"{100 * r['overhead_3d']:.2f}% | 2D {100 * r['overhead_2d']:.2f}%"
        )
    Path(out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    main(ap.parse_args().out)
