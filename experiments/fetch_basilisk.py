"""Download the SDRBench Basilisk 2D decaying turbulence, a dataset beside ERA5.

usage: python fetch_basilisk.py <out.zarr> [steps]

The x-velocity u of the 1024 x 1024 run at Re = 10,000, 1501 float32 frames, the first
member of SDRBench's Basilisk-Turbulence archive. Frame 0 is an all-zero placeholder
and is skipped; the default keeps the next 1472 frames, a multiple of 64. Unlike the
ERA5 fields, the velocity is centered on zero and has no daily cycle, so it tests
pruning on a different distribution. The archive is read as a stream and only up to
the frames needed, so an interrupted download starts over.
"""

import sys
import tarfile
import time
import urllib.request

import numpy as np
import zarr

SRC = (
    "https://g-d0cd3f.fd635.8443.data.globus.org/raw-data/Basilisk-Turbulence/"
    "SDRBENCH-Basilisk-Turbulence.tar.gz"
)
MEMBER = "SDRBENCH-Basilisk-Turbulence/u_1024x1024x1501Re10k.f32"
N = 1024
FRAMES = 1501
STEP = 64  # frames written at once


def main(out, nt):
    nt = min(nt, FRAMES - 1)
    frame = N * N * 4
    root = zarr.open_group(out, mode="w")
    u = root.create_array(
        "u",
        shape=(nt, N, N),
        chunks=(STEP, N, N),
        dtype="float32",
        compressors=None,
        fill_value=np.nan,
    )
    root.create_array("time", data=np.arange(1, nt + 1, dtype=np.float64))
    root.attrs.update(
        source=SRC,
        member=MEMBER,
        steps=nt,
        time_unit="simulation frames, frame 0 skipped",
    )
    t0 = time.time()
    with (
        urllib.request.urlopen(SRC) as resp,
        tarfile.open(fileobj=resp, mode="r|gz") as tar,
    ):
        for m in tar:
            if m.name != MEMBER:
                continue
            f = tar.extractfile(m)
            f.read(frame)  # the all-zero frame 0
            for s in range(0, nt, STEP):
                k = min(STEP, nt - s)
                x = np.frombuffer(f.read(k * frame), dtype="<f4").reshape(k, N, N)
                assert np.isfinite(x).all()
                u[s : s + k] = x
                print(
                    f"{s + k}/{nt} frames, {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
            break
        else:
            raise SystemExit(f"{MEMBER} not found in the archive")
    print(f"finished: {nt} frames", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1472)
