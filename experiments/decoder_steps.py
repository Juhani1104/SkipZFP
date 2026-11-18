"""Single-thread decode-and-count speed of each decoder step (the paper's decoder table).

usage: python decoder_steps.py <era5_2000.zarr> [out.json]

Takes the first 256 hours of t2m and ws from the ERA5 store that fetch_era5.py
writes, cuts them into 64 x 16 x 32 sub-chunks and encodes each at 8 bits/value,
which gives 8.4 million values per variable. It then counts the values above
280 K (t2m) or 5 m/s (ws) on one thread with each decoder step, from libzfp to
the full fast path, and reports the best of five runs of five passes. Every step
must return the same count. Run it on the machine you want to measure: steps the
CPU lacks (AVX-512, BMI2) fall back to slower code. With out.json, the run is
recorded in out.json.runs.json like the other experiments.
"""

import ctypes
import json
import sys
import time

import numpy as np
import zarr

from skipzfp import _native, codec

from common import start_run

HOURS = 256
SUB = (64, 16, 32)
RATE = 8.0
THRESHOLDS = (("t2m", 280.0), ("ws", 5.0))
# szfp_set_fast_decode modes, in the order the paper adds them
STEPS = (
    (0, "libzfp"),
    (3, "previous (AVX-512 transpose)"),
    (4, "+ AVX-512 inverse transform"),
    (5, "+ BMI2 bit-plane parsing"),
    (1, "+ counting in registers"),
)


def encode(x):
    """Encode x sub-chunk by sub-chunk, in time, y, x order, into one buffer."""
    n = codec.native()
    t, y, z = SUB
    units = (
        np.ascontiguousarray(x[a : a + t, b : b + y, c : c + z])
        for a in range(0, x.shape[0], t)
        for b in range(0, x.shape[1], y)
        for c in range(0, x.shape[2], z)
    )
    return np.concatenate(
        [np.frombuffer(n.encode(u, RATE, 4), np.uint8) for u in units]
    )


def bench(lib, buf, threshold):
    """Return (step, million values/s, count) for every decoder step."""
    block = int(64 * RATE / 8)
    offs = np.arange(0, buf.size, block, dtype=np.uint64)
    out = []
    for mode, label in STEPS:
        lib.szfp_set_fast_decode(mode)
        n = ctypes.c_size_t()
        args = (
            buf.ctypes.data,
            buf.size,
            offs.ctypes.data,
            offs.size,
            block,
            RATE,
            4,
            threshold,
            ctypes.byref(n),
        )
        lib.szfp_count_offsets(*args)
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(5):
                lib.szfp_count_offsets(*args)
            best = min(best, time.perf_counter() - t)
        out.append((label, offs.size * 64 * 5 / best / 1e6, n.value))
    lib.szfp_set_fast_decode(1)
    return out


def main(raw, out_path=None):
    lib = ctypes.CDLL(str(_native.load_library()._name))
    lib.szfp_set_fast_decode.argtypes = [ctypes.c_int]
    lib.szfp_count_offsets.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    g = zarr.open_group(raw, mode="r")
    result = {}
    if out_path:
        result["run_id"] = start_run(out_path, raw, script="decoder_steps")
    for name, threshold in THRESHOLDS:
        buf = encode(np.asarray(g[name][:HOURS], dtype=np.float32))
        rows = bench(lib, buf, threshold)
        base = rows[1][1]
        equal = len({c for _, _, c in rows}) == 1
        print(f"{name}: {buf.size} bytes, counts all equal {equal}")
        for label, v, _ in rows:
            print(f"  {label:<30} {v:6.0f} M values/s  {v / base:5.2f}x of previous")
        result[name] = {
            "counts_equal": equal,
            "mvalues_per_s": {label: round(v) for label, v, _ in rows},
        }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
