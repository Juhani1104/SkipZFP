"""Experiment 3: AVG / MIN / MAX over regions, bytes read vs guaranteed accuracy.

usage: python exp3_aggregate.py <raw.zarr> <out.json>

SkipZFP reads a 2 / 4 / 8 bpv prefix of each block and reports the actual error and a
deterministic bound (A: max eps, B: per-value eps, C: 4-bit per-block error). The
baseline samples zstd chunks (64x16x32) with the same number of bytes, gives a 95%
CLT interval, and counts how often that interval misses the truth over SEEDS runs.
MIN / MAX from metadata alone: exact cmin / cmax for sub-chunks fully inside the
region, block [lo, hi] elsewhere.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np

from common import SUB_CHUNK, VARS, codec, load_era5

RATES = [2.0, 4.0, 8.0]
SIZES = {"europe": (128, 256), "8deg": (32, 32), "1deg": (4, 4)}
N_REGIONS = 20
SEEDS = 200


def encode(a, rate):
    nat = codec.native()
    grid = [s // c for s, c in zip(a.shape, SUB_CHUNK)]
    rec = np.empty_like(a)
    eps = np.empty(grid)
    blo = np.empty((*grid, 16, 4, 8))
    bhi = np.empty((*grid, 16, 4, 8))
    cmm = np.empty((*grid, 2))
    bq = np.empty((*grid, 16, 4, 8))

    def one(idx):
        sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, SUB_CHUNK))
        c = np.ascontiguousarray(a[sl])
        r = nat.decode(nat.encode(c, rate, 4), SUB_CHUNK, rate, 4)
        rec[sl] = r
        m = nat.meta(c, rate, 4)
        cmin, cmax, e = np.frombuffer(m[:12].tobytes(), np.float32).astype(np.float64)
        off = m[12:].reshape(16, 4, 8, 2).astype(np.float64)
        eps[idx] = e
        cmm[idx] = cmin, cmax
        blo[idx] = cmin + off[..., 0] / 255 * (cmax - cmin)
        bhi[idx] = cmin + off[..., 1] / 255 * (cmax - cmin)
        be = (
            np.abs(r.astype(np.float64) - c)
            .reshape(16, 4, 4, 4, 8, 4)
            .max(axis=(1, 3, 5))
        )
        bq[idx] = np.ceil(15 * be / e) / 15 * e if e > 0 else 0.0

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, np.ndindex(*grid)))
    return rec, eps, blo, bhi, cmm, bq


def regions(rng):
    for size, (h, w) in SIZES.items():
        for _ in range(1 if size == "europe" else N_REGIONS):
            y = int(rng.integers(0, 128 - h + 1))
            x = int(rng.integers(0, 256 - w + 1))
            yield size, (y, x, h, w)


def meta_minmax(blo, bhi, cmm, y, x, h, w):
    by0, by1, bx0, bx1 = y // 4, (y + h - 1) // 4, x // 4, (x + w - 1) // 4
    lo = blo.transpose(0, 3, 1, 4, 2, 5).reshape(-1, 32, 64)
    hi = bhi.transpose(0, 3, 1, 4, 2, 5).reshape(-1, 32, 64)
    touch = (slice(None), slice(by0, by1 + 1), slice(bx0, bx1 + 1))
    iy0, iy1, ix0, ix1 = -(-y // 4), (y + h) // 4, -(-x // 4), (x + w) // 4
    inside = (slice(None), slice(iy0, iy1), slice(ix0, ix1))
    min_lo = float(lo[touch].min())
    max_hi = float(hi[touch].max())
    has_in = iy1 > iy0 and ix1 > ix0
    min_hi = float(hi[inside].min()) if has_in else float(hi[touch].max())
    max_lo = float(lo[inside].max()) if has_in else float(lo[touch].min())
    sy0, sy1, sx0, sx1 = -(-y // 16), (y + h) // 16, -(-x // 32), (x + w) // 32
    if sy1 > sy0 and sx1 > sx0:
        sub = cmm[:, sy0:sy1, sx0:sx1]
        min_hi = min(min_hi, float(sub[..., 0].min()))
        max_lo = max(max_lo, float(sub[..., 1].max()))
        if (y, x, h, w) == (0, 0, 128, 256):
            min_lo, max_hi = float(sub[..., 0].min()), float(sub[..., 1].max())
    return (min_lo, min_hi), (max_lo, max_hi)


def minmax_bound(rr, got, e):
    """Error bound of MIN / MAX from the guaranteed interval [rr - e, rr + e]."""
    return dict(
        MIN=max(float((rr + e).min()) - got["MIN"], got["MIN"] - float((rr - e).min())),
        MAX=max(float((rr + e).max()) - got["MAX"], got["MAX"] - float((rr - e).max())),
    )


def sample_avg(o, y, x, h, w, frac, rng):
    cy0, cy1, cx0, cx1 = (
        y // SUB_CHUNK[1],
        (y + h - 1) // SUB_CHUNK[1],
        x // SUB_CHUNK[2],
        (x + w - 1) // SUB_CHUNK[2],
    )
    cells = [
        (t, cy, cx)
        for t in range(o.shape[0] // SUB_CHUNK[0])
        for cy in range(cy0, cy1 + 1)
        for cx in range(cx0, cx1 + 1)
    ]
    k = max(2, int(round(frac * len(cells))))
    pick = rng.choice(len(cells), size=k, replace=False)
    sums, cnts = [], []
    for i in pick:
        t, cy, cx = cells[i]
        ys = slice(max(y, cy * SUB_CHUNK[1]), min(y + h, (cy + 1) * SUB_CHUNK[1]))
        xs = slice(max(x, cx * SUB_CHUNK[2]), min(x + w, (cx + 1) * SUB_CHUNK[2]))
        v = o[t * SUB_CHUNK[0] : (t + 1) * SUB_CHUNK[0], ys, xs]
        sums.append(float(v.sum()))
        cnts.append(v.size)
    sums, cnts = np.array(sums), np.array(cnts)
    est = sums.sum() / cnts.sum()
    n, N = k, len(cells)
    resid = sums - est * cnts
    var = (1 - n / N) * resid.var(ddof=1) / n / cnts.mean() ** 2
    return est, 1.96 * np.sqrt(var)


def zstd_bytes(a):
    """Size of the data as zstd chunks of SUB_CHUNK (what a sampler would read)."""
    blosc = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    grid = [s // c for s, c in zip(a.shape, SUB_CHUNK)]

    def one(idx):
        sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, SUB_CHUNK))
        return len(blosc.encode(np.ascontiguousarray(a[sl])))

    with ThreadPoolExecutor(32) as ex:
        return sum(ex.map(one, np.ndindex(*grid)))


def main(raw, out_path):
    res = []
    for var in VARS:
        a = load_era5(raw, var)
        zbytes = zstd_bytes(a)
        print(var, "zstd bytes", zbytes, flush=True)
        enc = {r: encode(a, r) for r in RATES}
        rng = np.random.default_rng(0)
        for size, (y, x, h, w) in regions(rng):
            o = a[:, y : y + h, x : x + w].astype(np.float64)
            exact = dict(AVG=float(o.mean()), MIN=float(o.min()), MAX=float(o.max()))
            row = dict(
                var=var,
                size=size,
                region=[y, x, h, w],
                exact=exact,
                skipzfp={},
                sampling={},
            )
            for r in RATES:
                rec, eps, blo, bhi, cmm, bq = enc[r]
                rr = rec[:, y : y + h, x : x + w].astype(np.float64)
                sy, sx = y // 16, x // 32
                ee = eps[:, sy : (y + h - 1) // 16 + 1, sx : (x + w - 1) // 32 + 1]
                ev = np.repeat(np.repeat(np.repeat(ee, 64, 0), 16, 1), 32, 2)[
                    :, y - sy * 16 : y - sy * 16 + h, x - sx * 32 : x - sx * 32 + w
                ]
                bg = bq.transpose(0, 3, 1, 4, 2, 5).reshape(-1, 32, 64)
                by, bx = y // 4, x // 4
                cv = np.repeat(
                    np.repeat(
                        np.repeat(
                            bg[:, by : (y + h - 1) // 4 + 1, bx : (x + w - 1) // 4 + 1],
                            4,
                            0,
                        ),
                        4,
                        1,
                    ),
                    4,
                    2,
                )
                cv = cv[:, y - by * 4 : y - by * 4 + h, x - bx * 4 : x - bx * 4 + w]
                got = dict(
                    AVG=float(rr.mean()), MIN=float(rr.min()), MAX=float(rr.max())
                )
                err = {k: abs(got[k] - exact[k]) for k in got}

                bounds = dict(
                    A=dict(
                        AVG=float(ee.max()), MIN=float(ee.max()), MAX=float(ee.max())
                    ),
                    B=dict(AVG=float(ev.mean()), **minmax_bound(rr, got, ev)),
                    C=dict(AVG=float(cv.mean()), **minmax_bound(rr, got, cv)),
                )
                for name, bd in bounds.items():
                    assert all(err[k] <= bd[k] + 1e-9 for k in err), (
                        var,
                        size,
                        r,
                        name,
                        err,
                        bd,
                    )
                row["skipzfp"][str(int(r))] = dict(
                    err=err, bound=bounds["A"], bounds=bounds, bytes_frac=r / 32
                )
                frac = (a.size * r / 8) / zbytes
                srng = np.random.default_rng(1)
                ests = [sample_avg(a, y, x, h, w, frac, srng) for _ in range(SEEDS)]
                errs = np.array([abs(e - exact["AVG"]) for e, _ in ests])
                miss = float(np.mean([abs(e - exact["AVG"]) > hw for e, hw in ests]))
                row["sampling"][str(int(r))] = dict(
                    frac=frac,
                    err_median=float(np.median(errs)),
                    err_max=float(errs.max()),
                    ci_miss=miss,
                )
            (mn_lo, mn_hi), (mx_lo, mx_hi) = meta_minmax(*enc[8.0][2:5], y, x, h, w)
            assert mn_lo <= exact["MIN"] <= mn_hi and mx_lo <= exact["MAX"] <= mx_hi, (
                var,
                size,
                (mn_lo, mn_hi),
                exact,
            )
            row["meta_only"] = dict(MIN=[mn_lo, mn_hi], MAX=[mx_lo, mx_hi])
            res.append(row)
        print(var, "done", flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1))

    for var in ("t2m", "ws"):
        print(f"\n== {var}")
        for size in SIZES:
            rows = [r for r in res if r["var"] == var and r["size"] == size]
            print(f"  {size} ({len(rows)} regions)")
            for r in RATES:
                k = str(int(r))
                ae = max(x["skipzfp"][k]["err"]["AVG"] for x in rows)
                me = max(x["skipzfp"][k]["err"]["MAX"] for x in rows)
                ab = {
                    n: np.median([x["skipzfp"][k]["bounds"][n]["AVG"] for x in rows])
                    for n in "ABC"
                }
                mb = {
                    n: np.median([x["skipzfp"][k]["bounds"][n]["MAX"] for x in rows])
                    for n in "ABC"
                }
                se = max(x["sampling"][k]["err_max"] for x in rows)
                sm = np.mean([x["sampling"][k]["ci_miss"] for x in rows])
                print(
                    f"    {int(r)} bpv (reads {r / 32:.1%}): AVG max err {ae:.2e}, bound A/B/C {ab['A']:.2e}/{ab['B']:.2e}/{ab['C']:.2e}"
                    f" | MAX max err {me:.2e}, bound A/B/C {mb['A']:.2e}/{mb['B']:.2e}/{mb['C']:.2e}"
                    f" | sampling AVG max err {se:.2e}, CI miss {sm:.1%}"
                )
            wmin = [x["meta_only"]["MIN"][1] - x["meta_only"]["MIN"][0] for x in rows]
            wmax = [x["meta_only"]["MAX"][1] - x["meta_only"]["MAX"][0] for x in rows]
            print(
                f"    metadata only: MIN width median {np.median(wmin):.3f} max {max(wmin):.3f}; MAX width median {np.median(wmax):.3f} max {max(wmax):.3f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    args = ap.parse_args()
    main(args.raw, args.out)
