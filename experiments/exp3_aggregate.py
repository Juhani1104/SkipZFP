"""Experiment 3: AVG / MIN / MAX over regions, bytes read vs guaranteed accuracy.

usage: python exp3_aggregate.py <raw.zarr> <out.json>

SkipZFP reads a 2 / 4 / 8 bpv prefix of each block and reports the actual error and a
deterministic bound (A: max eps, B: per-value eps, C: 4-bit per-block error). Its
bytes are counted exactly: every block that touches the region at the prefix rate,
plus the zone map record (layers 2, 4, 8) of every sub-chunk it touches. Merging
nearby reads would add over-read on top; this offline study fixes no merge policy.
The baseline samples whole zstd chunks (64x16x32), in random order, until the
actual compressed bytes of the chunks it took reach SkipZFP's bytes, so both read
the same bytes, over-read of chunks that stick out of the region included. It gives
a 95% CLT interval, and each region records how often that interval misses the
truth over SEEDS runs, with a Wilson 95% interval for that rate; the summary
reports the rate pooled over all regions and its spread across regions. A CLT
interval is a probabilistic statement, unlike SkipZFP's deterministic bounds.
MIN / MAX from metadata alone: exact cmin / cmax for sub-chunks fully inside the
region, block [lo, hi] elsewhere.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np

from common import SUB_CHUNK, VARS, codec, load_era5, start_run

RATES = [2.0, 4.0, 8.0]
# a zone map record of a 64x16x32 sub-chunk stored with layers (2, 4, 8)
META_RECORD = codec.meta_size(codec.SkipZFPCodec(rate=8.0, layers=RATES), SUB_CHUNK)
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


def sample_avg(o, y, x, h, w, budget, zsizes, rng):
    """Estimate the region's AVG from whole chunks taken until budget bytes are read.

    Chunks come in random order and are taken while the compressed bytes read so
    far are below budget, at least two of them. Returns the estimate, the 95% CLT
    half-width, the bytes actually read and the number of chunks.
    """
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
    order = rng.permutation(len(cells))
    sizes = np.array([zsizes[cells[i]] for i in order])
    k = min(len(cells), max(2, int(np.searchsorted(np.cumsum(sizes), budget)) + 1))
    pick = order[:k]
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
    return est, 1.96 * np.sqrt(var), int(sizes[:k].sum()), k


def skipzfp_bytes(y, x, h, w, nt, rate):
    """Bytes SkipZFP must read for the region: touched blocks and zone map records."""
    by = (y + h - 1) // 4 - y // 4 + 1
    bx = (x + w - 1) // 4 - x // 4 + 1
    payload = (nt // 4) * by * bx * int(64 * rate / 8)
    sy = (y + h - 1) // SUB_CHUNK[1] - y // SUB_CHUNK[1] + 1
    sx = (x + w - 1) // SUB_CHUNK[2] - x // SUB_CHUNK[2] + 1
    meta = (nt // SUB_CHUNK[0]) * sy * sx * META_RECORD
    return payload + meta


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a rate of k out of n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - hw), min(1.0, c + hw)]


def zstd_bytes(a):
    """Compressed size of every zstd chunk of SUB_CHUNK (what a sampler would read)."""
    blosc = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    grid = [s // c for s, c in zip(a.shape, SUB_CHUNK)]

    def one(idx):
        sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, SUB_CHUNK))
        return len(blosc.encode(np.ascontiguousarray(a[sl])))

    idx = list(np.ndindex(*grid))
    with ThreadPoolExecutor(32) as ex:
        return np.array(list(ex.map(one, idx))).reshape(grid)


def main(raw, out_path):
    run_id = start_run(out_path, raw, script="exp3")
    res = []
    for var in VARS:
        a = load_era5(raw, var)
        zsizes = zstd_bytes(a)
        print(var, "zstd bytes", int(zsizes.sum()), flush=True)
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
                cell = bg[:, by : (y + h - 1) // 4 + 1, bx : (x + w - 1) // 4 + 1]
                cv = cell.repeat(4, 0).repeat(4, 1).repeat(4, 2)
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
                sk_bytes = skipzfp_bytes(y, x, h, w, a.shape[0], r)
                row["skipzfp"][str(int(r))] = dict(
                    err=err,
                    bound=bounds["A"],
                    bounds=bounds,
                    bytes=sk_bytes,
                    bytes_frac=r / 32,
                )
                srng = np.random.default_rng(1)
                ests = [
                    sample_avg(a, y, x, h, w, sk_bytes, zsizes, srng)
                    for _ in range(SEEDS)
                ]
                errs = np.array([abs(e[0] - exact["AVG"]) for e in ests])
                misses = int(sum(abs(e[0] - exact["AVG"]) > e[1] for e in ests))
                row["sampling"][str(int(r))] = dict(
                    budget_bytes=sk_bytes,
                    bytes_median=float(np.median([e[2] for e in ests])),
                    bytes_max=int(max(e[2] for e in ests)),
                    chunks_median=float(np.median([e[3] for e in ests])),
                    err_median=float(np.median(errs)),
                    err_max=float(errs.max()),
                    ci_miss=misses / SEEDS,
                    ci_misses=misses,
                    trials=SEEDS,
                    ci_miss_wilson95=wilson(misses, SEEDS),
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
    for row in res:
        row["run_id"] = run_id
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
                miss = [x["sampling"][k]["ci_miss"] for x in rows]
                pooled = sum(x["sampling"][k]["ci_misses"] for x in rows) / sum(
                    x["sampling"][k]["trials"] for x in rows
                )
                print(
                    f"    {int(r)} bpv (reads {r / 32:.1%}): AVG max err {ae:.2e}, bound A/B/C {ab['A']:.2e}/{ab['B']:.2e}/{ab['C']:.2e}"
                    f" | MAX max err {me:.2e}, bound A/B/C {mb['A']:.2e}/{mb['B']:.2e}/{mb['C']:.2e}"
                    f" | sampling AVG max err {se:.2e}, CI miss pooled {pooled:.1%}"
                    f" (regions: median {np.median(miss):.1%}, max {max(miss):.1%})"
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
