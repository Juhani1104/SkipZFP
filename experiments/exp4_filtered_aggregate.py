"""Experiment 4: filtered aggregates COUNT / AVG(x > T) over Europe and a full year.

usage: python exp4_filtered_aggregate.py <raw.zarr> <out.json>

Stage 1 (metadata only): a guaranteed COUNT interval from SkipZFP block metadata,
compared with zone maps (exact min / max) of several chunk sizes.
Stage 2 (decode MAYBE blocks): SkipZFP 8 bpv actual error and deterministic intervals
(value error plus set-membership error), with per-sub-chunk eps (B) or 4-bit
per-block error (C). Costs are the metadata and MAYBE-block payload bytes.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from common import SUB_CHUNK, VARS, codec, load_era5

RATE = 8.0
SELS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
ZONES = [(16, 4, 4), (16, 16, 16), (64, 16, 32), (64, 64, 128), (64, 128, 256)]


def encode_all(a):
    nat = codec.native()
    grid = [s // c for s, c in zip(a.shape, SUB_CHUNK)]
    rec = np.empty_like(a)
    eps = np.empty(grid, np.float64)
    lo = np.empty((*grid, 512))
    hi = np.empty((*grid, 512))
    bq = np.empty((*grid, 16, 4, 8))

    def one(idx):
        sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, SUB_CHUNK))
        c = np.ascontiguousarray(a[sl])
        r = nat.decode(nat.encode(c, RATE, 4), SUB_CHUNK, RATE, 4)
        rec[sl] = r
        m = nat.meta(c, RATE, 4)
        cmin, cmax, e = np.frombuffer(m[:12].tobytes(), np.float32).astype(np.float64)
        off = m[12:].reshape(-1, 2).astype(np.float64)
        span = cmax - cmin
        eps[idx] = e
        lo[idx] = cmin + off[:, 0] / 255 * span - e
        hi[idx] = cmin + off[:, 1] / 255 * span + e
        be = (
            np.abs(r.astype(np.float64) - c)
            .reshape(16, 4, 4, 4, 8, 4)
            .max(axis=(1, 3, 5))
        )
        bq[idx] = np.ceil(15 * be / e) / 15 * e if e > 0 else 0.0

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, np.ndindex(*grid)))
    return rec, eps, lo.reshape(-1), hi.reshape(-1), bq


def zone(a, shape):
    g = [s // c for s, c in zip(a.shape, shape)]
    b = a.reshape(g[0], shape[0], g[1], shape[1], g[2], shape[2])
    return b.min(axis=(1, 3, 5)).ravel().astype(np.float64), b.max(
        axis=(1, 3, 5)
    ).ravel().astype(np.float64)


def efield_sub(eps, t):
    return np.repeat(
        np.repeat(np.repeat(eps[t : t + 1], SUB_CHUNK[0], 0), SUB_CHUNK[1], 1),
        SUB_CHUNK[2],
        2,
    )


def efield_block(bq, t):
    g = bq[t].transpose(2, 0, 3, 1, 4)  # (16, 8, 4, 8, 8) = (bt, sy, by, sx, bx)
    g = g.reshape(16, 32, 64)
    return np.repeat(np.repeat(np.repeat(g, 4, 0), 4, 1), 4, 2)


def stage2(a, rec, efield, th):
    nc = na = t_cnt = a_cnt = 0
    s_c_lo = s_c_hi = t_sum = a_sum = 0.0
    ups = []
    for t in range(a.shape[0] // SUB_CHUNK[0]):
        sl = slice(t * SUB_CHUNK[0], (t + 1) * SUB_CHUNK[0])
        o = a[sl].astype(np.float64)
        r = rec[sl].astype(np.float64)
        e = efield(t)
        cert = r > th + e
        amb = (r > th - e) & ~cert
        nc += int(cert.sum())
        na += int(amb.sum())
        s_c_lo += float((r[cert] - e[cert]).sum())
        s_c_hi += float((r[cert] + e[cert]).sum())
        ups.append(r[amb] + e[amb])
        tm = o > th
        t_cnt += int(tm.sum())
        t_sum += float(o[tm].sum())
        am = r > th
        a_cnt += int(am.sum())
        a_sum += float(r[am].sum())
    up = np.sort(np.concatenate(ups))[::-1]
    if nc:
        cs = s_c_hi + np.cumsum(up)
        k = nc + np.arange(1, len(up) + 1)
        avg_hi = max(s_c_hi / nc, float((cs / k).max()) if len(up) else -np.inf)
        avg_lo = (s_c_lo + na * th) / (nc + na)
    else:
        avg_hi, avg_lo = float(up[0]) if len(up) else th, th
    t_avg, a_avg = t_sum / t_cnt, a_sum / a_cnt
    assert nc <= t_cnt <= nc + na and avg_lo <= t_avg <= avg_hi, "interval misses truth"
    return dict(
        count_true=t_cnt,
        count_approx=a_cnt,
        count_rel_err=abs(a_cnt - t_cnt) / t_cnt,
        count_interval=[nc, nc + na],
        avg_true=t_avg,
        avg_approx=a_avg,
        avg_err=abs(a_avg - t_avg),
        avg_interval=[avg_lo, avg_hi],
    )


def main(raw, out_path):
    res = []
    for var in VARS:
        a = load_era5(raw, var)
        rec, eps, blo, bhi, bq = encode_all(a)
        zones = {"x".join(map(str, s)): (*zone(a, s), int(np.prod(s))) for s in ZONES}
        meta_skip = blo.size * 2 + eps.size * 12
        for sel in SELS:
            th = float(np.quantile(a, 1 - sel))
            truth = int((a.astype(np.float64) > th).sum())
            row = dict(var=var, sel=sel, threshold=th, truth=truth)
            is_in = blo > th
            maybe = ~is_in & (bhi > th)
            n_in, n_maybe = int(is_in.sum()) * 64, int(maybe.sum()) * 64
            assert n_in <= truth <= n_in + n_maybe
            row["meta_only"] = {
                "skipzfp": dict(
                    width=n_maybe / truth,
                    meta_bytes=meta_skip,
                    interval=[n_in, n_in + n_maybe],
                )
            }
            for name, (zlo, zhi, n) in zones.items():
                zi = zlo > th
                zm = ~zi & (zhi > th)
                lo_c, hi_c = int(zi.sum()) * n, int(zi.sum() + zm.sum()) * n
                assert lo_c <= truth <= hi_c
                row["meta_only"]["zone_" + name] = dict(
                    width=(hi_c - lo_c) / truth,
                    meta_bytes=zlo.size * 8,
                    interval=[lo_c, hi_c],
                )
            row["decoded"] = stage2(a, rec, partial(efield_sub, eps), th)
            row["decoded_C"] = stage2(a, rec, partial(efield_block, bq), th)
            row["decoded"]["payload_bytes"] = int(maybe.sum()) * int(8 * RATE)
            res.append(row)
            d, dc = row["decoded"], row["decoded_C"]

            def w(iv):
                return iv[1] - iv[0]

            print(
                f"{var} {sel * 100:>6g}%: metadata width {row['meta_only']['skipzfp']['width']:.2f}x truth | decoded COUNT err "
                f"{d['count_rel_err']:.2e}, AVG err {d['avg_err']:.2e} | AVG width B {w(d['avg_interval']):.3f} -> C {w(dc['avg_interval']):.3f}"
                f" | COUNT width B {w(d['count_interval']) / d['count_true']:.2e} -> C {w(dc['count_interval']) / d['count_true']:.2e} (/truth)",
                flush=True,
            )
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    args = ap.parse_args()
    main(args.raw, args.out)
