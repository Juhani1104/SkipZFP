"""Experiment 4: filtered aggregates COUNT / AVG(x > T) over Europe and a full year.

usage: python exp4_filtered_aggregate.py <raw.zarr> <out.json> [--stores DIR]

Stage 1 (metadata only): a guaranteed COUNT interval from SkipZFP block metadata,
compared with zone maps (exact min / max) of several chunk sizes.
Stage 2, offline (decoded, decoded_C): SkipZFP 8 bpv actual error and
deterministic intervals (value error plus set-membership error), computed from the
whole reconstruction in memory, with per-sub-chunk eps (B) or a 4-bit per-block
error (C) that the format does not store. These rows have no I/O cost.
Stage 3, from a store (store): the same year written as a SkipZFP store and
queried with aggregate_gt, which reads the IN and MAYBE blocks and skips the OUT
blocks. Its bytes and requests are what the query actually read, counted again by
a wrapper store, next to what a COUNT of the same predicate reads. The original
data are used only to check the answers.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from prepare_stores import write_skipzfp

from skipzfp import QueryOptions, aggregate_gt, query_gt

from common import (
    CONCURRENCY,
    SKIPZFP_GAP_BYTES,
    SUB_CHUNK,
    VARS,
    codec,
    load_era5,
    open_counted,
    start_run,
)

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
    if nc + na == 0:  # no original value can match: there is no mean to bound
        avg_hi = avg_lo = None
    elif nc:
        cs = s_c_hi + np.cumsum(up)
        k = nc + np.arange(1, len(up) + 1)
        avg_hi = max(s_c_hi / nc, float((cs / k).max()) if len(up) else -np.inf)
        avg_lo = (s_c_lo + na * th) / (nc + na)
    else:
        avg_hi, avg_lo = float(up[0]) if len(up) else th, th
    # an empty selection has no mean, so its AVG fields are None
    t_avg = t_sum / t_cnt if t_cnt else None
    a_avg = a_sum / a_cnt if a_cnt else None
    assert nc <= t_cnt <= nc + na, "count interval misses truth"
    assert t_avg is None or avg_lo <= t_avg <= avg_hi, "avg interval misses truth"
    return dict(
        count_true=t_cnt,
        count_approx=a_cnt,
        count_rel_err=abs(a_cnt - t_cnt) / t_cnt if t_cnt else None,
        count_interval=[nc, nc + na],
        avg_true=t_avg,
        avg_approx=a_avg,
        avg_err=abs(a_avg - t_avg) if t_cnt and a_cnt else None,
        avg_interval=None if avg_lo is None else [avg_lo, avg_hi],
    )


def fmt(x, spec):
    """Format x, or show - for a value an empty selection does not have."""
    return "-" if x is None else format(x, spec)


def width(iv):
    return None if iv is None else iv[1] - iv[0]


def from_store(path, a, rec, th, blocks_file):
    """Run aggregate_gt on the store, check it, and return what it read.

    The (chunk, block) ids of the blocks it read are saved to blocks_file.
    """
    opts = QueryOptions(
        request_concurrency=CONCURRENCY, merge_gap_blocks=SKIPZFP_GAP_BYTES // 64
    )
    arr, counted = open_counted(str(path))
    counted.requests = counted.bytes = 0
    r = aggregate_gt(arr, th, options=opts, return_blocks=True)
    seen = (counted.requests, counted.bytes)
    c = query_gt(arr, th, options=opts)

    m = rec > th
    o = a > th
    t_cnt = int(o.sum())
    t_avg = float(a[o].astype(np.float64).mean()) if t_cnt else None
    assert r.count == int(m.sum()) and r.count == c.count
    assert np.isclose(r.sum, rec[m].astype(np.float64).sum(), rtol=1e-12)
    assert r.original_count[0] <= t_cnt <= r.original_count[1], "count misses truth"
    if t_avg is None:  # an empty selection has no mean; the bounds may still allow one
        assert r.original_count[0] == 0
    else:
        assert r.original_mean[0] <= t_avg <= r.original_mean[1], "avg misses truth"
    assert seen == (r.range_requests, r.bytes_read), "read accounting is off"
    assert len(r.read_blocks) == r.in_blocks + r.maybe_blocks
    np.save(blocks_file, r.read_blocks)
    return dict(
        count=r.count,
        avg=r.mean,
        original_count_interval=list(r.original_count),
        original_avg_interval=None
        if r.original_mean is None
        else list(r.original_mean),
        in_blocks=r.in_blocks,
        out_blocks=r.out_blocks,
        maybe_blocks=r.maybe_blocks,
        total_blocks=r.total_blocks,
        metadata_bytes=r.metadata_bytes_read,
        payload_bytes=r.payload_bytes_read,
        useful_payload_bytes=r.useful_payload_bytes,
        overread_bytes=r.payload_overread_bytes,
        metadata_requests=r.metadata_requests,
        payload_requests=r.payload_requests,
        count_query_payload_bytes=c.payload_bytes_read,
        count_query_payload_requests=c.payload_requests,
        full_payload_bytes=r.total_blocks * int(8 * RATE),
        read_blocks_file=Path(blocks_file).name,
    )


def main(raw, out_path, stores):
    run_id = start_run(out_path, raw, script="exp4", stores=str(stores))
    res = []
    for var in VARS:
        a = load_era5(raw, var)
        rec, eps, blo, bhi, bq = encode_all(a)
        path = Path(stores, f"{var}.zarr")
        z = write_skipzfp(path, a, "k8")
        assert np.array_equal(z[:], rec), "store decodes differently from encode_all"
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
                    width=n_maybe / truth if truth else None,
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
                    width=(hi_c - lo_c) / truth if truth else None,
                    meta_bytes=zlo.size * 8,
                    interval=[lo_c, hi_c],
                )
            row["decoded"] = stage2(a, rec, partial(efield_sub, eps), th)
            row["decoded_C"] = stage2(a, rec, partial(efield_block, bq), th)
            blocks_file = Path(stores, f"{var}_{sel:g}_read_blocks.npy")
            row["store"] = from_store(path, a, rec, th, blocks_file)
            res.append(row)
            d, dc, s = row["decoded"], row["decoded_C"], row["store"]
            print(
                f"{var} {sel * 100:>6g}%: metadata width "
                f"{fmt(row['meta_only']['skipzfp']['width'], '.2f')}x truth | decoded "
                f"COUNT err {fmt(d['count_rel_err'], '.2e')}, AVG err {fmt(d['avg_err'], '.2e')}"
                f" | AVG width B {fmt(width(d['avg_interval']), '.3f')} -> C "
                f"{fmt(width(dc['avg_interval']), '.3f')}",
                flush=True,
            )
            print(
                f"{'':>11}store: AVG reads {s['payload_bytes'] / 1e6:.1f} MB in "
                f"{s['payload_requests']} requests (COUNT "
                f"{s['count_query_payload_bytes'] / 1e6:.1f} MB, full scan "
                f"{s['full_payload_bytes'] / 1e6:.1f} MB) | original AVG width "
                f"{fmt(width(s['original_avg_interval']), '.4f')}",
                flush=True,
            )
    for row in res:
        row["run_id"] = run_id
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument(
        "--stores",
        help="where to write the SkipZFP stores (default: exp4_stores next to out)",
    )
    args = ap.parse_args()
    main(args.raw, args.out, args.stores or Path(args.out).parent / "exp4_stores")
