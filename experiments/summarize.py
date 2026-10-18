"""Turn the raw JSONL results into the tables the paper uses, with nothing typed in.

usage: python summarize.py <results dir> [--out summary.md]

Reads whichever result files are in <results dir> and writes Markdown tables:

- exp1: bytes, requests, latency and throughput of single-point queries, and the
  time of batches of points.
- exp2: per variable, method and selectivity, the wall time (p25, median, p75,
  p95), bytes, requests and fraction of blocks read; each SkipZFP layout's speedup
  over the fastest baseline, with a 95% bootstrap interval that picks the fastest
  baseline again in every resample, and over every baseline; the median time of
  each step of a query, critical-path and summed kept apart; and cold open.
- ablation: per variable, config and selectivity, the wall time, bytes and
  requests, and the ratio of each config's median time to its reference's, with a
  95% bootstrap interval (resampling both groups, seed 0, 2000 resamples); p50,
  p95 and p99 for the tail configs; how much 4-bit error bounds narrow the AVG and
  COUNT intervals; and, from the traced configs, when the zone map and the reads
  arrive with the pipeline on and off.
- memory: per method and limit, how many jobs were killed for memory, the median
  wall time of the rest, and the largest VmHWM, cgroup peak and page cache.
- build cost: seconds, peak memory and bytes of the payload and the zone map, the
  zone map's share of the payload, and how many queries at each selectivity repay
  the zone map: its build seconds over the time a query saves against a full scan
  with the same decoder (ablation's no_pruning minus default).
- object size: SkipZFP with storage objects longer than 64 steps.
- exp3: regional AVG error and bound, and sampling at the same bytes, with the CI
  miss rate over all trials (Wilson interval) and per region.
- exp4: the metadata-only COUNT interval against zone maps, and COUNT and AVG read
  from the store.
- exp5: zone map bytes over payload bytes for 3D and 2D blocks.
- runs: the commit, host and calibration timings of every run.

Every table lists the run_ids its rows came from; <file>.runs.json has their code,
machine, packages and data.
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

SEED = 0
RESAMPLES = 2000


def load(path):
    try:
        return [json.loads(line) for line in open(path)]
    except FileNotFoundError:
        return []


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def ratio_ci(a, b, rng):
    """95% bootstrap interval of median(a) / median(b), resampling both."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rs = sorted(
        median(rng.choices(a, k=len(a))) / median(rng.choices(b, k=len(b)))
        for _ in range(RESAMPLES)
    )
    return rs[int(0.025 * RESAMPLES)], rs[int(0.975 * RESAMPLES) - 1]


def best_ratio_ci(baselines, ours, rng):
    """Median of the fastest baseline over ours, with a 95% bootstrap interval.

    Each resample redraws every baseline's times and picks the fastest again, so
    the interval also covers the chance that another baseline is the fastest.
    """
    point = min(median(b) for b in baselines) / median(ours)
    if len(ours) < 2 or any(len(b) < 2 for b in baselines):
        return point, float("nan"), float("nan")
    rs = sorted(
        min(median(rng.choices(b, k=len(b))) for b in baselines)
        / median(rng.choices(ours, k=len(ours)))
        for _ in range(RESAMPLES)
    )
    return point, rs[int(0.025 * RESAMPLES)], rs[int(0.975 * RESAMPLES) - 1]


def p_faster(a, b):
    """Chance that a random run of a is faster than a random run of b (ties count half)."""
    if not a or not b:
        return float("nan")
    wins = sum((x < y) + 0.5 * (x == y) for x in a for y in b)
    return wins / (len(a) * len(b))


def spread(xs):
    """p25, median, p75 and p95 of xs, formatted."""
    return [f"{quantile(xs, q):.3f}" for q in (0.25, 0.5, 0.75, 0.95)]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def runs_of(rows):
    return "runs: " + ", ".join(sorted({r.get("run_id", "?") for r in rows}))


def read_fraction(r):
    total = r.get("in_blocks", 0) + r.get("out_blocks", 0) + r.get("maybe_blocks", 0)
    return r["maybe_blocks"] / total if total and "in_blocks" in r else float("nan")


def is_baseline(method):
    return method.endswith(("_zm", "_izm"))


def exp2(rows):
    """Every method's times, then SkipZFP against the fastest and each baseline."""
    rng = random.Random(SEED)
    g = defaultdict(list)
    for r in rows:
        g[(r["var"], r["sel"], r["method"], r["size"])].append(r)
    secs = {k: [r["seconds"] for r in v] for k, v in g.items()}
    lines = []
    for (var, sel, method, size), v in sorted(g.items()):
        lines.append(
            [var, sel, method, size, len(v), *spread(secs[(var, sel, method, size)])]
            + [
                f"{median([r['bytes'] for r in v]) / 1e6:.1f}",
                f"{median([r['requests'] for r in v]):.0f}",
                f"{median([read_fraction(r) for r in v]):.4f}",
            ]
        )
    head = ["var", "sel", "method", "size", "n", "p25 s", "median s", "p75 s"]
    head += ["p95 s", "MB", "requests", "MAYBE frac"]
    out = "## exp2\n\n" + runs_of(rows) + "\n\n" + table(head, lines)

    best, each = [], []
    for (var, sel, method, size), ours in sorted(secs.items()):
        if method != "skipzfp":
            continue
        base = {
            (m, s): t
            for (vv, ss, m, s), t in secs.items()
            if (vv, ss) == (var, sel) and is_baseline(m)
        }
        if not base:
            continue
        point, lo, hi = best_ratio_ci(list(base.values()), ours, rng)
        fastest = min(base, key=lambda k: median(base[k]))
        best.append(
            [var, sel, size, f"{point:.2f}x", f"[{lo:.2f}, {hi:.2f}]"]
            + [" ".join(fastest), f"{p_faster(ours, base[fastest]):.2f}"]
        )
        for (m, s), t in sorted(base.items()):
            a, b = ratio_ci(t, ours, rng)
            each.append(
                [var, sel, size, f"{m} {s}", f"{median(t) / median(ours):.2f}x"]
                + [f"[{a:.2f}, {b:.2f}]", f"{p_faster(ours, t):.2f}"]
            )
    out += "\n\n### SkipZFP over the fastest baseline\n\n"
    out += (
        "The interval picks the fastest baseline again in every resample, so it "
        "also covers which baseline is fastest. P(faster) is the chance that a "
        "random SkipZFP run beats a random run of the baseline fastest by median."
        "\n\n"
    )
    out += table(
        ["var", "sel", "layout", "speedup", "95% CI", "fastest baseline", "P(faster)"],
        best,
    )
    out += "\n\n### SkipZFP over every baseline\n\n"
    out += table(
        ["var", "sel", "layout", "baseline", "speedup", "95% CI", "P(faster)"], each
    )
    return out


def ablation(rows):
    rng = random.Random(SEED)
    g = defaultdict(list)
    for r in rows:
        g[(r["var"], r["sel"], r["config"])].append(r)
    lines, tail = [], []
    for (var, sel, config), v in sorted(g.items()):
        t = [r["seconds"] for r in v]
        ref = v[0].get("reference", "default")
        d = [r["seconds"] for r in g.get((var, sel, ref), [])]
        lo, hi = ratio_ci(t, d, rng)
        lines.append(
            [
                var,
                sel,
                config,
                v[0]["factor"],
                len(v),
                *spread(t),
                ref,
                f"{median(t) / median(d):.3f}" if d else "",
                f"[{lo:.3f}, {hi:.3f}]",
                f"{p_faster(t, d):.2f}" if d else "",
                f"{median([r['bytes'] for r in v]) / 1e6:.1f}",
                f"{median([r.get('useful_payload_bytes', 0) for r in v]) / 1e6:.1f}",
                f"{median([r['requests'] for r in v]):.0f}",
                f"{median([read_fraction(r) for r in v]):.4f}",
                f"{max(r.get('peak_rss_kb') or 0 for r in v) / 1024:.0f}",
            ]
        )
        if len(v) >= 20:
            tail.append(
                [var, sel, config, len(v)]
                + [f"{quantile(t, q):.3f}" for q in (0.5, 0.95, 0.99)]
            )
    head = ["var", "sel", "config", "factor", "n", "p25 s", "median s", "p75 s"]
    head += ["p95 s", "reference", "/ reference", "95% CI", "P(faster than ref.)"]
    head += ["MB"]
    head += ["useful MB", "requests", "MAYBE frac", "peak RSS MB"]
    out = "## ablation\n\n" + runs_of(rows) + "\n\n" + table(head, lines)
    out += (
        "\n\nMB counts every byte read, metadata and over-read included; useful MB"
        " is the IN and MAYBE blocks an AVG reads, or the MAYBE blocks a COUNT reads."
        " no_pruning reads whole chunks: its MAYBE frac counts chunks, not blocks."
    )
    if tail:
        out += "\n\n### tail latency\n\n"
        out += table(["var", "sel", "config", "n", "p50 s", "p95 s", "p99 s"], tail)
    return out


def memory(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["name"], r["limit"])].append(r)
    lines = []
    for (name, limit), v in sorted(g.items()):
        ok = [r for r in v if not r["oom"]]

        def most(key, scale, ok=ok):
            return f"{max((r.get(key) or 0) for r in ok) / scale:.0f}" if ok else ""

        lines.append(
            [
                name,
                limit,
                f"{sum(r['oom'] for r in v)}/{len(v)}",
                f"{median([r['seconds'] for r in ok]):.3f}" if ok else "",
                most("peak_rss_kb", 1024),
                most("cgroup_peak_bytes", 2**20),
                most("cgroup_file_bytes", 2**20),
                most("peak_inflight_bytes", 2**20),
            ]
        )
    head = ["method", "limit", "killed", "median s (survivors)", "VmHWM MB"]
    head += ["cgroup peak MB", "page cache MB", "in-flight MB"]
    out = "## memory\n\n" + runs_of(rows) + "\n\n" + table(head, lines)
    return out + (
        "\n\nThe median time covers only the jobs that were not killed, which at a"
        " tight limit are biased toward the cheaper selectivity."
    )


def build(rows, abl):
    g = defaultdict(list)
    for r in rows:
        g[(r["var"], r["layout"])].append(r)
    lines = []
    for (var, layout), v in sorted(g.items()):
        meta_s = median([r["meta_seconds"] for r in v])
        repay = []
        for sel in sorted({r["sel"] for r in abl if r["var"] == var}):

            def med(c, var=var, sel=sel):
                return median(
                    [
                        r["seconds"]
                        for r in abl
                        if (r["var"], r["sel"], r["config"]) == (var, sel, c)
                    ]
                )

            saved = med("no_pruning") - med("bits4" if "-b4" in layout else "default")
            repay.append(f"{sel}: {meta_s / saved:.1f}" if saved > 0 else f"{sel}: -")
        lines.append(
            [
                var,
                layout,
                len(v),
                f"{median([r['payload_seconds'] for r in v]):.1f}",
                f"{meta_s:.2f}",
                f"{max(r['meta_peak_anon_kb'] for r in v) / 1024:.0f}",
                f"{v[0]['payload_bytes'] / 1e6:.1f}",
                f"{v[0]['meta_bytes'] / 1e6:.2f}",
                f"{100 * v[0]['meta_bytes'] / v[0]['payload_bytes']:.2f}%",
                "; ".join(repay),
            ]
        )
    head = ["var", "layout", "n", "payload s", "zone map s", "zone map anon MB"]
    head += ["payload MB", "zone map MB", "share", "queries to repay, by sel"]
    return "## build cost\n\n" + runs_of(rows) + "\n\n" + table(head, lines)


def stages(rows):
    """Median time of each step of a query, and cold open, per method."""
    g = defaultdict(list)
    for r in rows:
        g[(r["var"], r["sel"], r["method"], r["size"])].append(r)

    def med(v, k):
        xs = [r[k] for r in v if r.get(k) is not None]
        return f"{median(xs):.3f}" if xs else ""

    lines = [
        [var, sel, method, size, len(v)]
        + [med(v, k) for k in ("seconds", "metadata_seconds", "planning_seconds")]
        + [med(v, k) for k in ("payload_seconds", "decode_seconds")]
        for (var, sel, method, size), v in sorted(g.items())
    ]
    head = ["var", "sel", "method", "size", "n", "wall s", "zone map s", "planning s"]
    head += ["payload s", "decode s (summed)"]
    out = "## exp2 query steps\n\n" + runs_of(rows) + "\n\n"
    out += (
        "Wall, zone map and payload are wall times on the critical path; with the "
        "pipeline they overlap, so they need not add up to the wall time. Planning "
        "with the pipeline, and decode always, are summed over work running in "
        "parallel, so they can exceed the wall time and must not be stacked.\n\n"
    )
    out += table(head, lines)

    o = defaultdict(list)
    for r in rows:
        o[(r["method"], r["size"])].append(r)
    lines = [
        [m, sz, len(v)]
        + [f"{quantile([r['open_seconds'] for r in v], q):.3f}" for q in (0.5, 0.95)]
        + [f"{median([r['open_requests'] for r in v]):.0f}"]
        + [f"{median([r['open_bytes'] for r in v]):.0f}"]
        for (m, sz), v in sorted(o.items())
    ]
    out += "\n\n### cold open\n\n"
    out += (
        "Opening the array, and for a baseline its zone map and shard index arrays, "
        "before the query is timed.\n\n"
    )
    return out + table(
        ["method", "size", "n", "median s", "p95 s", "requests", "bytes"], lines
    )


def tightness(rows):
    """How much 4-bit per-block error bounds narrow the AVG and COUNT intervals."""
    g = defaultdict(list)
    for r in rows:
        if r.get("original_mean"):
            g[(r["var"], r["sel"], r["config"])].append(r)

    def width(v, k):
        return median([r[k][1] - r[k][0] for r in v])

    lines = []
    for var, sel in sorted({(k[0], k[1]) for k in g}):
        a, b = g.get((var, sel, "avg")), g.get((var, sel, "avg_bits4"))
        if not a or not b:
            continue
        wa, wb = width(a, "original_mean"), width(b, "original_mean")
        ca, cb = width(a, "original_count"), width(b, "original_count")
        lines.append(
            [var, sel, f"{wa:.4g}", f"{wb:.4g}", f"{wa / wb:.2f}x"]
            + [
                f"{ca / a[0]['truth']:.4g}",
                f"{cb / b[0]['truth']:.4g}",
                f"{ca / cb:.2f}x",
            ]
            + [f"{median([r['maybe_blocks'] for r in a]):.0f}"]
            + [f"{median([r['maybe_blocks'] for r in b]):.0f}"]
            + [f"{median([r['metadata_bytes'] for r in a]) / 1e6:.1f}"]
            + [f"{median([r['metadata_bytes'] for r in b]) / 1e6:.1f}"]
        )
    head = ["var", "sel", "AVG width", "AVG width 4-bit", "narrower"]
    head += ["COUNT width / truth", "4-bit", "narrower", "MAYBE", "MAYBE 4-bit"]
    head += ["zone map MB", "4-bit MB"]
    out = "## 4-bit error bounds: interval width\n\n" + runs_of(rows) + "\n\n"
    out += (
        "Width of the guaranteed interval on the original AVG and COUNT, read from "
        "the store, with one error bound per sub-chunk and with a 4-bit one per "
        "block.\n\n"
    )
    return out + table(head, lines)


def timeline(rows):
    """Where a query's time goes with the pipeline on and off, from the traces."""
    g = defaultdict(list)
    for r in rows:
        if r.get("trace"):
            g[(r["var"], r["sel"], r["config"])].append(r["trace"])

    def med(xs):
        return f"{median(xs):.3f}" if xs else ""

    lines = []
    for (var, sel, config), ts in sorted(g.items()):
        reads = [t["reads"] for t in ts if t["reads"]]
        lines.append(
            [var, sel, config, len(ts)]
            + [med([min(t["meta"]) for t in ts]), med([max(t["meta"]) for t in ts])]
            + [med([min(x[0] for x in rd) for rd in reads])]
            + [med([max(x[2] for x in rd) for rd in reads])]
            + [med([statistics.mean(x[1] - x[0] for x in rd) for rd in reads])]
            + [med([statistics.mean(x[2] - x[1] for x in rd) for rd in reads])]
        )
    head = ["var", "sel", "config", "n", "first zone map s", "last zone map s"]
    head += ["first read issued s", "last read decoded s"]
    head += ["mean issued to arrived s", "mean arrived to decoded s"]
    out = "## pipeline timeline\n\n" + runs_of(rows) + "\n\n"
    out += (
        "Seconds since the query started, medians over runs. Issued to arrived "
        "includes waiting for a free request slot; arrived to decoded includes "
        "waiting for a decoder thread.\n\n"
    )
    return out + table(head, lines)


def point(rows):
    """exp1: bytes, requests, latency and throughput of single-point queries, and
    the time of batches of points."""
    g = defaultdict(list)
    for r in rows:
        g[(r["mode"], r["method"], r["size"])].append(r)
    lines = []
    for (mode, method, size), v in sorted(g.items()):
        if mode != "latency":
            continue
        lat = [x for r in v for x in r["lat"]]
        n = sum(r["n"] for r in v)
        qps = [r["qps"] for r in g.get(("throughput", method, size), [])]
        lines.append(
            [method, size, len(lat)]
            + [f"{sum(r['bytes'] for r in v) / n / 1e3:.0f}"]
            + [f"{sum(r['requests'] for r in v) / n:.0f}"]
            + [f"{1e3 * median(lat):.0f}", f"{1e3 * quantile(lat, 0.9):.0f}"]
            + [f"{median(qps):.1f}" if qps else ""]
        )
    head = ["method", "size", "queries", "KB per query", "requests per query"]
    head += ["median ms", "p90 ms", "queries/s"]
    out = "## exp1 point queries\n\n" + runs_of(rows) + "\n\n"
    out += (
        "Latency pools every single-point query of every run; queries/s is the "
        "median over runs of N queries issued together.\n\n"
    )
    out += table(head, lines)
    batch = []
    for (mode, method, size), v in sorted(g.items()):
        if mode not in ("random", "cluster"):
            continue
        for n in sorted({r["n"] for r in v}):
            w = [r for r in v if r["n"] == n]
            batch.append(
                [
                    mode,
                    n,
                    method,
                    size,
                    len(w),
                    f"{median([r['seconds'] for r in w]):.3f}",
                ]
                + [f"{median([r['bytes'] for r in w]) / 1e6:.1f}"]
                + [f"{median([r['requests'] for r in w]):.0f}"]
            )
    out += "\n\n### batches of points\n\n"
    return out + table(
        ["mode", "points", "method", "size", "n", "median s", "MB", "requests"], batch
    )


def aggregate(rows):
    """exp3: regional AVG error, its bound, and sampling at the same bytes."""
    lines, worst = [], 0
    for var in sorted({r["var"] for r in rows}):
        v = [r for r in rows if r["var"] == var]
        for rate in sorted(v[0]["skipzfp"], key=float):
            sk = [r["skipzfp"][rate] for r in v]
            sa = [r["sampling"][rate] for r in v]
            worst += sum(s["err"][op] > s["bound"][op] for s in sk for op in s["err"])
            k, n = sum(s["ci_misses"] for s in sa), sum(s["trials"] for s in sa)
            lines.append(
                [var, rate, len(v), f"{median([s['bytes_frac'] for s in sk]):.4f}"]
                + [f"{median([s['err']['AVG'] for s in sk]):.2g}"]
                + [f"{median([s['bound']['AVG'] for s in sk]):.3g}"]
                + [f"{median([s['err_median'] for s in sa]):.3g}"]
                + [f"{median([s['bytes_median'] / s['budget_bytes'] for s in sa]):.3f}"]
                + [f"{k / n:.3f}", "[{:.3f}, {:.3f}]".format(*wilson(k, n))]
                + [f"{median([s['ci_miss'] for s in sa]):.3f}"]
                + [f"{max(s['ci_miss'] for s in sa):.3f}"]
            )
    head = ["var", "bpv", "regions", "payload frac", "AVG error", "AVG bound"]
    head += ["sampling error", "sampled / budget bytes", "CI miss overall", "95% CI"]
    head += ["median region", "worst region"]
    out = "## exp3 regional aggregates\n\n" + runs_of(rows) + "\n\n"
    out += (
        f"Medians over regions. {worst} of the AVG, MIN and MAX errors exceed their "
        "bound. CI miss overall pools every trial of every region, with a Wilson "
        "interval; the region columns give the spread of the per-region rate.\n\n"
    )
    return out + table(head, lines)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def filtered(rows):
    """exp4: the metadata-only COUNT interval against zone maps, and the
    store-read COUNT and AVG."""
    # SkipZFP first, then the zone maps from the finest (most metadata) down
    sizes = rows[0]["meta_only"]
    names = sorted(sizes, key=lambda k: (k != "skipzfp", -sizes[k]["meta_bytes"]))
    lines = []
    for r in sorted(rows, key=lambda r: (r["var"], r["sel"])):
        lines.append(
            [r["var"], r["sel"]] + [f"{r['meta_only'][k]['width']:.3g}" for k in names]
        )
    head = ["var", "sel"] + [f"{k} width" for k in names]
    out = "## exp4 filtered aggregates\n\n" + runs_of(rows) + "\n\n"
    out += (
        "Width of the metadata-only interval for the original COUNT, over the true "
        "count. Zone map metadata bytes: "
        + ", ".join(
            f"{k} {rows[0]['meta_only'][k]['meta_bytes'] / 1e6:.3g} MB" for k in names
        )
        + ".\n\n"
    )
    out += table(head, lines)
    store = []
    for r in sorted(rows, key=lambda r: (r["var"], r["sel"])):
        st, dec = r["store"], r["decoded"]
        lo, hi = st["original_count_interval"]
        store.append(
            [r["var"], r["sel"], f"{dec['count_rel_err']:.2g}"]
            + [f"{(hi - lo) / r['truth']:.3g}"]
            + [f"{st['original_avg_interval'][1] - st['original_avg_interval'][0]:.3g}"]
            + [
                f"{st['payload_bytes'] / 1e6:.1f}",
                f"{st['useful_payload_bytes'] / 1e6:.1f}",
            ]
            + [f"{st['metadata_bytes'] / 1e6:.1f}", st["payload_requests"]]
            + [lo <= r["truth"] <= hi]
        )
    out += "\n\n### read from the store\n\n"
    return out + table(
        ["var", "sel", "count rel. error", "COUNT width / truth", "AVG width"]
        + ["payload MB", "useful MB", "zone map MB", "requests", "contains truth"],
        store,
    )


def overhead(rows):
    """exp5: zone map bytes over payload bytes, for 3D and 2D blocks."""
    lines = [
        [r["rate"], r["meta_3d"], r["payload_3d"], f"{100 * r['overhead_3d']:.2f}%"]
        + [f"{100 * r['overhead_2d']:.2f}%"]
        for r in sorted(rows, key=lambda r: r["rate"])
    ]
    head = ["bpv", "zone map bytes", "payload bytes", "3D overhead", "2D overhead"]
    return "## exp5 metadata overhead\n\n" + runs_of(rows) + "\n\n" + table(head, lines)


def sensitivity(rows):
    """Object size: SkipZFP with 64-step and longer storage objects."""
    g = defaultdict(list)
    for r in rows:
        g[(r["var"], r["sel"], r["cfg"])].append(r)
    lines = [
        [var, sel, cfg, len(v), *spread([r["seconds"] for r in v])]
        + [f"{median([r['bytes'] for r in v]) / 1e6:.1f}"]
        + [f"{median([r['requests'] for r in v]):.0f}"]
        for (var, sel, cfg), v in sorted(g.items())
    ]
    head = ["var", "sel", "config", "n", "p25 s", "median s", "p75 s", "p95 s", "MB"]
    head += ["requests"]
    return "## object size\n\n" + runs_of(rows) + "\n\n" + table(head, lines)


def machines(root):
    """Host and calibration of every run behind the results."""
    lines = []
    for side in sorted(Path(root).glob("*.runs.json")):
        for run_id, r in json.loads(side.read_text()).items():
            cal = r.get("calibration") or {}
            src = r.get("source") or {}
            lines.append(
                [side.name.removesuffix(".runs.json"), run_id]
                + [str(src.get("commit", ""))[:10], src.get("dirty", "")]
                + [r["machine"].get("host", ""), r["machine"].get("cpu", "")]
                + [f"{cal[k]:.4f}" if k in cal else "" for k in CALIBRATION]
            )
    head = ["file", "run", "commit", "dirty", "host", "cpu"] + list(CALIBRATION)
    return "## runs\n\n" + table(head, lines)


CALIBRATION = ("encode_chunk_s", "decode_chunk_s", "sum_512mb_s")


def main(root, out):
    root = Path(root)
    parts = []
    rows = load(root / "exp1_point_query.jsonl")
    if rows:
        parts.append(point(rows))
    rows = load(root / "exp2_threshold_query.jsonl")
    if rows:
        parts.append(exp2(rows))
        parts.append(stages(rows))
    abl = load(root / "ablation.jsonl")
    if abl:
        parts.append(ablation(abl))
        parts.append(tightness(abl))
        if any(r.get("trace") for r in abl):
            parts.append(timeline(abl))
    rows = load(root / "memory_limit.jsonl")
    if rows:
        parts.append(memory(rows))
    rows = load(root / "build_cost.jsonl")
    if rows:
        parts.append(build(rows, abl))
    rows = load(root / "sensitivity_object_size.jsonl")
    if rows:
        parts.append(sensitivity(rows))
    for name, fn in (
        ("exp3_aggregate.json", aggregate),
        ("exp4_filtered_aggregate.json", filtered),
        ("exp5_overhead.json", overhead),
    ):
        if (root / name).exists():
            parts.append(fn(json.loads((root / name).read_text())))
    parts.append(machines(root))
    text = "\n\n".join(parts) + "\n"
    Path(out).write_text(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="summary.md")
    args = ap.parse_args()
    main(args.root, args.out)
