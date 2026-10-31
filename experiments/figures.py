"""Draw the paper's measurement figures from the raw JSONL results, nothing typed in.

usage: python figures.py <results dir> [--out DIR]

Reads exp2_threshold_query.jsonl, exp2_cmip6_pr.jsonl, exp2_basilisk_u.jsonl,
exp2_global.jsonl, exp2_global_o.jsonl, ablation.jsonl and memory_limit.jsonl from
<results dir> (whichever exist) and writes PDF and PNG files to --out (default
<results dir>/figures):

- exp2_speedup: the fastest baseline's median time over each SkipZFP layout's, per
  selectivity, with 95% bootstrap intervals that pick the fastest baseline again in
  every resample (summarize.best_ratio_ci); exp2_speedup_cmip6_pr,
  exp2_speedup_basilisk_u and exp2_speedup_global the same for the other datasets.
- ablation_time: each config's median time over its reference config's (the
  default unless the rows name another), per selectivity, with 95% bootstrap
  intervals.
- ablation_io: each config's median bytes and requests.
- tail_latency: the distribution of wall times of the tail configs.
- memory: the largest VmHWM of each method with no limit, and how many of its jobs
  each limit killed.

The intervals use summarize.ratio_ci: both groups resampled, seed 0, 2000 resamples.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from summarize import SEED, best_ratio_ci, load, median, ratio_ci  # noqa: E402

COLORS = ["#1f63c6", "#c8621b", "#2a9d5c", "#8b4fbf", "#b23a48", "#6b7280"]


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out / name)


def times(rows, **match):
    return [r["seconds"] for r in rows if all(r.get(k) == v for k, v in match.items())]


def exp2_speedup(rows, out, name="exp2_speedup"):
    rng = random.Random(SEED)
    vars_ = sorted({r["var"] for r in rows})
    fig, axes = plt.subplots(
        1, len(vars_), figsize=(4.2 * len(vars_), 3.2), layout="constrained"
    )
    axes = axes if len(vars_) > 1 else [axes]
    for ax, var in zip(axes, vars_):
        sels = sorted(
            {r["sel"] for r in rows if r["var"] == var and r["method"] == "skipzfp"}
        )
        layouts = sorted({r["size"] for r in rows if r["method"] == "skipzfp"})
        for color, lay in zip(COLORS, layouts):
            ys, lo, hi = [], [], []
            for sel in sels:
                base = defaultdict(list)
                for r in rows:
                    if (
                        r["var"] == var
                        and r["sel"] == sel
                        and r["method"].endswith(("_zm", "_izm"))
                    ):
                        base[(r["method"], r["size"])].append(r["seconds"])
                ours = times(rows, var=var, sel=sel, method="skipzfp", size=lay)
                y, a, b = best_ratio_ci(list(base.values()), ours, rng)
                ys.append(y)
                lo.append(max(0.0, y - a))
                hi.append(max(0.0, b - y))
            ax.errorbar(
                sels, ys, yerr=[lo, hi], marker="o", color=color, label=lay, capsize=3
            )
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("selectivity")
        ax.set_title(var)
    axes[0].set_ylabel("speedup over fastest baseline")
    axes[-1].legend(frameon=False)
    save(fig, out, name)


def ablation_time(rows, out):
    rng = random.Random(SEED)
    ref = {r["config"]: r.get("reference", "default") for r in rows}
    # every config against its reference, leaving out the ones that are their own
    # reference and the traced runs, which repeat default and pipeline_off
    configs = [
        c
        for c in sorted(ref)
        if ref[c] != c and c != "default" and not c.startswith("trace_")
    ]
    vars_ = sorted({r["var"] for r in rows})
    sels = sorted({r["sel"] for r in rows if r["config"] == "default"})
    fig, axes = plt.subplots(
        len(vars_),
        1,
        figsize=(9, 3.2 * len(vars_)),
        squeeze=False,
        layout="constrained",
    )
    width = 0.8 / len(sels)
    for ax, var in zip(axes[:, 0], vars_):
        for j, sel in enumerate(sels):
            ys, lo, hi = [], [], []
            for c in configs:
                d = times(rows, var=var, sel=sel, config=ref[c])
                t = times(rows, var=var, sel=sel, config=c)
                y = median(t) / median(d)
                a, b = ratio_ci(t, d, rng)
                ys.append(y)
                lo.append(max(0.0, y - a))
                hi.append(max(0.0, b - y))
            xs = [i + (j - (len(sels) - 1) / 2) * width for i in range(len(configs))]
            ax.bar(
                xs,
                ys,
                width,
                yerr=[lo, hi],
                color=COLORS[j % len(COLORS)],
                label=f"{sel:g}",
                capsize=2,
            )
        ax.axhline(1, color="0.3", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(configs)), configs, rotation=30, ha="right")
        ax.set_ylabel(f"{var}: time / reference")
    axes[0, 0].legend(title="selectivity", frameon=False, ncol=len(sels))
    save(fig, out, "ablation_time")


def ablation_io(rows, out):
    configs = sorted({r["config"] for r in rows})
    vars_ = sorted({r["var"] for r in rows})
    fig, axes = plt.subplots(
        2, len(vars_), figsize=(5 * len(vars_), 5), squeeze=False, layout="constrained"
    )
    sels = sorted({r["sel"] for r in rows if r["config"] == "default"})
    for k, var in enumerate(vars_):
        for row, key, label in ((0, "bytes", "MB read"), (1, "requests", "requests")):
            ax = axes[row, k]
            for j, sel in enumerate(sels):
                ys = []
                for c in configs:
                    v = [
                        r[key]
                        for r in rows
                        if (r["var"], r["sel"], r["config"]) == (var, sel, c)
                    ]
                    ys.append(median(v) / (1e6 if key == "bytes" else 1))
                ax.plot(
                    range(len(configs)),
                    ys,
                    marker="o",
                    color=COLORS[j % len(COLORS)],
                    label=f"{sel:g}",
                )
            ax.set_yscale("log")
            ax.set_xticks(
                range(len(configs)), configs, rotation=40, ha="right", fontsize=7
            )
            ax.set_ylabel(label)
            ax.set_title(var if row == 0 else "")
    axes[0, -1].legend(title="selectivity", frameon=False, fontsize=7)
    save(fig, out, "ablation_io")


def tail_latency(rows, out):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["var"], r["sel"], r["config"])].append(r["seconds"])
    tails = {k: sorted(v) for k, v in groups.items() if len(v) >= 20}
    if not tails:
        return
    keys = sorted({(v, s) for v, s, _ in tails})
    fig, axes = plt.subplots(
        1, len(keys), figsize=(3.6 * len(keys), 3), squeeze=False, layout="constrained"
    )
    for ax, (var, sel) in zip(axes[0], keys):
        for color, ((_, _, c), t) in zip(
            COLORS, ((k, t) for k, t in tails.items() if k[:2] == (var, sel))
        ):
            ax.step(
                t,
                [(i + 1) / len(t) for i in range(len(t))],
                where="post",
                color=color,
                label=c,
            )
        ax.set_title(f"{var}, sel {sel:g}")
        ax.set_xlabel("seconds")
    axes[0, 0].set_ylabel("fraction of queries")
    axes[0, -1].legend(frameon=False, fontsize=7)
    save(fig, out, "tail_latency")


def limit_bytes(limit):
    """A memory limit such as 300M or 2G in bytes, with none last."""
    if limit == "none":
        return float("inf")
    scale = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    return float(limit[:-1]) * scale[limit[-1].upper()]


def memory(rows, out):
    names = sorted({r["name"] for r in rows})
    limits = sorted({r["limit"] for r in rows}, key=limit_bytes)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    peak = [
        max(
            (r.get("peak_rss_kb") or 0)
            for r in rows
            if r["name"] == n and r["limit"] == "none" and not r["oom"]
        )
        / 1024
        if any(r["name"] == n and r["limit"] == "none" for r in rows)
        else 0
        for n in names
    ]
    a1.barh(names, peak, color=COLORS[0])
    a1.set_xlabel("peak resident MB, no limit")
    width = 0.8 / len(limits)
    for j, lim in enumerate(limits):
        killed = [
            sum(r["oom"] for r in rows if r["name"] == n and r["limit"] == lim)
            / max(1, sum(1 for r in rows if r["name"] == n and r["limit"] == lim))
            for n in names
        ]
        a2.barh(
            [i + (j - (len(limits) - 1) / 2) * width for i in range(len(names))],
            killed,
            width,
            color=COLORS[j % len(COLORS)],
            label=lim,
        )
    a2.set_yticks(range(len(names)), names)
    a2.set_xlabel("fraction of jobs killed for memory")
    a2.legend(
        title="limit",
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    save(fig, out, "memory")


def main(root, out):
    root = Path(root)
    out = Path(out) if out else root / "figures"
    rows = load(root / "exp2_threshold_query.jsonl")
    if rows:
        exp2_speedup(rows, out)
    for name in ("cmip6_pr", "basilisk_u", "global", "global_o"):
        rows = load(root / f"exp2_{name}.jsonl")
        if rows:
            exp2_speedup(rows, out, f"exp2_speedup_{name}")
    rows = load(root / "ablation.jsonl")
    if rows:
        ablation_time(rows, out)
        ablation_io(rows, out)
        tail_latency(rows, out)
    rows = load(root / "memory_limit.jsonl")
    if rows:
        memory(rows, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out")
    args = ap.parse_args()
    main(args.root, args.out)
