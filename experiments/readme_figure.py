"""README figure: threshold-query speedup over a zstd full scan (ten years of t2m).

usage: python readme_figure.py

Reads results/exp2_threshold_query.jsonl and writes .github/speedup-light.svg and
.github/speedup-dark.svg (GitHub picks one by the reader's theme). SkipZFP is the k8t
layout; "best baseline" is the fastest zone-map or sharded method at each selectivity.
All times are medians.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "exp2_threshold_query.jsonl"
OUT = HERE.parent / ".github"
FONT = "sans-serif"
# Linux Biolinum, the sans companion of the paper's Libertine, if a TeX cache has it
for f in sorted(Path.home().glob(".cache/tectonic/bundles/data/*/LinBiolinum_R*.otf")):
    font_manager.fontManager.addfont(str(f))
    FONT = "Linux Biolinum O"

THEMES = {
    "light": dict(
        ours="#2272dc",
        base="#747d87",
        ink="#1f2328",
        muted="#59636e",
        grid="#d8dee4",
        zstd_zm="#e0701f",
        zshard_izm="#1a9a82",
        zfpacc_zm="#8250df",
    ),
    "dark": dict(
        ours="#4493f8",
        base="#848d97",
        ink="#f0f6fc",
        muted="#9198a1",
        grid="#30363d",
        zstd_zm="#db7630",
        zshard_izm="#2ba48b",
        zfpacc_zm="#a371f7",
    ),
}
SELECTIVITIES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
LABELS = ["0.1%", "1%", "5%", "10%", "20%", "50%"]
BASELINES = {
    "zstd_zm": "zstd + zone map",
    "zfpacc_zm": "ZFP + zone map",
    "zshard_izm": "Zarr sharding",
}


def speedups(rows, var):
    def med(**kw):
        return st.median(
            r["seconds"] for r in rows if all(r[k] == v for k, v in kw.items())
        )

    full = med(var=var, method="fullscan_zstd")
    ours, best, winner = [], [], []
    for s in SELECTIVITIES:
        ours.append(full / med(var=var, method="skipzfp", size="k8t", sel=s))
        configs = {
            (r["method"], r["size"])
            for r in rows
            if r["var"] == var and r["sel"] == s and r["method"] in BASELINES
        }
        t, m = min((med(var=var, method=m, size=z, sel=s), m) for m, z in configs)
        best.append(full / t)
        winner.append(m)
    return np.array(ours), np.array(best), winner


def figure(t, ours, best, winner):
    mpl.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 11,
            "svg.fonttype": "path",
            "axes.edgecolor": t["grid"],
            "axes.labelcolor": t["muted"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "text.color": t["ink"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    x = np.arange(len(SELECTIVITIES))
    ax.plot(x, best, color=t["base"], lw=1.6, solid_capstyle="round")
    ax.plot(x, ours, color=t["ours"], lw=2.6, solid_capstyle="round")
    ax.scatter(x, ours, s=22, color=t["ours"], zorder=3, linewidths=0)
    ax.scatter(x, best, s=34, color=[t[w] for w in winner], zorder=3, linewidths=0)
    families = [m for m in BASELINES if m in winner]
    ax.text(3.55, 4.15, "fastest baseline", color=t["muted"], va="center")
    for k, m in enumerate(families):
        y = 3.72 - 0.4 * k
        ax.scatter([3.65], [y], s=34, color=t[m], linewidths=0)
        ax.text(3.82, y, BASELINES[m], color=t["ink"], va="center")
    ax.axhline(1, color=t["grid"], lw=1, zorder=0)
    ax.set_xticks(x, LABELS)
    ax.set_xlim(-0.25, len(x) + 0.9)
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks([1, 2, 3, 4], ["1×", "2×", "3×", "4×"])
    ax.yaxis.grid(True, color=t["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.text(
        x[-1] + 0.18,
        ours[-1] + 0.12,
        "SkipZFP",
        color=t["ink"],
        fontweight="bold",
        va="center",
    )
    ax.text(
        x[-1] + 0.18, best[-1] - 0.16, "best baseline", color=t["muted"], va="center"
    )
    ax.text(
        -0.25,
        4.5,
        "How many times faster than reading the whole array",
        color=t["muted"],
        va="bottom",
    )
    ax.set_xlabel("selectivity (fraction of values that match)")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.88)
    return fig


def main():
    rows = [json.loads(line) for line in open(RES)]
    ours, best, winner = speedups(rows, "t2m")
    for theme, t in THEMES.items():
        fig = figure(t, ours, best, winner)
        path = OUT / f"speedup-{theme}.svg"
        fig.savefig(path, format="svg", metadata={"Date": None})
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
