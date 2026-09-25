"""README diagram: how SkipZFP writes a chunk and plans a threshold query.

usage: python readme_diagram.py

Writes .github/how-it-works-light.svg and .github/how-it-works-dark.svg.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / ".github"
FONT = "sans-serif"
for f in sorted(Path.home().glob(".cache/tectonic/bundles/data/*/LinBiolinum_R*.otf")):
    font_manager.fontManager.addfont(str(f))
    FONT = "Linux Biolinum O"

THEMES = {
    "light": dict(
        ink="#1f2328",
        muted="#59636e",
        line="#8c959f",
        box="#f6f8fa",
        edge="#d1d9e0",
        store="#eaeef2",
        maybe="#2272dc",
        inn="#0f4c9c",
        out="#8c959f",
        on="#ffffff",
    ),
    "dark": dict(
        ink="#f0f6fc",
        muted="#9198a1",
        line="#6e7681",
        box="#151b23",
        edge="#3d444d",
        store="#212830",
        maybe="#4493f8",
        inn="#1f6feb",
        out="#6e7681",
        on="#ffffff",
    ),
}


def box(
    ax, t, x, y, w, h, text, fill=None, edge=None, color=None, bold=False, size=10.5
):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.4",
            facecolor=fill or t["box"],
            edgecolor=edge or t["edge"],
            linewidth=1,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=color or t["ink"],
        fontsize=size,
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
    )


def arrow(ax, t, a, b, color=None, rad=0.0, label=None, label_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>,head_length=4,head_width=2.4",
            mutation_scale=1,
            color=color or t["line"],
            linewidth=1.2,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        ax.text(
            *label_xy, label, ha="center", va="center", color=t["muted"], fontsize=9.5
        )


def figure(t):
    mpl.rcParams.update(
        {
            "font.family": FONT,
            "svg.fonttype": "path",
            "figure.facecolor": "none",
            "savefig.transparent": True,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    ax.text(1, 33, "write", color=t["muted"], fontsize=10)

    box(ax, t, 7, 17, 12, 7, "float32\nchunk")
    box(ax, t, 25, 25, 17, 7, "ZFP blocks\n4×4×4, fixed rate")
    box(ax, t, 25, 9, 17, 7, "block bounds\nmin, max, error")
    box(ax, t, 43, 25, 12, 7, "payload\nobject", fill=t["store"])
    box(ax, t, 43, 9, 12, 7, "metadata\narray", fill=t["store"])
    arrow(ax, t, (13, 19), (16.5, 24))
    arrow(ax, t, (13, 15), (16.5, 10))
    arrow(ax, t, (33.5, 25), (37, 25))
    arrow(ax, t, (33.5, 9), (37, 9))

    ax.text(59, 14.2, "query  x > T", color=t["muted"], fontsize=10, ha="center")
    box(ax, t, 59, 9, 12, 7, "classify\neach block")
    arrow(ax, t, (49, 9), (53, 9))
    chips = [
        (27, "MAYBE", "range-read, decode", t["maybe"]),
        (17, "IN", "count from metadata", t["inn"]),
        (7, "OUT", "skip", t["out"]),
    ]
    for y, tag, what, col in chips:
        box(
            ax,
            t,
            75,
            y,
            9,
            5.2,
            tag,
            fill=col,
            edge=col,
            color=t["on"],
            bold=True,
            size=9.5,
        )
        ax.text(80.5, y, what, va="center", color=t["ink"], fontsize=10)
        arrow(ax, t, (65, 9 + (y - 9) * 0.1), (70.4, y))
    arrow(
        ax,
        t,
        (70.5, 29),
        (49, 27.5),
        color=t["maybe"],
        rad=0.22,
        label="only these bytes",
        label_xy=(60, 33.3),
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def main():
    for theme, t in THEMES.items():
        fig = figure(t)
        path = OUT / f"how-it-works-{theme}.svg"
        fig.savefig(path, format="svg", metadata={"Date": None})
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
