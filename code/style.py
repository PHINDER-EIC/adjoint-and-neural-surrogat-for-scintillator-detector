"""Figure style: one colour per method, used in every figure; vector PDF + 300-dpi PNG."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import os
import sys
FIG = Path(os.environ.get("FIG_DIR", Path(sys.argv[0]).resolve().parent))   # default: next to the figure script
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COL = dict(adjoint="#2a78d6", network="#eb6834", gp="#1baf7a", jax="#8a8984", g4="#0b0b0b")
MRK = dict(adjoint="o", network="s", gp="D", jax="^", g4="v")
LAB = dict(adjoint="Adjoint (numba)", network="Network surrogate", gp="GP surrogate", jax="JAX forward",
           g4="Geant4 forward")
BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE = ["#f6b89c", "#f29b76", "#ee8255", "#eb6834", "#cf5526", "#a9441d", "#823415"]
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                    "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("divbr", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#9c2423"])


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5,
        "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5, "legend.frameon": False, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.3, "lines.markersize": 4,
        "errorbar.capsize": 0, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "figure.dpi": 150, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "mathtext.default": "regular", "figure.constrained_layout.use": True,
    })


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name, flush=True)


def panel(ax, letter, title=""):
    ax.set_title(f"{letter}  {title}" if title else letter, loc="left", fontsize=7.5, fontweight="bold", pad=4)


def pct(ax, axis="y"):
    f = mpl.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def logpct(ax, axis="y", ticks=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)):
    """log axis with plain percent labels and no minor labels."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    t = [v for v in ticks if lo <= v <= hi]
    a.set_major_locator(mpl.ticker.FixedLocator(t))
    a.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    a.set_minor_formatter(mpl.ticker.NullFormatter())


def logx_int(ax, ticks):
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
