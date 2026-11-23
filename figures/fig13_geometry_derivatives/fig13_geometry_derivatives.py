"""Fig. 10 (figM5_derivatives) from the line scans (fd/best_derivs.json, fd_best.py): adjoint derivative = symmetric
least-squares line over +-0.04 ... +-0.16 of the design range where the design stays valid, else a quadratic through
the valid points, else the original central difference; network = slope of its own scan over the same range.
Layout as paper_figs.m5, no error bars. Saves figM5_derivatives_scan.{pdf,png}."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty

Sty.setup()
COLW = 3.4
NETC, ADJC = Sty.COL["network"], Sty.COL["adjoint"]
B = json.load(open(C.ST / "fd/best_derivs.json"))
PL = dict(cell_z="cell thickness", width="guide width", r_over_w="bend radius", turn2_x="turn-point x", turn2_y="turn-point y")
PM = dict(cell_z="o", width="s", r_over_w="D", turn2_x="^", turn2_y="v")
fig, axs = plt.subplots(2, 2, figsize=(COLW, 3.55))
spec = (("p", r"$\partial\ln\bar p/\partial u$", "a"), ("t", r"$\partial\bar t/\partial u$ (ns)", "b"),
        ("sd", r"$\partial s_t/\partial u$ (ns)", "c"), ("t10", r"$\partial t_{10}/\partial u$ (ns)", "d"))
for ax, (q, lab, letter) in zip(axs.ravel(), spec):
    vals = []
    for r in B.values():
        guide = r["param"] not in ("cell_z", "width")
        ax.plot(r[q]["adj"], r[q]["net"], PM[r["param"]], color=ADJC if guide else NETC, ms=2.8, mfc="white", mew=0.8)
        vals += [r[q]["adj"], r[q]["net"]]
    lim = max(abs(np.array(vals))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=Sty.INK3, ls="--", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    if letter in "cd":
        ax.set_xlabel("adjoint (finite difference)", fontsize=6.5)
    if letter in "ac":
        ax.set_ylabel("network", fontsize=6.5)
    Sty.panel(ax, letter, lab)
for p in PL:
    axs[0, 0].plot([], [], PM[p], color=ADJC if p not in ("cell_z", "width") else NETC, mfc="white", ms=3, label=PL[p])
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=5.6)
Sty.save(fig, "figM5_derivatives_scan")
