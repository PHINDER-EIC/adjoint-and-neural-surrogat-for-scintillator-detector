"""figM4_surrogate_scaling with the 40 test geometries and the 40 new designs pooled (80 geometries).
Points: mean +- s.d. over three seeds of the pooled error; small open markers: test set and new designs alone.
Power law fitted to the pooled means. The previous test-only version is kept as figM4_surrogate_scaling_testonly.
"""
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
R = json.load(open(C.out("scaling_combined.json")))
conv = json.load(open(C.out("adj_convergence.json")))
ns = sorted({r["n"] for r in R.values()})

fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.95))
fits = {}
for ax, key, lab, letter in ((axs[0], "p", r"voxel $p$ error", "a"), (axs[1], "t", r"voxel $\bar t$ error", "b")):
    agg = {}
    for which in ("pooled", "test", "new"):
        agg[which] = np.array([[np.mean(v), np.std(v, ddof=1)] for v in
                               ([r[which][key] for r in R.values() if r["n"] == n] for n in ns)])
    m = agg["pooled"]
    ax.plot(ns, m[:, 0], color=NETC, marker="s", ms=2.8, lw=1.1, label="network (mean of 3 seeds)", zorder=3)
    # guide line n^-1/2 (fixed slope), placed through the points with n >= 100 (least squares in log)
    sel = np.array(ns) >= 100
    b, a = np.polyfit(np.log(np.array(ns)[sel]), np.log(m[sel, 0]), 1)
    a_half = float(np.mean(np.log(m[sel, 0]) + 0.5 * np.log(np.array(ns)[sel])))
    fits[key] = dict(exponent=float(b), n_min=100, a_half=a_half)
    xx = np.geomspace(70, 700, 50)
    ax.plot(xx, np.exp(a_half) * xx ** -0.5, color=Sty.INK3, lw=0.7, ls=":", zorder=1, label=r"$\propto n^{-1/2}$")
    ax.text(330, np.exp(a_half) * 330 ** -0.5 * 1.25, r"$\propto n^{-1/2}$", fontsize=6, color=Sty.INK3, ha="left")
    ref = np.mean([r[key]["rms"] for r in conv["rows"] if r["N"] == 20000 and r["steps"] == 5000])
    ax.axhline(ref, color=ADJC, ls="--", lw=0.8)
    ax.text(48, ref * 1.07, r"adjoint $2\times10^4$", fontsize=5.6, color=ADJC)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(42, 720)
    Sty.logx_int(ax, (50, 100, 200, 500))
    Sty.logpct(ax, ticks=(0.01, 0.02, 0.03, 0.05, 0.1, 0.2))
    ax.set_xlabel("training geometries"); ax.set_ylabel(lab)
    Sty.panel(ax, letter)
Sty.save(fig, "figM4_surrogate_scaling")
summary = {which: {key: {str(n): [float(np.mean(v)), float(np.std(v, ddof=1))] for n, v in
                         ((n, [r[which][key] for r in R.values() if r["n"] == n]) for n in ns)} for key in ("p", "t")}
           for which in ("pooled", "test", "new")}
summary["fits"] = fits
json.dump(summary, open(C.out("scaling_combined_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=0)[:3000])
