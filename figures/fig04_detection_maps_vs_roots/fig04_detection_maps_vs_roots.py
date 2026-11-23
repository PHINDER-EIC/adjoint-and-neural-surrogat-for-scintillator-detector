"""Reverse tracing, simplified (replaces fig_backtracing; the old script fig_backtrace.py and its figure are kept).

(a) one root launched from the readout face, top view: thin grey inside the guide, coloured by its remaining weight
    inside the scintillator (the weight drops at every partial reflection and with absorption; splits create branches);
(b) the same root from the side;
(c, separate figure fig_backtrace_maps) the detection-probability map (averaged over depth) estimated from the first 1, 30, 1000 and 20,000 roots, with the
    fraction of voxels reached and the voxel error against the 2e5-root reference.
The root geometry is recorded from the repo's Python tracer; the maps come from the numba tracer with the same Halton
launch states and physics stream (roots start at the recorded root's index).
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as mt

import style as Sty

ROOT = Path(str(paths.REPO))
T = paths.DATA
sys.path[:0] = [str(ROOT / "Scintillator-Simulator-main/extras/numba_adjoint"), str(ROOT / "phinder_bridge/scripts"),
                str(ROOT / "Scintillator-Simulator-main/src")]
import numba_adjoint as na  # noqa: E402
import phinder_stage6_adjoint_geometry_atlas as A  # noqa: E402
import phinder_stage6_adjoint_field as field  # noqa: E402
import common as C  # noqa: E402

GID = "B_train_g0180"
LS, PS = 606271, 808291
ARGS = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005,
                             max_live_branches=64, max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0,
                             start_epsilon_cm=1e-7)
raw = next(g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"] if g["id"] == GID)
spec, path0, path1, cfg = A._geometry_cfg(raw, ARGS, LS, PS)
tr = na.from_stage6_cfg(cfg)
field._worker_init(cfg, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
X, Y, Z = map(float, cfg["cell_cm"])

# ---------------------------------------------------------------- record one root with the repo's Python tracer
SEGS, SCORED = [], []
_hit = field._W["scene"].first_boundary_hit
_score = field._score_scint_segment


def hit_rec(p, d, *a, **k):
    h = _hit(p, d, *a, **k)
    if h is not None:
        SEGS.append((np.array(p, float), np.array(h.point, float), getattr(h.from_volume, "name", "")))
    return h


def score_rec(p0, p1, t0, w0, *a, **k):
    SCORED.append((np.array(p0, float), np.array(p1, float), float(w0)))
    return _score(p0, p1, t0, w0, *a, **k)


field._W["scene"].first_boundary_hit = hit_rec
field._score_scint_segment = score_rec
U = tr.halton(400, LS)
root = None
for i in range(400):
    SEGS.clear(); SCORED.clear()
    field._track_root(i, U[i])
    nsc = sum(1 for s in SEGS if s[2] == "scintillator")
    if 6 <= nsc <= 40 and len(SEGS) <= 400:
        root = dict(i=i, segs=list(SEGS), scored=list(SCORED))
        break
print("root", root["i"], len(root["segs"]), len(root["scored"]), flush=True)

# ---------------------------------------------------------------- maps from the first N roots (numba tracer)
_, _, core_w, core_h = field.base._build_from_cfg(cfg)
norm = 0.25 * core_w * core_h * (1.59 / 1.58) ** 2
vox = X * Y * Z / 3200
with h5py.File(C.TRUTH_H5) as f:
    ref = f["geometries"][GID]["field/p_per_emitted"][:]
NS = [1, 30, 1000, 20000]
maps = {}
for n in NS:
    m, _ = tr.run(n, LS, PS, start=root["i"])
    p = norm * m["sumL"] / m["n"] / vox
    w = ref / ref.sum()
    ok = p > 0
    reached = float(ok.mean())
    # error over the voxels reached (all of them from 30 roots on), p-weighted rms of ln(map / reference)
    err = float(np.sqrt(np.sum(w[ok] * C.resid(p[ok], ref[ok]) ** 2) / w[ok].sum())) if n > 1 else None
    maps[n] = dict(p=p.mean(2), side=p.mean(1), reached=reached, err=err)
    print(n, "reached", reached, "err", err, flush=True)

# ---------------------------------------------------------------- branch statistics of the recorded root
starts = {}
for a, b, v in root["segs"]:
    starts.setdefault(tuple(np.round(a, 6)), 0)
    starts[tuple(np.round(a, 6))] += 1
n_split = sum(1 for k, c in starts.items() if c >= 2)
wts = np.array([w for *_, w in root["scored"]])
print("branch points", n_split, "scored segments", len(root["scored"]), "weights", wts.min(), wts.max(), flush=True)

Sty.setup()
cmap = plt.get_cmap("viridis_r")
wnorm = LogNorm(max(wts.min(), 1e-3), 1.0)
port = tr.endpoint
P0 = np.asarray(path0.points); P1 = np.asarray(path1.points)

# ================================================================ figure 1: one root (single column)
fig = plt.figure(figsize=(3.4, 3.55))
axA = fig.add_axes((0.14, 0.40, 0.66, 0.54))
axB = fig.add_axes((0.14, 0.08, 0.66, 0.2))
cax = fig.add_axes((0.83, 0.44, 0.025, 0.46))
axA.plot([0, X, X, 0, 0], [0, 0, Y, Y, 0], color=Sty.INK2, lw=0.8)
axA.plot(P0[:, 0], P0[:, 1], color=Sty.COL["gp"], lw=5, alpha=0.3, solid_capstyle="butt")
for xx in np.linspace(0, X, 21):
    axA.plot([xx, xx], [0, Y], color=Sty.GRID, lw=0.25, zorder=0)
for yy in np.linspace(0, Y, 21):
    axA.plot([0, X], [yy, yy], color=Sty.GRID, lw=0.25, zorder=0)
axA.add_collection(LineCollection([(a[:2], b[:2]) for a, b, v in root["segs"] if v != "scintillator"],
                                  colors=Sty.INK2, linewidths=0.5))
lc = LineCollection([(a[:2], b[:2]) for a, b, _ in root["scored"]], array=wts, cmap=cmap, norm=wnorm, linewidths=1.0)
axA.add_collection(lc)
axA.plot(port[0], port[1], "*", color=Sty.INK, ms=8, zorder=5)
axA.annotate("readout", (port[0], port[1]), xytext=(port[0] - 1.7, port[1] + 0.3), fontsize=6, va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=Sty.INK3))
axA.annotate("guide 0", (P0[2, 0], P0[2, 1]), xytext=(P0[2, 0] + 0.9, P0[2, 1] - 0.5), fontsize=6, color=Sty.INK2,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=Sty.INK3))
axA.set_aspect("equal"); axA.set_xlim(-0.15, X + 0.15); axA.set_ylim(-0.15, Y + 0.6); axA.grid(False)
axA.set_xlabel("x (cm)", labelpad=1); axA.set_ylabel("y (cm)")
Sty.panel(axA, "a", "One reverse root, top view")
cb = fig.colorbar(lc, cax=cax); cb.set_label("remaining weight", fontsize=6); cb.ax.tick_params(labelsize=5.5)
axB.plot([0, X, X, 0, 0], [0, 0, Z, Z, 0], color=Sty.INK2, lw=0.8)
axB.fill_between([P0[:, 0].min(), P0[:, 0].max()], Z, Z + 0.1, color=Sty.COL["gp"], alpha=0.35, lw=0)
axB.fill_between([P1[:, 0].min(), P1[:, 0].max()], -0.1, 0, color=Sty.COL["gp"], alpha=0.35, lw=0)
axB.text(X + 0.08, Z + 0.05, "guide 0", fontsize=5.5, va="center", color=Sty.INK2)
axB.text(X + 0.08, -0.05, "guide 1", fontsize=5.5, va="center", color=Sty.INK2)
axB.add_collection(LineCollection([(a[[0, 2]], b[[0, 2]]) for a, b, v in root["segs"] if v != "scintillator"],
                                  colors=Sty.INK2, linewidths=0.5))
axB.add_collection(LineCollection([(a[[0, 2]], b[[0, 2]]) for a, b, _ in root["scored"]], array=wts, cmap=cmap,
                                  norm=wnorm, linewidths=1.0))
axB.plot(port[0], port[2], "*", color=Sty.INK, ms=8, zorder=5)
axB.set_aspect("equal"); axB.set_xlim(-0.15, X + 0.15); axB.set_ylim(-0.2, Z + 0.2); axB.grid(False)
axB.set_xlabel("x (cm)", labelpad=1); axB.set_ylabel("z (cm)")
Sty.panel(axB, "b", "Same root, side view")
plt.close(fig)

# ================================================================ figure 2: maps from the first N roots, top and side view
fig = plt.figure(figsize=(7.1, 2.45))
W = 0.19
xs = [0.035 + k * 0.205 + (0.045 if k else 0) for k in range(4)]
axM = [fig.add_axes((x, 0.30, W, 0.62)) for x in xs]           # top view (average over depth)
axS = [fig.add_axes((x, 0.03, W, 0.25)) for x in xs]           # side view (average over y)
cax1 = fig.add_axes((xs[0] + W + 0.006, 0.10, 0.008, 0.74))
cax2 = fig.add_axes((xs[3] + W + 0.008, 0.10, 0.008, 0.74))
fin = np.concatenate([maps[NS[-1]]["p"].ravel(), maps[NS[-1]]["side"].ravel()])
mnorm = Normalize(np.quantile(fin, 0.02), fin.max())   # range is a factor ~1.7: linear
one = np.concatenate([maps[1]["p"].ravel(), maps[1]["side"].ravel()])
onorm = LogNorm(max(one[one > 0].min(), one.max() * 1e-3), one.max())
port1 = (X - port[0], port[1])
for k, n in enumerate(NS):
    d = maps[n]
    nm = onorm if n == 1 else mnorm
    # top view
    ax = axM[k]
    im = ax.imshow(np.ma.masked_less_equal(d["p"], 0).T, origin="lower", extent=(0, X, 0, Y), cmap=Sty.SEQ, norm=nm,
                   interpolation="nearest")
    if n == 1:
        im1 = im
    else:
        imN = im
    ax.set_facecolor("white")
    ax.plot([0, X, X, 0, 0], [0, 0, Y, Y, 0], color=Sty.INK2, lw=0.6)
    ax.plot(P0[:, 0], P0[:, 1], color=Sty.INK, lw=0.9, alpha=0.8, solid_capstyle="butt")
    ax.plot(P1[:, 0], P1[:, 1], color=Sty.INK, lw=0.9, alpha=0.8, ls=(0, (3, 2)))
    ax.plot(port[0], port[1], "*", color=Sty.INK, ms=6, zorder=5, clip_on=False)
    ax.plot(*port1, "*", mfc="white", mec=Sty.INK, mew=0.6, ms=6, zorder=5, clip_on=False)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_anchor("S")
    ax.set_xlim(0, X); ax.set_ylim(0, Y + 0.3)
    nl = f"{n:,}" if n < 10000 else r"$2\times10^4$"
    ax.set_title(f"{nl} root{'s' if n > 1 else ''}", fontsize=6.6, fontweight="normal", pad=3)
    txt = f"{100*d['reached']:.0f}% of voxels reached" if d["err"] is None else f"error {100*d['err']:.1f}%"
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=5.4, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    # side view
    ax = axS[k]
    ax.imshow(np.ma.masked_less_equal(d["side"], 0).T, origin="lower", extent=(0, X, 0, Z), cmap=Sty.SEQ, norm=nm,
              interpolation="nearest")
    ax.set_facecolor("white")
    ax.plot([0, X, X, 0, 0], [0, 0, Z, Z, 0], color=Sty.INK2, lw=0.6)
    ax.fill_between([P0[:, 0].min(), P0[:, 0].max()], Z, Z + 0.12, color=Sty.INK, alpha=0.75, lw=0)
    ax.fill_between([P1[:, 0].min(), P1[:, 0].max()], -0.12, 0, color=Sty.INK, alpha=0.35, lw=0, hatch="////",
                    edgecolor="white")
    ax.plot(port[0], Z + 0.06, "*", color=Sty.INK, ms=5, zorder=5, clip_on=False)
    ax.plot(X - port[0], -0.06, "*", mfc="white", mec=Sty.INK, mew=0.6, ms=5, zorder=5, clip_on=False)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlim(0, X); ax.set_ylim(-0.14, Z + 0.14); ax.set_anchor("N")
    for s_ in ax.spines.values():
        s_.set_visible(False)
axS[0].text(-0.03, 0.5, "side", transform=axS[0].transAxes, rotation=90, fontsize=5.8, ha="right", va="center",
            color=Sty.INK2)
axM[0].text(-0.03, 0.5, "top", transform=axM[0].transAxes, rotation=90, fontsize=5.8, ha="right", va="center",
            color=Sty.INK2)
c1 = fig.colorbar(im1, cax=cax1)
c1.ax.yaxis.set_major_locator(mt.LogLocator(10, numticks=10))
c1.ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{100*v:g}%"))
c1.ax.yaxis.set_minor_formatter(mt.NullFormatter()); c1.ax.tick_params(which="both", labelsize=5)
c2 = fig.colorbar(imN, cax=cax2)
c2.ax.yaxis.set_major_locator(mt.MultipleLocator(0.001))
c2.ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{100*v:.1f}%"))
c2.ax.yaxis.set_minor_locator(mt.NullLocator()); c2.ax.tick_params(which="both", labelsize=5)
c2.set_label("detection probability\n(top: depth average; side: average over y)", fontsize=5.6)
from matplotlib.lines import Line2D  # noqa: E402
fig.legend([Line2D([], [], color=Sty.INK, lw=0.9), Line2D([], [], color=Sty.INK, lw=0.9, ls=(0, (3, 2))),
            Line2D([], [], color=Sty.INK, marker="*", lw=0, ms=6),
            Line2D([], [], color=Sty.INK, marker="*", lw=0, ms=6, mfc="white", mew=0.6)],
           ["guide 0 (top face)", "guide 1 (bottom face)", "readout 0 (roots start here)", "readout 1"],
           loc="upper center", ncol=4, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.0))
Sty.save(fig, "fig_backtrace_maps")
json.dump(dict(root=root["i"], n_segments=len(root["segs"]), n_scored=len(root["scored"]), branch_points=n_split,
               weight_range=[float(wts.min()), float(wts.max())],
               maps={str(n): dict(reached=v["reached"], err=v["err"]) for n, v in maps.items()}),
          open(Sty.FIG / "backtrace2.json", "w"), indent=1)
