"""Reciprocity figure (fig:reciprocal), drawn to scale from an actual 2D trace.

Side view (x along the cell, z through it) of test cell B_train_g0180 (5.10 x 1.26 cm) with guide 0 on the top face.
(a) forward: photons from one emission point, one path reaches the port.
(b) reverse: one root launched from the port and traced with the tracer's rules in 2D: real refractive indices, the
    average Fresnel reflectance at each interface (core -> cladding -> scintillator through the 10-um cladding, treated
    as two interfaces), total internal reflection, splitting when both children keep >= 0.5% of the weight, otherwise
    the larger part, and absorption exp(-mu l). Every scintillator segment scores w l / V into the 20 x 8 voxels of the
    real grid (V = voxel volume, 0.2123 cm deep); multiplied by c = A_port/4 (n_core/n_scint)^2 this is p_v from this root.
The second guide (bottom face) is omitted for clarity. Writes fig_reciprocal_schematic and study/reciprocal_trace.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import common as C
import style as Sty

Sty.setup()
X, Z = 5.1019, 1.2588                      # cell (cm)
GX0, GX1, GT = 1.6, X + 0.25, 0.10         # guide footprint start, port position, guide thickness (cm)
N_S, N_CL, N_CO, N_A = 1.58, 1.49, 1.59, 1.0
MU_S, MU_CO = 1 / 160, 1 / 200
TAU = 0.005
NXV, NZV, DY = 20, 8, 4.2464 / 20
V = (X / NXV) * (Z / NZV) * DY
C_PORT = 0.25 * (0.42606 - 0.002) * (0.1 - 0.002) * (N_CO / N_S) ** 2


def fres(n1, n2, cos_i):
    s2 = (n1 / n2) ** 2 * (1 - cos_i ** 2)
    if s2 >= 1:
        return 1.0
    ct = np.sqrt(1 - s2)
    rs = ((n1 * cos_i - n2 * ct) / (n1 * cos_i + n2 * ct)) ** 2
    rp = ((n1 * ct - n2 * cos_i) / (n1 * ct + n2 * cos_i)) ** 2
    return 0.5 * (rs + rp)


def refract(d, nrm, n1, n2):
    """direction after refraction; nrm points into the incident medium"""
    ci = -np.dot(d, nrm); eta = n1 / n2
    k = 1 - eta ** 2 * (1 - ci ** 2)
    return eta * d + (eta * ci - np.sqrt(k)) * nrm


def reflect(d, nrm):
    return d - 2 * np.dot(d, nrm) * nrm


SEGS = []          # (p0, p1, w, medium, branch_id)
GUIDE_SPLITS = [0]  # splits drawn along the guide (the rest of the guide branch is only indicated)
MAX_GUIDE_SPLITS = 2
W_MIN = 0.01        # branches below this weight are not drawn further
EVENTS = []        # (point, kind, text)


def trace(p, d, w, med, bid, depth=0, nseg=0):
    while nseg < 14 and w > W_MIN:
        if med == "core":
            zlo, zhi, xlo, xhi = Z, Z + GT, GX0, GX1
        else:
            zlo, zhi, xlo, xhi = 0.0, Z, 0.0, X
        ts = []
        if d[0] < 0: ts.append(((xlo - p[0]) / d[0], "x0"))
        if d[0] > 0: ts.append(((xhi - p[0]) / d[0], "x1"))
        if d[1] < 0: ts.append(((zlo - p[1]) / d[1], "z0"))
        if d[1] > 0: ts.append(((zhi - p[1]) / d[1], "z1"))
        t, face = min(ts)
        q = p + t * d
        w_end = w * np.exp(-(MU_S if med == "scint" else MU_CO) * t)
        SEGS.append((p.copy(), q.copy(), w, med, bid)); nseg += 1
        nrm = {"x0": np.array([1.0, 0]), "x1": np.array([-1.0, 0]), "z0": np.array([0, 1.0]), "z1": np.array([0, -1.0])}[face]
        ci = -np.dot(d, nrm)
        if med == "core":
            if face == "x0":
                EVENTS.append((q, "cont", "continues along the guide")); return
            if face == "x1":
                EVENTS.append((q, "port", "")); return
            if face == "z1" or q[0] > X:          # guide top (cladding/air) or guide end outside the cell: total
                p, d, w = q, reflect(d, nrm), w_end; continue
            # guide bottom over the cell: core -> cladding -> scintillator
            R1 = fres(N_CO, N_CL, ci)
            if R1 >= 1:
                p, d, w = q, reflect(d, nrm), w_end; continue
            dcl = refract(d, nrm, N_CO, N_CL)
            R2 = fres(N_CL, N_S, -np.dot(dcl, nrm))
            T = (1 - R1) * (1 - R2); R = 1 - T
            dsc = refract(dcl, nrm, N_CL, N_S)
            if GUIDE_SPLITS[0] >= MAX_GUIDE_SPLITS:
                EVENTS.append((q, "cont", "")); return
            if min(R, T) >= TAU:
                GUIDE_SPLITS[0] += 1
                EVENTS.append((q, "split", f"{R:.2f} / {T:.2f}"))
                trace(q, dsc, w_end * T, "scint", bid + 1, depth + 1, nseg)
                p, d, w = q, reflect(d, nrm), w_end * R; bid += 10; continue
            if T > R:
                p, d, w, med = q, dsc, w_end, "scint"; continue
            p, d, w = q, reflect(d, nrm), w_end; continue
        # scintillator
        over_guide = face == "z1" and GX0 <= q[0] <= X
        n2 = N_CL if over_guide else N_A
        R = fres(N_S, n2, ci)
        if R >= 1:
            EVENTS.append((q, "tir", "")); p, d, w = q, reflect(d, nrm), w_end; continue
        T = 1 - R
        if over_guide:                     # back into the guide: the branch leaves the scintillator
            EVENTS.append((q, "toguide", f"{T:.2f} into guide"))
            if min(R, T) >= TAU:
                p, d, w = q, reflect(d, nrm), w_end * R; continue
            return
        if min(R, T) >= TAU:
            EVENTS.append((q, "air", f"{R:.3f} reflected, {T:.3f} lost to air"))
            dout = refract(d, nrm, N_S, N_A)
            SEGS.append((q.copy(), q + 0.12 * dout, w_end * T, "air", -1))
            p, d, w = q, reflect(d, nrm), w_end * R; continue
        if R > T:
            p, d, w = q, reflect(d, nrm), w_end; continue
        EVENTS.append((q, "lost", "")); return


theta = np.radians(68.0)                    # angle of the launch direction to the guide's normal
start = np.array([GX1, Z + 0.06])
trace(start, np.array([-np.sin(theta), -np.cos(theta)]), 1.0, "core", 0)

# voxel scores of the scintillator segments
xs, zs = np.linspace(0, X, NXV + 1), np.linspace(0, Z, NZV + 1)
score = np.zeros((NXV, NZV))
for p0, p1, w, med, _ in SEGS:
    if med != "scint":
        continue
    L = np.linalg.norm(p1 - p0)
    for t_ in (np.arange(4000) + 0.5) / 4000:
        q = p0 + t_ * (p1 - p0)
        i_, j_ = min(int(q[0] / X * NXV), NXV - 1), min(int(q[1] / Z * NZV), NZV - 1)
        score[i_, j_] += w * L / 4000 / V
pv = C_PORT * score

# ------------------------------------------------------------------ figure
fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.1, 4.0))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.06)
GREEN = Sty.COL["gp"]


def scene(a):
    a.add_patch(Rectangle((0, 0), X, Z, fc="#f4f8fd", ec=Sty.INK2, lw=0.8, zorder=0))
    a.add_patch(Rectangle((GX0, Z), GX1 - GX0, GT, fc="#cfeee2", ec=GREEN, lw=0.8, zorder=1))
    a.add_patch(Rectangle((GX1 - 0.03, Z), 0.05, GT, fc=Sty.INK, ec=Sty.INK, zorder=4))
    a.text(GX1 + 0.08, Z + 0.05, "readout\nport", fontsize=6, va="center", ha="left")
    a.text((GX0 + X) / 2, Z + GT + 0.06, "guide 0 (core + cladding)", fontsize=6, color=GREEN, ha="center", va="bottom")
    a.text(0.05, 0.06, "scintillator", fontsize=6, color=Sty.INK2, zorder=5)
    for xx in xs:
        a.plot([xx, xx], [0, Z], color=Sty.GRID, lw=0.3, zorder=0.5)
    for zz in zs:
        a.plot([0, X], [zz, zz], color=Sty.GRID, lw=0.3, zorder=0.5)
    a.set_xlim(-1.25, X + 1.25); a.set_ylim(-0.25, Z + 0.72); a.set_aspect("equal"); a.axis("off")


# (a) forward
scene(ax)
src = np.array([1.2, 0.45])
for ang in np.linspace(0, 2 * np.pi, 20, endpoint=False) + 0.13:
    ax.plot([src[0], src[0] + 0.32 * np.cos(ang)], [src[1], src[1] + 0.32 * np.sin(ang)], color=Sty.INK3, lw=0.6)
fwd = np.array([src, [2.45, Z], [2.75, Z + GT], [3.6, Z], [4.45, Z + GT], [5.3, Z + 0.01], [GX1, Z + 0.03]])
ax.plot(fwd[:, 0], fwd[:, 1], color=Sty.INK, lw=0.9)
ax.plot(*src, "o", color=Sty.COL["network"], ms=4, zorder=6)
ax.text(src[0] - 0.05, src[1] - 0.38, "emission point r", fontsize=6, ha="center")
ax.text(0, Z + 0.5, "a  Forward: photons from one emission point; about 1% reach the port (schematic)",
        fontsize=7, fontweight="bold", ha="left", va="bottom")

# (b) reverse, from the trace
scene(bx)
pm = bx.pcolormesh(xs, zs, np.ma.masked_equal(pv, 0).T, cmap=Sty.SEQ, norm=Normalize(0, pv.max()), alpha=0.85,
                   zorder=1.5, edgecolors="none")
bids = sorted({b for *_, b in SEGS if b >= 0 and _[-1] if True})
palette = [Sty.COL["adjoint"], Sty.COL["network"], "#7b3fa0", "#2a9d8f", "#b5179e", "#6c757d"]
order = []
for p0, p1, w, med, b in SEGS:
    if b not in order and b >= 0:
        order.append(b)
for p0, p1, w, med, b in SEGS:
    if med == "air":
        bx.plot([p0[0], p1[0]], [p0[1], p1[1]], color=Sty.INK3, lw=0.8, ls=(0, (2, 1.5)), zorder=3)
        bx.plot(*p1, "x", color=Sty.INK3, ms=3.5, mew=0.8, zorder=3)
        continue
    col = palette[order.index(b) % len(palette)]
    bx.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=0.5 + 2.4 * np.sqrt(w), solid_capstyle="round", zorder=3)
labelled = set()
prev_air = False
for p0, p1, w, med, b in SEGS:
    if med == "air":
        prev_air = True
        continue
    after_wall = prev_air and med == "scint"       # weak branch reflected at a bare wall
    prev_air = False
    if b in labelled and not after_wall:
        continue
    labelled.add(b)
    m = p0 + 0.3 * (p1 - p0) if med == "scint" else 0.5 * (p0 + p1)
    bx.text(m[0] + (0.12 if med == "scint" else 0), m[1] + (0.0 if med == "scint" else 0.07), f"w = {w:.3f}" if w < 0.995 else "w = 1", fontsize=5.6, ha="left" if med == "scint" else "center", va="bottom",
            color=palette[order.index(b) % len(palette)], fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.3), zorder=6)
for q, kind, txt in EVENTS:
    if kind == "split":
        bx.plot(*q, "o", ms=3, mfc="white", mec=Sty.INK, mew=0.7, zorder=5)
    elif kind == "air":
        bx.plot(*q, "o", ms=3, mfc="white", mec=Sty.INK, mew=0.7, zorder=5)
    elif kind == "cont":
        bx.annotate("further splits along the guide (not drawn)", q, xytext=(q[0] - 1.6, q[1] + 0.3), fontsize=5.4,
                    ha="center", va="center", color=Sty.INK2, arrowprops=dict(arrowstyle="-", lw=0.5, color=Sty.INK3))
air_events = [e for e in EVENTS if e[1] == "air"]
if air_events:
    q = air_events[0][0]
    bx.text(q[0] - 0.12, q[1] + 0.12, "5.4% reflected,\n94.6% into air (dropped)", fontsize=5.2, ha="right", va="bottom",
            color=Sty.INK2)
bx.text(0, Z + 0.5, "b  Reverse: one root from the port, traced with real Fresnel weights; every voxel crossed is scored",
        fontsize=7, fontweight="bold", ha="left", va="bottom")
cax = bx.inset_axes((X + 0.72, 0.0, 0.07, Z), transform=bx.transData)
cb = fig.colorbar(pm, cax=cax); cb.ax.tick_params(labelsize=5, length=2, pad=1)
cb.set_label("$p_v$ from one root", fontsize=5.8, labelpad=2)
Sty.save(fig, "fig_reciprocal_schematic")
out = dict(theta_deg=68.0, segments=[dict(p0=list(map(float, a)), p1=list(map(float, b)), w=float(w), medium=m, branch=int(k))
                                     for a, b, w, m, k in SEGS],
           events=[dict(at=list(map(float, q)), kind=k, text=t) for q, k, t in EVENTS], V=V, c=C_PORT, pv_max=float(pv.max()))
json.dump(out, open(C.ST / "reciprocal_trace.json", "w"), indent=1)
for s in out["segments"]:
    print(s["medium"], s["branch"], round(s["w"], 4), [round(v, 2) for v in s["p0"]], "->", [round(v, 2) for v in s["p1"]])
for e in out["events"]:
    print(e["kind"], [round(v, 2) for v in e["at"]], e["text"])
