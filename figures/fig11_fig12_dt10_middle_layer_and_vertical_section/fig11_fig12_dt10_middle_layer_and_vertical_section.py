"""Extra views of the channel asymmetry dt10 = t10(ch1) - t10(ch0) (companions of fig10_early_time_quantiles).

  fig10b_t10_middle_layer : B_train_g0180, layer 5 of 8 (the layer just above mid-depth), reference / adjoint 20k / network
  fig10c_t10_cross_section: B_train_g0180, x-z section through the middle row of voxels (y index 10)
  fig10d_t10_xgeom        : geometry 00 (X_test_g0000): two straight guides crossing as an X; top, middle and bottom
                            layers and the x-z section, reference / adjoint 20k / network

Maps of B_train_g0180 come from t10_maps.npz (t10_study.py). For X_test_g0000: reference = 2e5-root adjoint map
(xgeom/ref), adjoint 20k = independent roots (xgeom/adj20k, val-split seeds), network = sur2_628_final.
usage: python t10_views.py [--xonly]
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import common as C
import style as Sty
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig10_dt10_top_bottom_layers"))
from fig10_dt10_top_bottom_layers import quantile, mirror, guide_labels, MODELS, S

GT = 0.10                     # guide thickness (cm): guide 0 on z in [Z, Z+GT], guide 1 on z in [-GT, 0]
LIM = 1.0
METH = (("ref", "Adjoint 200k (reference)"), ("adj", "Adjoint 20k"), ("net", "Network"))
XG = C.T / "xgeom"


def dt(A, m, k):
    return (A[f"t10_1_{m}"] - A[f"t10_0_{m}"])[k]


def show_layer(ax, d, z, X, Y, g0, g1):
    im = ax.imshow(d[:, :, z].T, origin="lower", extent=(0, X, 0, Y), cmap=Sty.DIV, vmin=-LIM, vmax=LIM,
                   interpolation="nearest")
    ax.plot(g0[:, 0], g0[:, 1], color=Sty.INK, lw=0.6); ax.plot(g1[:, 0], g1[:, 1], color=Sty.INK, lw=0.6, ls="--")
    ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    return im


def crossings(g, y0, W):
    """x-intervals where the guide band (centre line g, width W) crosses the line y = y0."""
    out = []
    for a, b in zip(g[:-1], g[1:]):
        if (a[1] - y0) * (b[1] - y0) <= 0 and a[1] != b[1]:
            f = (y0 - a[1]) / (b[1] - a[1]); x = a[0] + f * (b[0] - a[0])
            t = (b - a) / np.linalg.norm(b - a)
            hw = 0.5 * W / max(abs(t[1]), 0.2)
            out.append((x - hw, x + hw))
    return out


def show_xsec(ax, d, iy, X, Y, Z, g0, g1, W):
    im = ax.imshow(d[:, iy, :].T, origin="lower", extent=(0, X, 0, Z), cmap=Sty.DIV, vmin=-LIM, vmax=LIM,
                   interpolation="nearest")
    y0 = (iy + 0.5) * Y / 20
    for lo, hi in crossings(g0, y0, W):
        ax.add_patch(Rectangle((lo, Z), hi - lo, GT, fc=Sty.INK2, ec="none", clip_on=False))
    for lo, hi in crossings(g1, y0, W):
        ax.add_patch(Rectangle((lo, -GT), hi - lo, GT, fc="white", ec=Sty.INK2, lw=0.6, ls="--", clip_on=False))
    ax.set_xlim(0, X); ax.set_ylim(-GT, Z + GT); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot([0, X, X, 0, 0], [0, 0, Z, Z, 0], color=Sty.INK3, lw=0.5)
    return im


def guide_xsec_labels(ax, d, iy, X, Y, Z, g0, g1, W, fs=5.2):
    y0 = (iy + 0.5) * Y / 20
    c0 = crossings(g0, y0, W); c1 = crossings(g1, y0, W)
    if c0:
        ax.text(np.mean(c0[0]), Z + GT * 1.6, "guide 0", fontsize=fs, ha="center", va="bottom")
    if c1:
        ax.text(np.mean(c1[0]), -GT * 1.6, "guide 1", fontsize=fs, ha="center", va="top")


def colorbar(fig, im, gid):
    cb = fig.colorbar(im, ax=fig.axes, location="bottom", shrink=0.6, pad=0.03, aspect=35)
    cb.set_label(r"$\Delta t_{10} = t_{10,1} - t_{10,0}$ (ns), " + gid, fontsize=6.3)


def fig_middle(A, k, gid, X, Y, g0, g1, name, z=4):
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.75))
    for ax, (m, lab) in zip(axs, METH):
        im = show_layer(ax, dt(A, m, k), z, X, Y, g0, g1)
        ax.set_title(lab, fontsize=6.5, fontweight="normal", loc="center")
    guide_labels(axs[0], g0, g1)
    axs[0].set_ylabel(f"layer {z + 1} of 8 (middle)", fontsize=6.3)
    colorbar(fig, im, gid)
    Sty.save(fig, name)


def fig_xsec(A, k, gid, X, Y, Z, g0, g1, W, name, iy=10):
    fig, axs = plt.subplots(3, 1, figsize=(3.4, 3.3))
    for ax, (m, lab) in zip(axs, METH):
        im = show_xsec(ax, dt(A, m, k), iy, X, Y, Z, g0, g1, W)
        ax.set_title(lab, fontsize=6.3, fontweight="normal", loc="right", pad=7)
    guide_xsec_labels(axs[0], None, iy, X, Y, Z, g0, g1, W)
    colorbar(fig, im, gid)
    fig.suptitle(f"x-z section at y = {(iy + 0.5) * Y / 20:.2f} cm (voxel row {iy + 1} of 20)", fontsize=6.3,
                 fontweight="normal")
    Sty.save(fig, name)


def fig_x(A, k, gid, X, Y, Z, g0, g1, W, name, iy=10):
    fig = plt.figure(figsize=(7.1, 7.2))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.62])
    for col, (m, lab) in enumerate(METH):
        d = dt(A, m, k)
        for row, (z, rl) in enumerate(((7, "top layer"), (4, "layer 5 of 8 (middle)"), (0, "bottom layer"))):
            ax = fig.add_subplot(gs[row, col])
            im = show_layer(ax, d, z, X, Y, g0, g1)
            if row == 0:
                ax.set_title(lab, fontsize=6.5, fontweight="normal", loc="center")
            if col == 0:
                ax.set_ylabel(rl, fontsize=6.3)
                if row == 0:
                    guide_labels(ax, g0, g1)
            ax.axhline((iy + 0.5) * Y / 20, color=Sty.INK3, lw=0.4, ls=":")
        ax = fig.add_subplot(gs[3, col])
        show_xsec(ax, d, iy, X, Y, Z, g0, g1, W)
        if col == 0:
            guide_xsec_labels(ax, None, iy, X, Y, Z, g0, g1, W)
            ax.text(-0.04 * X, Z / 2, f"x-z section\ny = {(iy + 0.5) * Y / 20:.2f} cm", fontsize=5.8, ha="right",
                    va="center", rotation=90)
    colorbar(fig, im, gid + " (straight guides)")
    Sty.save(fig, name)


def x_maps():
    """t10 maps of X_test_g0000 for reference, independent 20k adjoint and the network."""
    gid = "X_test_g0000"
    ref_h5 = XG / "ref/stage6B_adjoint_geometry_atlas_numba.h5"
    adj_h5 = XG / "adj20k/stage6B_adjoint_geometry_atlas_numba.h5"
    with h5py.File(ref_h5) as f:
        g = f["geometries"][gid]; h = g["field/h_per_ns"][:]
        cell = g["cell_cm"][:]; g0 = g["guide0_points_cm"][:]; g1 = g["guide1_points_cm"][:]; W = float(g["design_vector"][3])
        p = g["field/p_per_emitted"][:]
    with h5py.File(adj_h5) as f:
        ha = f["geometries"][gid]["field/h_per_ns"][:]
    data = S.load_split_data(ref_h5, [gid])
    mdl, _ = S.load_model(MODELS[0], torch.device("cpu")); mdl.eval()
    with torch.no_grad():
        geom, nodes, uvs, *_ = S._batch_from_indices(data, np.arange(3200), torch.device("cpu"))
        o = mdl(geom, nodes, uvs)
        M0 = (mdl.timing_mass(o["mix0_logits"]).numpy() * o["p0"].numpy()[:, None]).reshape(20, 20, 8, 240)
        M1 = (mdl.timing_mass(o["mix1_logits"]).numpy() * o["p1"].numpy()[:, None]).reshape(20, 20, 8, 240)
        pn = o["p0"].numpy().reshape(20, 20, 8)
    A = {"t10_0_ref": quantile(h, 0.1)[None], "t10_1_ref": quantile(mirror(h), 0.1)[None],
         "t10_0_adj": quantile(ha, 0.1)[None], "t10_1_adj": quantile(mirror(ha), 0.1)[None],
         "t10_0_net": quantile(M0, 0.1)[None], "t10_1_net": quantile(M1, 0.1)[None]}
    w = p / p.sum()
    stats = {}
    for m in ("adj", "net"):
        e = (A[f"t10_1_{m}"] - A[f"t10_0_{m}"] - (A["t10_1_ref"] - A["t10_0_ref"]))[0]
        stats[m] = dict(dt10_rms_ns=float(np.sqrt(np.mean(e ** 2))), t10_rms_ns=float(np.sqrt(np.sum(w * (A[f"t10_0_{m}"][0] - A["t10_0_ref"][0]) ** 2))))
    stats["p_whole_ref"] = float(p.mean()); stats["p_whole_net"] = float(pn.mean())
    stats["p_err_net"] = float(np.sqrt(np.sum(w * ((pn - p) / p) ** 2)))
    print(gid, stats, flush=True)
    return A, gid, cell, g0, g1, W


if __name__ == "__main__":
    Sty.setup()
    if "--xonly" not in sys.argv:
        Z_ = dict(np.load(C.ST / "t10_maps.npz")); ids = list(Z_["ids"]); gid = "B_train_g0180"; k = ids.index(gid)
        tr = C.truth(); X, Y, Zc = tr["cell"][k]
        with h5py.File(C.TRUTH_H5) as f:
            g = f["geometries"][gid]; g0 = g["guide0_points_cm"][:]; g1 = g["guide1_points_cm"][:]; W = float(g["design_vector"][3])
        fig_middle(Z_, k, gid, X, Y, g0, g1, "fig10b_t10_middle_layer")
        fig_xsec(Z_, k, gid, X, Y, Zc, g0, g1, W, "fig10c_t10_cross_section")
    if (XG / "adj20k/stage6B_adjoint_geometry_atlas_numba.h5").exists():
        A, gid, cell, g0, g1, W = x_maps()
        fig_x(A, 0, gid, *cell, g0, g1, W, "fig10d_t10_xgeom")
