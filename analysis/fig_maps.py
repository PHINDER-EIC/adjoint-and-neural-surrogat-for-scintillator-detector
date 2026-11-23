"""Figure: detection-probability maps and residual maps of each method for one test geometry."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import common as C
import style as Sty

LAYERS = [(7, "top layer (guide side)"), (4, "mid depth"), (0, "bottom layer")]


def plot(gid=None):
    Sty.setup()
    ids, gi, M = C.methods()
    tr = C.sub_truth(gi)
    net = "net_628_ens" if "net_628_ens" in M else "net_628"
    if gid is None:   # the geometry with median network error
        r = np.log(M[net]["p0"] / tr["p"])
        w = tr["p"] / tr["p"].sum(1, keepdims=True)
        e = np.sqrt((w * r ** 2).sum(1))
        gid = ids[int(np.argsort(e)[len(e) // 2])]
    k = ids.index(gid)
    X, Y, Z = tr["cell"][k]
    with h5py.File(C.TRUTH_H5) as f:
        g0 = f["geometries"][gid]["guide0_points_cm"][:]
        g1 = f["geometries"][gid]["guide1_points_cm"][:]
    nlab = "Network ensemble\n(3 nets, 628 geom.)" if net.endswith("ens") else "Network surrogate\n(628 geometries)"
    cols = [("truth", "Adjoint 200k roots\n(reference)"), (net, nlab),
            ("res_" + net, "Network residual"), ("res_gp_628", "GP residual\n(628 geometries)"),
            ("res_adj_1k", "Adjoint 1k roots\nresidual"), ("res_adj_20k", "Adjoint 20k roots\nresidual")]
    cols = [c for c in cols if c[0] == "truth" or c[0].replace("res_", "") in M]
    fig, axs = plt.subplots(len(LAYERS), len(cols), figsize=(7.1, 1.18 * len(LAYERS) + 0.8))
    pt = tr["p"][k].reshape(20, 20, 8)
    vmin, vmax = np.quantile(pt, 0.01), pt.max()
    lnorm = LogNorm(vmin, vmax)
    dnorm = Normalize(-0.2, 0.2)
    ims = {}
    for i, (z, zl) in enumerate(LAYERS):
        for j, (key, lab) in enumerate(cols):
            ax = axs[i, j]
            if key == "truth":
                img, cmap, norm = pt[:, :, z], Sty.SEQ, lnorm
            elif not key.startswith("res_"):
                img, cmap, norm = M[key]["p0"][k].reshape(20, 20, 8)[:, :, z], Sty.SEQ, lnorm
            else:
                pm = M[key[4:]]["p0"][k].reshape(20, 20, 8)
                img, cmap, norm = np.log(pm[:, :, z] / pt[:, :, z]), Sty.DIV, dnorm
            ims["res" if key.startswith("res_") else "p"] = ax.imshow(img.T, origin="lower", extent=(0, X, 0, Y),
                                                                    cmap=cmap, norm=norm, interpolation="nearest")
            if z == 7:
                ax.plot(g0[:, 0], g0[:, 1], color=Sty.INK, lw=0.6, alpha=0.7)
            if z == 0:
                ax.plot(g1[:, 0], g1[:, 1], color=Sty.INK, lw=0.6, alpha=0.7, ls="--")
            ax.set_xlim(0, X); ax.set_ylim(0, Y)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(Sty.GRID)
            if i == 0:
                ax.set_title(lab, fontsize=6.3, fontweight="normal", loc="center")
            if j == 0:
                ax.set_ylabel(zl, fontsize=6.3)
            if key.startswith("res_"):
                pm = M[key[4:]]["p0"][k]
                r = np.log(pm / tr["p"][k]); w = tr["p"][k] / tr["p"][k].sum()
                zz = np.arange(3200) % 8 == z
                ax.text(0.03, 0.04, f"rms {100*np.sqrt(np.sum(w[zz]*r[zz]**2)/w[zz].sum()):.1f}%", transform=ax.transAxes,
                        fontsize=5.3, color=Sty.INK, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
    c1 = fig.colorbar(ims["p"], ax=axs[:, :2], location="bottom", shrink=0.8, pad=0.03, aspect=30)
    c1.set_label("detection probability per emitted photon, channel 0", fontsize=6.3)
    c2 = fig.colorbar(ims["res"], ax=axs[:, 2:], location="bottom", shrink=0.6, pad=0.03, aspect=40)
    c2.set_label("ln(method / reference)", fontsize=6.3)
    c2.set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
    fig.suptitle(f"Test geometry {gid}: {X:.2f} x {Y:.2f} x {Z:.2f} cm cell, guide width {tr['width'][k]:.2f} cm",
                 fontsize=7, x=0.01, ha="left")
    Sty.save(fig, "fig4_maps_residuals")
    return gid


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else None)
