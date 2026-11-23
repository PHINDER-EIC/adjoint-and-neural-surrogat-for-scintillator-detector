"""Figure: channel-0 arrival-time distributions per emission region: Geant4, JAX, adjoint map, network."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty

GEOMS = sys.argv[1:] or ["A_train_g0013", "B_train_g0180"]
REGS = [("hotspot", "next to the guide"), ("centre", "cell centre"), ("far_corner", "far corner")]
EDGES = np.arange(241) * 0.25


IDX = np.concatenate([np.arange(0, 8), np.arange(8, 40, 4), np.arange(40, 241, 8)])


def rebin(h, k=None):
    """density on variable bins: 0.25 ns to 2 ns, 1 ns to 10 ns, 2 ns beyond."""
    out = np.array([h[a:b].mean() for a, b in zip(IDX[:-1], IDX[1:])])
    return out, EDGES[IDX]


def plot():
    Sty.setup()
    H = dict(np.load(C.ST / "regions_hist.npz"))
    R = json.load(open(C.ST / "regions.json"))["regions"]
    fig, axs = plt.subplots(len(GEOMS), 3, figsize=(7.1, 2.25 * len(GEOMS)), sharex=True)
    axs = np.atleast_2d(axs)
    tr = C.truth()
    for i, g in enumerate(GEOMS):
        gi = tr["ids"].index(g)
        X, Y, Z = tr["cell"][gi]
        for j, (reg, rlab) in enumerate(REGS):
            ax = axs[i, j]
            row = R[g][reg]
            for src, key, k in (("g4_unpol", "g4", 2), ("jax", "jax", 4)):
                hk = f"{g}|{reg}|{src}"
                if hk in H:
                    h, e = rebin(H[hk], k)
                    ax.stairs(np.maximum(h, 1e-9), e, color=Sty.COL[key], lw=1.0 if key == "g4" else 0.9,
                              label=f"{Sty.LAB[key]}" + (" (unpol.)" if key == "g4" else ""))
            for src in ("adjoint", "network"):
                h = H[f"{g}|{reg}|{src}"]
                hb, e = rebin(h)                                   # same variable bins as the forward histograms
                ax.stairs(np.maximum(hb, 1e-9), e, color=Sty.COL[src], lw=1.1,
                          label=Sty.LAB[src] + (" 200k" if src == "adjoint" else ""))
            ax.set_yscale("log"); ax.set_xlim(0, 30)
            top = max(H[f"{g}|{reg}|adjoint"].max(), 1e-6)
            ax.set_ylim(top * 3e-4, top * 2.5)
            txt = []
            for src, nm in (("g4_unpol", "G4"), ("jax", "JAX"), ("adjoint", "adj"), ("network", "net")):
                if src in row:
                    txt.append(f"{nm}: {row[src]['t_mean']:.2f} / {row[src]['t_sd60']:.2f}")
            ax.text(0.98, 0.97, "mean / s.d. (ns)\n" + "\n".join(txt), transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.6, color=Sty.INK2, family="monospace")
            if j == 0:
                ax.set_ylabel(f"{g}\n" + r"dP/dt per emitted photon (ns$^{-1}$)", fontsize=6.8)
            if i == len(GEOMS) - 1:
                ax.set_xlabel("arrival time at readout, channel 0 (ns)")
            Sty.panel(ax, "abcdefghi"[i * 3 + j], f"{rlab}  ({X:.1f}x{Y:.1f}x{Z:.2f} cm)" if j == 0 else rlab)
    axs[0, 0].legend(loc="lower left", fontsize=5.8)
    Sty.save(fig, "fig3_time_distributions")


if __name__ == "__main__":
    plot()
