"""Fig. 8: three-network ensemble against the 2e5-root adjoint reference for one test geometry (sobol_independent__021),
top and bottom voxel layers. Columns: reference A, ensemble S, (S - A)/A, P = (S - A)/sigma(z), C = (B - A)/R(z), where
R(z) is the RMS of A - B over the 400 voxels of a layer and sigma(z) = R(z)/sqrt(2).

The figure comes from a separate training run (new split, networks with seeds 62801-62803). Its data are released at
the level of the figure (data/derived/fig08/figure-data.h5: maps A and B, the three network predictions, their mean and
the derived maps); those networks and that bank are not part of this release.
usage: python fig08_ensemble_vs_reference.py [--tight]   (--tight: relative difference +-5%, normalized residual +-6)
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402
import argparse
import io
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

FIG = Path(os.environ.get("FIG_DIR", Path(__file__).resolve().parent))
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 18.5, "ytick.labelsize": 18.5,
                     "legend.fontsize": 18.5})
parser = argparse.ArgumentParser()
parser.add_argument("--tight", action="store_true")
args = parser.parse_args()

with h5py.File(paths.DATA / "derived" / "fig08" / "figure-data.h5", "r") as f:
    a = f["reference_A/field/p_per_emitted"][:]
    s = f["predictions/ensemble/p"][:]
    fields = [a, s, *[f["derived/" + k][:] for k in
                      ("relative_difference", "normalized_surrogate_residual", "normalized_replica_control")]]
    edges = [f["reference_A/" + axis + "_edges_cm"][:] for axis in "xyz"]
    guides = [f[f"reference_A/guide{k}_points_cm"][:] for k in (0, 1)]

rel = 0.05 if args.tight else 0.15
pn = 6 if args.tight else 35
fig, axes = plt.subplots(2, 5, figsize=(20, 9.7), layout="constrained")
fig.get_layout_engine().set(h_pad=0.18, w_pad=0.06)
titles = ["Adjoint 200k\nreference", "Surrogate\nensemble", "Surrogate\nrelative error", "Normalized\nsurrogate residual",
          "Normalized\n200k–200k control"]
images = []
for row, k in enumerate((7, 0)):                       # top layer, bottom layer
    for col, field in enumerate(fields):
        ax = axes[row, col]
        norm = LogNorm(0.008, 0.039) if col < 2 else Normalize(-rel, rel) if col == 2 else Normalize(-pn, pn)
        im = ax.pcolormesh(edges[0], edges[1], field[:, :, k].T, cmap="Blues" if col < 2 else "RdBu_r", norm=norm,
                           rasterized=True)
        images.append(im)
        for guide, color, ls in zip(guides, ["#282e39", "#187e69"], ["-", "--"]):
            ax.plot(guide[:, 0], guide[:, 1], color=color, ls=ls, lw=1.1)
        ax.set(xlim=(edges[0][0], edges[0][-1]), ylim=(edges[1][0], edges[1][-1]), aspect="equal", xticks=[], yticks=[])
        if row == 0:
            ax.set_title(titles[col], fontsize=20, pad=14)
        if col == 0:
            ax.set_ylabel(("Top" if k == 7 else "Bottom") + f" layer\nz = {(edges[2][k] + edges[2][k + 1]) / 2:.3f} cm")
        if col >= 2:                                   # relative: p-weighted RMS; normalized maps: plain RMS
            rms = (np.sqrt(np.average(field[:, :, k] ** 2, weights=a[:, :, k])) if col == 2
                   else np.sqrt(np.mean(field[:, :, k] ** 2)))
            label = f"RMS {100 * rms:.1f}%" if col == 2 else f"RMS {rms:.2f}"
            ax.text(0.03, 0.04, label, transform=ax.transAxes, fontsize=20,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
fig.colorbar(images[0], ax=axes[:, :2], orientation="horizontal", shrink=0.9, pad=0.03, extend="max",
             label="Detection probability per emitted photon")
cb = fig.colorbar(images[2], ax=axes[:, 2], orientation="horizontal", pad=0.03, extend="both",
                  label="Relative difference")
cb.set_ticks([-rel, 0, rel], labels=[f"−{100 * rel:g}%", "0%", f"+{100 * rel:g}%"])
fig.colorbar(images[3], ax=axes[:, 3:], orientation="horizontal", pad=0.03, extend="both", label="Normalized residual")
name = "Fig-8_tighter-scales" if args.tight else "Fig-8"
# The paper's PDF was saved after a PNG and an SVG of the same figure; each save re-runs the constrained layout, so
# both are drawn first (in memory) to reproduce the paper's layout exactly.
for fmt in ("png", "svg"):
    fig.savefig(io.BytesIO(), format=fmt, dpi=300, bbox_inches="tight", pad_inches=0.22)
fig.savefig(FIG / (name + ".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.22)
print("wrote", FIG / (name + ".pdf"))
