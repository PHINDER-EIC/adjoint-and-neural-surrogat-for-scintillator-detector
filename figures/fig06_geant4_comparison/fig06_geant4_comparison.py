"""Fig. 6 (figM1_validation_geant4) for the rewritten Sec. VI A: channel 0 only, network = three training seeds.

(a) channel-0 region values / Geant4: adjoint (40), forward engine JAX (20), network (40 per seed, all three seeds drawn);
    bars and rms labels from study/ch0_table3.json (same numbers as Table III).
(b, c) channel-0 arrival-time density in B_train_g0180, as before; the network curve is the mean of the three seeds.
Saves figM1_validation_geant4_ch0.{pdf,png}; the original figure is not touched.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.argv = sys.argv[:1]
import common as C  # noqa: E402
import regions as RG  # noqa: E402
import style as Sty  # noqa: E402

Sty.setup()
DBLW = 7.1
NETC, ADJC, JAXC, G4C = Sty.COL["network"], Sty.COL["adjoint"], Sty.COL["jax"], Sty.COL["g4"]
S = RG.S
PATHS = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
GID = "B_train_g0180"

R = json.load(open(C.ST / "regions.json"))["regions"]
T = json.load(open(C.ST / "ch0_table3.json"))
H = dict(np.load(C.ST / "regions_hist_single.npz"))
models = [S.load_model(p, torch.device("cpu"))[0].eval() for p in PATHS]

# network channel-0 region values for every seed, and the 3-seed mean density in the panel-b/c cell
net_r = []
net_hist = {}
for gid, regs in R.items():
    defs = json.loads(str(np.load(C.ST / "g4/geom" / f"{gid}.npz")["regions_json"]))
    for m in models:
        pn0, _, mn = RG.net_voxels(gid, [m])
        for name, (lo, hi) in defs.items():
            sl = C.region_slices(lo, hi)
            net_r.append(float(pn0[sl].mean()) / regs[name]["g4_unpol"]["p0"])
            if gid == GID:
                mreg = (mn[sl] * pn0[sl][..., None]).reshape(-1, 240).sum(0) / pn0[sl].sum()
                net_hist.setdefault(name, []).append(mreg * pn0[sl].mean() / 0.25)
net_r = np.array(net_r)

fig = plt.figure(figsize=(DBLW, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15])
ax = fig.add_subplot(gs[0])
for j, (m, col, mk) in enumerate((("adjoint", ADJC, "o"), ("jax", JAXC, "^"), ("network", NETC, "s"))):
    if m == "network":
        r = net_r
    else:
        r = np.array([row[m]["p0"] / row["g4_unpol"]["p0"] for regs in R.values() for row in regs.values() if m in row])
    x = j + (np.random.default_rng(j).random(len(r)) - 0.5) * 0.45
    ax.plot(x, r, mk, color=col, ms=2.4, alpha=0.6 if m != "network" else 0.45, mew=0)
    s = T[m]
    ax.errorbar(j, s["ratio"], yerr=s["ratio_se"], fmt="_", color=Sty.INK, ms=12, mew=1.2, elinewidth=1.2)
    ax.text(j, 1.118, f"{100*s['rms']:.1f}%", ha="center", fontsize=6.2, color=Sty.INK2)
ax.axhline(1, color=Sty.INK2, lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(["adjoint", "forward\n(JAX)", "network"], fontsize=6.5)
ax.set_xlim(-0.6, 2.6); ax.set_ylim(0.88, 1.13)
ax.set_ylabel("captured fraction / Geant4")
ax.grid(axis="x", visible=False)
Sty.panel(ax, "a", "Channel-0 regions (rms above)")

idx = np.concatenate([np.arange(0, 8), np.arange(8, 40, 4), np.arange(40, 241, 8)])
edges = np.arange(241) * 0.25
for k, (reg, title) in enumerate((("hotspot", "next to the guide"), ("centre", "cell center"))):
    ax = fig.add_subplot(gs[k + 1])
    curves = [("g4_unpol", G4C, 1.0, "Geant4", H[f"{GID}|{reg}|g4_unpol"]),
              ("jax", JAXC, 0.9, "forward (JAX)", H[f"{GID}|{reg}|jax"]),
              ("adjoint", ADJC, 1.1, "adjoint", H[f"{GID}|{reg}|adjoint"]),
              ("network", NETC, 1.1, "network", np.mean(net_hist[reg], axis=0))]
    for _, col, lw, lab, h in curves:
        hb = np.array([h[a:b].mean() for a, b in zip(idx[:-1], idx[1:])])
        ax.stairs(np.maximum(hb, 1e-9), edges[idx], color=col, lw=lw, label=lab)
    top = H[f"{GID}|{reg}|adjoint"].max()
    ax.set_yscale("log"); ax.set_xlim(0, 25); ax.set_ylim(top * 3e-4, top * 2)
    ax.set_xlabel("arrival time at readout (ns)")
    if k == 0:
        ax.set_ylabel(r"detected per emitted photon (ns$^{-1}$)")
        ax.legend(fontsize=5.8, loc="upper right")
    Sty.panel(ax, "bc"[k], f"Arrival times, {title}")
Sty.save(fig, "figM1_validation_geant4_ch0")
print("network points outside the axis:", int(np.sum((net_r < 0.88) | (net_r > 1.11))), "range", net_r.min(), net_r.max())
