"""Surrogate derivatives for the designs of fd_study.py, and the derivative-fidelity figure.

Network: 3-model ensemble (628 training geometries, final weights) evaluated on the perturbed geometries written to
fd_atlas.h5; whole-volume p = voxel mean of p0, mean time = p-weighted voxel mean. Central differences with the same
step as the adjoint; the network is smooth, so this equals its autograd derivative to O(du^2) (checked on one case).
GP: PCA-48 + ARD Matern-5/2 on log p maps, refitted on the same 628 training geometries, evaluated at the perturbed
13-D design vectors.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

import common as C
import style as Sty

REPO = Path(str(paths.REPO))
sys.path.insert(0, str(REPO / "phinder_bridge/scripts"))
import phinder_stage6B_surrogate as S  # noqa: E402

FD = C.ST / "fd"
MODELS = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
PLAB = dict(cell_z="cell thickness", width="guide width", r_over_w="bend radius / width", turn2_x="turn-2 x", turn2_y="turn-2 y")
PMRK = dict(cell_z="o", width="s", r_over_w="D", turn2_x="^", turn2_y="v")


def network(rows):
    with h5py.File(FD / "fd_atlas.h5", "a") as f:
        for k, v in (("u_edges", np.linspace(0, 1, 21)), ("v_edges", np.linspace(0, 1, 21)), ("s_edges", np.linspace(0, 1, 9)),
                     ("time_edges_ns", np.arange(241) * 0.25)):
            if k not in f:
                f.create_dataset(k, data=v)
    models = [S.load_model(m, torch.device("cpu"))[0].eval() for m in MODELS]
    out = {}
    for key, r in rows.items():
        vals = {}
        for s in ("p", "m"):
            gid = f"{r['gid']}__{r['param']}__{s}"
            d = S.load_split_data(FD / "fd_atlas.h5", [gid])
            with torch.no_grad():
                geom, nodes, uvs, *_ = S._batch_from_indices(d, np.arange(d.n_vox), torch.device("cpu"))
                P, T = 0, 0
                for mdl in models:
                    o = mdl(geom, nodes, uvs)
                    P = P + o["p0"].numpy(); T = T + mdl.mean_time(o["mix0_logits"]).numpy() * o["p0"].numpy()
            vals[s] = (P.mean() / 3, (T / P).sum() / len(P) if False else float((T.sum() / P.sum())))
        out[key] = dict(dlnp=float((np.log(vals["p"][0]) - np.log(vals["m"][0])) / (2 * r["du"])),
                        dt=float((vals["p"][1] - vals["m"][1]) / (2 * r["du"])))
    return out


def gp(rows):
    sys.argv = sys.argv[:1]
    sys.path.insert(0, str(C.ST))
    import gp_study as G
    test, pool = G.pools()
    tr = pool[:628]
    P, M, _ = G.maps(G.R.ATLAS, tr)
    Xtr = np.array([G.R.spacefill[i] for i in tr])
    Xq, keys = [], []
    for key, r in rows.items():
        u0 = np.array(r["u0"])
        for sgn in (1, -1):
            u = u0.copy(); u[r["k"]] += sgn * r["du"]
            Xq.append(u); keys.append((key, sgn))
    (lp, mt), _, _ = G.gp_predict(Xtr, [np.log(np.maximum(P, 1e-6)), M], np.array(Xq))
    pv = np.exp(lp).mean(1)
    tv = (np.exp(lp) * mt).sum(1) / np.exp(lp).sum(1)
    out = {}
    for key, r in rows.items():
        ip = keys.index((key, 1)); im = keys.index((key, -1))
        out[key] = dict(dlnp=float((np.log(pv[ip]) - np.log(pv[im])) / (2 * r["du"])), dt=float((tv[ip] - tv[im]) / (2 * r["du"])))
    return out


def plot(rows, net, gpd):
    Sty.setup()
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    summ = {}
    for ax, q, qse, lab, unit in ((axs[0], "dlnp", "dlnp_se", r"$\partial \ln \bar p / \partial u$", ""),
                                  (axs[1], "dt", "dt_se", r"$\partial \bar t / \partial u$", " (ns)")):
        allv = []
        for key, r in rows.items():
            x, xe = r[q], r[qse]
            for src, D, col, off in (("network", net, Sty.COL["network"], 0), ("gp", gpd, Sty.COL["gp"], 0)):
                if D and key in D:
                    ax.errorbar(x, D[key][q], xerr=xe, fmt=PMRK[r["param"]], color=col, ms=3.2, elinewidth=0.6,
                                mfc=col if src == "network" else "white", mew=0.8)
                    allv += [x, D[key][q]]
        lim = max(abs(np.array(allv))) * 1.1 if allv else 1
        ax.plot([-lim, lim], [-lim, lim], color=Sty.INK3, lw=0.7, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.axhline(0, color=Sty.GRID, lw=0.6, zorder=0); ax.axvline(0, color=Sty.GRID, lw=0.6, zorder=0)
        ax.set_xlabel(f"adjoint finite difference{unit}"); ax.set_ylabel(f"surrogate{unit}")
        Sty.panel(ax, "ab"[0 if q == "dlnp" else 1], lab)
        for src, D in (("network", net), ("gp", gpd)):
            if not D:
                continue
            a = np.array([rows[k][q] for k in rows if k in D]); b = np.array([D[k][q] for k in rows if k in D])
            e = np.array([rows[k][qse] for k in rows if k in D])
            sig = np.abs(a) > 2 * e
            summ[f"{src}_{q}"] = dict(n=int(len(a)), corr=float(np.corrcoef(a, b)[0, 1]), rms_diff=float(np.sqrt(np.mean((a - b) ** 2))),
                                      rms_adjoint=float(np.sqrt(np.mean(a ** 2))), median_adjoint_se=float(np.median(e)),
                                      sign_agree_resolved=float(np.mean(np.sign(a[sig]) == np.sign(b[sig]))) if sig.any() else None,
                                      n_resolved=int(sig.sum()),
                                      chi2=float(np.sum(((a - b) / np.maximum(e, 1e-9)) ** 2)))
    ax = axs[2]
    params = list(PLAB)
    for j, p in enumerate(params):
        ks = [k for k in rows if rows[k]["param"] == p]
        if not ks:
            continue
        a = np.array([rows[k]["dlnp"] for k in ks])
        for src, D, col, dx in (("network", net, Sty.COL["network"], -0.13), ("gp", gpd, Sty.COL["gp"], 0.13)):
            if D:
                d = np.array([D[k]["dlnp"] for k in ks])
                ax.bar(j + dx, np.sqrt(np.mean((d - a) ** 2)), 0.24, color=col)
        ax.plot([j - 0.3, j + 0.3], [np.sqrt(np.mean(a ** 2))] * 2, color=Sty.COL["adjoint"], lw=1.4)
        ax.plot([j - 0.3, j + 0.3], [np.sqrt(np.mean(np.array([rows[k]["dlnp_se"] for k in ks]) ** 2))] * 2, color=Sty.COL["adjoint"],
                lw=0.9, ls=":")
    ax.set_xticks(range(len(params))); ax.set_xticklabels([PLAB[p].replace(" / ", "/\n").replace(" ", "\n", 1) for p in params], fontsize=5.6)
    ax.set_ylabel(r"rms of $\partial \ln \bar p / \partial u$")
    ax.grid(axis="x", visible=False)
    ax.plot([], [], color=Sty.COL["adjoint"], lw=1.4, label="adjoint derivative (size)")
    ax.plot([], [], color=Sty.COL["adjoint"], lw=0.9, ls=":", label="adjoint FD s.e.")
    ax.bar([0], [0], color=Sty.COL["network"], label="network - adjoint")
    ax.bar([0], [0], color=Sty.COL["gp"], label="GP - adjoint")
    ax.legend(fontsize=5.4, loc="upper right")
    Sty.panel(ax, "c", "Error per parameter")
    for p in params:
        axs[0].plot([], [], PMRK[p], color=Sty.INK2, mfc="none", ms=3.5, label=PLAB[p])
    axs[0].plot([], [], "o", color=Sty.COL["network"], ms=3.5, label="network (filled)")
    axs[0].plot([], [], "o", color=Sty.COL["gp"], mfc="white", ms=3.5, label="GP (open)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=7, fontsize=5.8)
    Sty.save(fig, "fig9_derivative_fidelity")
    return summ


if __name__ == "__main__":
    rows = {k: v for k, v in json.load(open(FD / "adjoint.json")).items() if v.get("ok")}
    cache = FD / "surrogate.json"
    if cache.exists() and "--refit" not in sys.argv:
        S_ = json.load(open(cache)); net, gpd = S_["network"], S_["gp"]
        net = {k: v for k, v in net.items() if k in rows}; gpd = {k: v for k, v in gpd.items() if k in rows}
        missing = [k for k in rows if k not in net]
        if missing:
            net.update(network({k: rows[k] for k in missing})); gpd.update(gp({k: rows[k] for k in missing}))
    else:
        net = network(rows); gpd = gp(rows)
    json.dump(dict(network=net, gp=gpd), open(cache, "w"), indent=1)
    summ = plot(rows, net, gpd)
    json.dump(summ, open(FD / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
