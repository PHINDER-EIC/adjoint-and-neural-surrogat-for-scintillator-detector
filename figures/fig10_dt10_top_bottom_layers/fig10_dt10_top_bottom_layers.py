"""Early-arrival quantiles: t10 per voxel and channel, and the channel asymmetry dt10 = t10(ch1) - t10(ch0).

Reference: 200k-root maps (h_per_ns, 0-60 ns). Compared: 20k-root adjoint maps of the same 40 test geometries
(combined.h5, independent seeds) and the single network sur2_628_final (see MODELS; the 3-network ensemble result is in
t10_ensemble.json). Quantiles are taken of the conditional time density over 0-60 ns with linear interpolation
inside 0.25-ns bins. Writes study/t10.json, study/t10_maps.npz and figures/fig10_early_time_quantiles.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
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

EDGES = np.arange(241) * 0.25
MODELS = [C.T / "networks/network_seed20261003.pt"]  # single network (main text); the ensemble result is in t10_ensemble.json


def quantile(mass, q):
    """mass [..., 240] (any normalisation) -> time where the CDF reaches q."""
    c = np.cumsum(mass, -1)
    tot = c[..., -1:]
    c = c / np.where(tot > 0, tot, 1)
    i = np.argmax(c >= q, axis=-1)
    c_prev = np.where(i > 0, np.take_along_axis(c, np.maximum(i - 1, 0)[..., None], -1)[..., 0], 0.0)
    c_i = np.take_along_axis(c, i[..., None], -1)[..., 0]
    frac = np.clip((q - c_prev) / np.maximum(c_i - c_prev, 1e-12), 0, 1)
    return EDGES[i] + frac * 0.25


def mirror(a):  # (20,20,8,...) -> channel-1 view
    return a[::-1, :, ::-1, ...]


def main():
    tr = C.truth()
    ids = tr["ids"]
    data = S.load_split_data(C.TRUTH_H5, ids)
    models = []
    for m in MODELS:
        mdl, _ = S.load_model(m, torch.device("cpu")); mdl.eval(); models.append(mdl)
    out = {k: [] for k in ("t10_0_ref", "t10_1_ref", "t10_0_adj", "t10_1_adj", "t10_0_net", "t10_1_net", "t50_0_ref", "t50_0_net",
                           "t50_0_adj", "w")}
    with h5py.File(C.TRUTH_H5) as fr, h5py.File(C.T / "atlas_684_geometries_2e4roots.h5") as fa:
        for gi, gid in enumerate(ids):
            h = fr["geometries"][gid]["field/h_per_ns"][:]
            ha = fa["geometries"][gid]["field/h_per_ns"][:]
            V = np.arange(gi * 3200, (gi + 1) * 3200)
            with torch.no_grad():
                geom, nodes, uvs, *_ = S._batch_from_indices(data, V, torch.device("cpu"))
                M0 = M1 = 0
                for mdl in models:
                    o = mdl(geom, nodes, uvs)
                    M0 = M0 + mdl.timing_mass(o["mix0_logits"]).numpy() * o["p0"].numpy()[:, None]
                    M1 = M1 + mdl.timing_mass(o["mix1_logits"]).numpy() * o["p1"].numpy()[:, None]
            M0 = M0.reshape(20, 20, 8, 240); M1 = M1.reshape(20, 20, 8, 240)
            out["t10_0_ref"].append(quantile(h, 0.1)); out["t10_1_ref"].append(quantile(mirror(h), 0.1))
            out["t10_0_adj"].append(quantile(ha, 0.1)); out["t10_1_adj"].append(quantile(mirror(ha), 0.1))
            out["t10_0_net"].append(quantile(M0, 0.1)); out["t10_1_net"].append(quantile(M1, 0.1))
            out["t50_0_ref"].append(quantile(h, 0.5)); out["t50_0_net"].append(quantile(M0, 0.5)); out["t50_0_adj"].append(quantile(ha, 0.5))
            out["w"].append(tr["p"][gi].reshape(20, 20, 8))
    A = {k: np.array(v) for k, v in out.items()}
    np.savez_compressed(C.ST / "t10_maps.npz", **A, ids=np.array(ids))
    w = A["w"] / A["w"].sum()
    rel = (tr["p_se"] / tr["p"]).reshape(-1, 20, 20, 8)
    res = {}
    d_ref = A["t10_1_ref"] - A["t10_0_ref"]
    indep = np.array(["_test_" not in g for g in ids])   # 20k maps of originally-test geometries share roots with the reference
    for m in ("adj", "net"):
        keep = indep if m == "adj" else np.ones(len(ids), bool)
        A_ = A
        e0 = (A[f"t10_0_{m}"] - A["t10_0_ref"])[keep]
        e50 = (A[f"t50_0_{m}"] - A["t50_0_ref"])[keep]
        d = (A[f"t10_1_{m}"] - A[f"t10_0_{m}"])[keep]
        dr = d_ref[keep]
        ed = d - dr
        big = np.abs(dr) > 0.25
        wk = w[keep] / w[keep].sum()
        res[m] = dict(n_geometries=int(keep.sum()), t10_rms_ns=float(np.sqrt(np.sum(wk * e0 ** 2))), t10_bias_ns=float(np.sum(wk * e0)),
                      t50_rms_ns=float(np.sqrt(np.sum(wk * e50 ** 2))),
                      dt10_rms_ns=float(np.sqrt(np.mean(ed ** 2))), dt10_mad_ns=float(np.median(np.abs(ed))),
                      dt10_sign_agree=float(np.mean(np.sign(d[big]) == np.sign(dr[big]))), n_sign=int(big.sum()),
                      dt10_corr=float(np.corrcoef(d.ravel(), dr.ravel())[0, 1]))
    res["dt10_ref_sd_ns"] = float(d_ref.std())
    res["dt10_ref_range_ns"] = [float(np.quantile(d_ref, 0.01)), float(np.quantile(d_ref, 0.99))]
    json.dump(res, open(C.ST / "t10.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    plot(A, ids, res)


def guide_labels(ax, g0, g1, fs=5.4):
    """Label the guides at their start (inside the cell): guide 0 solid (top face, read out as channel 0),
    guide 1 dashed (bottom face, channel 1)."""
    box = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8)
    ax.text(g0[0, 0], g0[0, 1], "guide 0", fontsize=fs, ha="left", va="top", bbox=box)
    ax.text(g1[0, 0], g1[0, 1], "guide 1", fontsize=fs, ha="right", va="top", bbox=box)


def plot(A, ids, res):
    Sty.setup()
    tr = C.truth()
    fig = plt.figure(figsize=(7.1, 3.9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25])
    k = ids.index("B_train_g0180")
    X, Y, Z = tr["cell"][k]
    with h5py.File(C.TRUTH_H5) as f:
        g0 = f["geometries"][ids[k]]["guide0_points_cm"][:]; g1 = f["geometries"][ids[k]]["guide1_points_cm"][:]
    lim = 1.0
    for row, z in enumerate((7, 0)):
        for col, (m, lab) in enumerate((("ref", "Adjoint 200k (reference)"), ("adj", "Adjoint 20k"), ("net", "Network"))):
            ax = fig.add_subplot(gs[row, col])
            d = (A[f"t10_1_{m}"] - A[f"t10_0_{m}"])[k][:, :, z]
            im = ax.imshow(d.T, origin="lower", extent=(0, X, 0, Y), cmap=Sty.DIV, vmin=-lim, vmax=lim, interpolation="nearest")
            ax.plot(g0[:, 0], g0[:, 1], color=Sty.INK, lw=0.6); ax.plot(g1[:, 0], g1[:, 1], color=Sty.INK, lw=0.6, ls="--")
            ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if col == 0:
                guide_labels(ax, g0, g1)
            if row == 0:
                ax.set_title(lab, fontsize=6.5, fontweight="normal", loc="center")
            if col == 0:
                ax.set_ylabel("top layer (guide 0)" if z == 7 else "bottom layer (guide 1)", fontsize=6.3)
    cb = fig.colorbar(im, ax=fig.axes, location="bottom", shrink=0.5, pad=0.04, aspect=35)
    cb.set_label(r"$\Delta t_{10} = t_{10,1} - t_{10,0}$ (ns), " + ids[k], fontsize=6.3)
    ax = fig.add_subplot(gs[:, 3])
    dref = (A["t10_1_ref"] - A["t10_0_ref"]).ravel()
    for m, col, lab in (("adj", Sty.COL["adjoint"], "adjoint 20k"), ("net", Sty.COL["network"], "network")):
        d = (A[f"t10_1_{m}"] - A[f"t10_0_{m}"]).ravel()
        sel = np.random.default_rng(0).choice(len(d), 6000, replace=False)
        ax.plot(dref[sel], d[sel], ".", ms=1.2, color=col, alpha=0.35, rasterized=True)
        ax.plot([], [], "o", color=col, ms=3, label=f"{lab}: rms {res[m]['dt10_rms_ns']:.2f} ns, sign {100*res[m]['dt10_sign_agree']:.0f}%")
    ax.plot([-1.5, 1.5], [-1.5, 1.5], color=Sty.INK3, lw=0.7, ls="--")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_xlabel(r"reference $\Delta t_{10}$ (ns)"); ax.set_ylabel(r"predicted $\Delta t_{10}$ (ns)")
    ax.legend(fontsize=5.6, loc="upper left")
    Sty.panel(ax, "b", "All 40 test geometries")
    fig.text(0.005, 0.93, "a", fontsize=7.5, fontweight="bold")
    Sty.save(fig, "fig10_early_time_quantiles")


if __name__ == "__main__":
    if sys.argv[1:] == ["--plot"]:
        Z = dict(np.load(C.ST / "t10_maps.npz"))
        plot(Z, list(Z["ids"]), json.load(open(C.ST / "t10.json")))
    else:
        main()
