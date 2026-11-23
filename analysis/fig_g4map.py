"""Figure: voxel-by-voxel comparison of the adjoint map with a Geant4 (unpolarised) forward map.

Geant4: uniform isotropic emission over the whole cell (runs tagged 'map'), each caught photon binned by its
emission voxel -> per-voxel captured fraction in channels 0 and 1 with binomial errors.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import common as C
import style as Sty

MODE = "unpol"


def load_map(gid):
    files = sorted((C.ST / "g4/runs").glob(f"{gid}__{MODE}__map__*.npz"))
    return [np.load(f) for f in files]


def counts(chunks):
    n = sum(c["emitted_vox"] for c in chunks)
    c0 = np.zeros(3200); c1 = np.zeros(3200)
    for c in chunks:
        np.add.at(c0, c["vox"][c["ch"] == 0], 1); np.add.at(c1, c["vox"][c["ch"] == 1], 1)
    return n.astype(float), c0, c1


def compare(gid):
    tr = C.truth()
    gi = tr["ids"].index(gid)
    pA = tr["p"][gi]; sA = tr["p_se"][gi] * C.ref_noise()["k_p"]   # measured, not self-reported, map noise
    pA1 = pA.reshape(20, 20, 8)[::-1, :, ::-1].ravel(); sA1 = sA.reshape(20, 20, 8)[::-1, :, ::-1].ravel()
    ch = load_map(gid)
    n, c0, c1 = counts(ch)
    out = dict(gid=gid, photons=int(n.sum()), chunks=len(ch), wall_core_s=float(sum(float(c["wall_s"]) for c in ch)))
    pulls = []
    for c, a, s, k in ((c0, pA, sA, 0), (c1, pA1, sA1, 1)):
        q = c / n
        se = np.sqrt(a * (1 - a) / n)          # expected binomial variance (earlier: max of observed and expected)
        pu = (q - a) / np.sqrt(se ** 2 + s ** 2)
        pulls.append(pu)
        w = a / a.sum()
        out[f"ch{k}"] = dict(ratio_total=float(c.sum() / (a * n).sum()), ratio_total_se=float(np.sqrt(c.sum()) / (a * n).sum()),
                             chi2=float(np.sum(pu ** 2)), ndf=3200, mean_pull=float(pu.mean()), sd_pull=float(pu.std()),
                             rms_rel=float(np.sqrt(np.sum(w * ((q - a) / a) ** 2))),
                             rms_rel_expected=float(np.sqrt(np.sum(w * (q * 0 + a * (1 - a) / n + s ** 2) / a ** 2))))
    # error vs photons: subsets of chunks
    sub = []
    for k in (1, 2, 5, 10, 20, 50, 100, 200, 400):
        if k > len(ch):
            break
        n_, c0_, _ = counts(ch[:k])
        q = c0_ / n_
        w = pA / pA.sum()
        meas2 = np.sum(w * ((q - pA) / pA) ** 2) - np.sum(w * (sA / pA) ** 2)
        pred = np.sqrt(3200 * (1 - pA).sum() / (n_.sum() * pA.sum()))
        sub.append(dict(photons=float(n_.sum()), measured=float(np.sqrt(max(meas2, 0))), predicted=float(pred)))
    out["subsets"] = sub
    return out, np.concatenate(pulls), c0 / n, pA, n


def plot(results):
    Sty.setup()
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.4))
    tr = C.truth()
    ax = axs[0, 0]
    for (res, pulls, *_), col in zip(results, (Sty.INK, Sty.INK3)):
        x = np.clip(pulls, -5.99, 5.99)
        ax.hist(x, bins=np.linspace(-6, 6, 49), density=True, histtype="step", color=col, lw=1.1,
                label=f"{res['gid']}: $\\chi^2$/ndf = {(res['ch0']['chi2'] + res['ch1']['chi2']) / 6400:.2f}")
    xx = np.linspace(-6, 6, 200)
    ax.plot(xx, stats.norm.pdf(xx), color=Sty.COL["adjoint"], ls="--", lw=0.9, label="N(0,1)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3)
    ax.set_xlabel("voxel pull, Geant4 - adjoint 200k (both channels)"); ax.set_ylabel("density")
    ax.legend(fontsize=5.6, loc="upper left")
    Sty.panel(ax, "a", "Per-voxel pulls")

    for j, ((res, pulls, q, pA, n), col) in enumerate(zip(results, (Sty.INK, Sty.INK3))):
        gi = tr["ids"].index(res["gid"])
        Q, A, N = q.reshape(20, 20, 8), pA.reshape(20, 20, 8), n.reshape(20, 20, 8)
        r = (Q * N).sum((0, 1)) / (A * N).sum((0, 1))
        e = np.sqrt((Q * N).sum((0, 1))) / (A * N).sum((0, 1))
        axs[0, 1].errorbar((np.arange(8) + 0.5) / 8, r, yerr=e, color=col, marker="o", ms=3, label=res["gid"])
    axs[0, 1].axhline(1, color=Sty.INK2, lw=0.6)
    axs[0, 1].set_xlabel("depth z / Z (guide 0 at z/Z = 1)"); axs[0, 1].set_ylabel("Geant4 / adjoint, channel 0")
    axs[0, 1].legend(fontsize=5.6)
    Sty.panel(axs[0, 1], "b", "Layer totals")

    ax = axs[0, 2]
    for (res, *_), col, mk in zip(results, (Sty.INK, Sty.INK3), "os"):
        s = res["subsets"]
        ax.plot([v["photons"] for v in s], [v["measured"] for v in s], mk, color=col, ms=3.5, label=f"{res['gid']} measured")
        ax.plot([v["photons"] for v in s], [v["predicted"] for v in s], "-", color=col, lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); Sty.logpct(ax, ticks=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0))
    ax.set_xlabel("Geant4 photons per map"); ax.set_ylabel("p-weighted rms error of the map")
    ax.legend(fontsize=5.6, title="line: binomial prediction", title_fontsize=5.6)
    Sty.panel(ax, "c", "Forward-map error vs photons")

    res, pulls, q, pA, n = results[0]
    gi = tr["ids"].index(res["gid"])
    X, Y, Z = tr["cell"][gi]
    Q, A = q.reshape(20, 20, 8), pA.reshape(20, 20, 8)
    for k, (z, lab) in enumerate(((7, "top layer"), (4, "mid depth"), (0, "bottom layer"))):
        ax = axs[1, k]
        # 2x2 rebinning for display (each map voxel has ~1/3200 of the photons)
        q2 = Q[:, :, z].reshape(10, 2, 10, 2).mean((1, 3)); a2 = A[:, :, z].reshape(10, 2, 10, 2).mean((1, 3))
        im = ax.imshow(np.log(q2 / a2).T, origin="lower", extent=(0, X, 0, Y), cmap=Sty.DIV, vmin=-0.3, vmax=0.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        Sty.panel(ax, "def"[k], f"{lab}" + (f"  ({res['gid']})" if k == 0 else ""))
    fig.colorbar(im, ax=axs[1, :], location="right", shrink=0.8, label="ln(Geant4 / adjoint)\n2x2-voxel blocks")
    Sty.save(fig, "fig8_geant4_voxel_maps")


if __name__ == "__main__":
    gids = sys.argv[1:] or ["B_train_g0180", "A_train_g0013"]
    R = [compare(g) for g in gids if load_map(g)]
    json.dump([r[0] for r in R], open(C.ST / "g4map_summary.json", "w"), indent=1)
    for r in R:
        print(json.dumps({k: v for k, v in r[0].items() if k != "subsets"}))
    plot(R)
