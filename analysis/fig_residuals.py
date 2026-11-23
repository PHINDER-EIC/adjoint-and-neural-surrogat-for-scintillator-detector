"""Figure: residual statistics of every method against the 200k-root reference maps (40 test geometries)."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty

ORDER = [("adj_1k", "Adjoint 1k", Sty.BLUE[1], "o"), ("adj_5k", "Adjoint 5k", Sty.BLUE[4], "o"),
         ("adj_20k", "Adjoint 20k", Sty.BLUE[8], "o"), ("gp_628", "GP (628)", Sty.COL["gp"], "D"),
         ("net_628", "Network (628)", Sty.COL["network"], "s"), ("net_628_ens", "Network ens. (628)", Sty.ORANGE[5], "s")]


def guide_distance(ids):
    """lateral distance (cm) of every voxel centre to the nearest guide-0 polyline point, [G, 3200]."""
    tr = C.truth()
    out = []
    with h5py.File(C.TRUTH_H5) as f:
        for g in ids:
            k = tr["ids"].index(g)
            X, Y, Z = tr["cell"][k]
            gp = f["geometries"][g]["guide0_points_cm"][:, :2]
            # densify the polyline
            seg = np.concatenate([np.linspace(gp[i], gp[i + 1], 20) for i in range(len(gp) - 1)])
            xc = (np.arange(20) + 0.5) * X / 20; yc = (np.arange(20) + 0.5) * Y / 20
            P = np.stack(np.meshgrid(xc, yc, indexing="ij"), -1).reshape(-1, 2)
            d = np.min(np.linalg.norm(P[:, None] - seg[None], axis=-1), 1).reshape(20, 20)
            out.append(np.repeat(d[:, :, None], 8, axis=2).ravel())
    return np.array(out)


def stats_table(ids, gi, M):
    tr = C.sub_truth(gi)
    T = {}
    for name in M:
        pm = C.pmetrics(M[name]["p0"], tr)
        T[name] = dict(p=pm, t=C.tmetrics(M[name]["mt"], "mt", tr, C.truth_time_noise()["mt"]), sd=C.tmetrics(M[name]["tsd"], "tsd", tr, C.truth_time_noise()["tsd"]),
                       geo_t=dict(rms=float(np.sqrt(np.mean(C.geo_time(M[name]["p0"], M[name]["mt"], tr) ** 2))),
                                  bias=float(np.mean(C.geo_time(M[name]["p0"], M[name]["mt"], tr)))))
    return T


def plot():
    Sty.setup()
    ids, gi, M = C.methods()
    tr = C.sub_truth(gi)
    order = [o for o in ORDER if o[0] in M]
    T = stats_table(ids, gi, M)
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.9))
    w = tr["p"] / tr["p"].sum()

    ax = axs[0, 0]
    bins = np.linspace(-0.4, 0.4, 81)
    for name, lab, col, mk in order:
        if name == "net_628_ens":
            continue
        r = np.log(M[name]["p0"] / tr["p"])
        ax.hist(np.clip(r, -0.399, 0.399).ravel(), bins=bins, weights=w.ravel(), density=True, histtype="step",
                color=col, lw=1.1, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 40)
    ax.set_xlabel("voxel residual  ln(p / p$_{200k}$)"); ax.set_ylabel("p-weighted density")
    Sty.panel(ax, "a", "Voxel p residuals")

    for ax, key, letter, title in ((axs[0, 1], "p", "b", "Whole-volume p per geometry"),
                                   (axs[0, 2], "t", "c", "Whole-volume mean time")):
        for j, (name, lab, col, mk) in enumerate(order):
            if key == "p":
                v = M[name]["p0"].mean(1) / tr["p"].mean(1) - 1
            else:
                v = C.geo_time(M[name]["p0"], M[name]["mt"], tr)
            jit = (np.random.default_rng(j).random(len(v)) - 0.5) * 0.35
            ax.plot(j + jit, v, mk, color=col, ms=2.6, alpha=0.75, mew=0)
            rms = np.sqrt(np.mean(v ** 2))
            ax.errorbar(j + 0.33, np.mean(v), yerr=np.std(v, ddof=1), fmt="_", color=Sty.INK, ms=6, elinewidth=0.8)
            ax.text(j, ax.get_ylim()[1] if False else 0, "", fontsize=1)
            ax.annotate(f"{100*rms:.1f}%", xy=(j, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=5.6, color=Sty.INK2)
        ax.axhline(0, color=Sty.INK2, lw=0.6)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([o[1] for o in order], rotation=40, ha="right", fontsize=6)
        ax.set_ylabel("relative error vs 200k"); Sty.pct(ax)
        lim = max(abs(np.array(ax.get_ylim()))) * 1.15
        ax.set_ylim(-lim, lim)
        Sty.panel(ax, letter, title)
        ax.grid(axis="x", visible=False)

    ax = axs[1, 0]
    z = np.arange(3200) % 8
    for name, lab, col, mk in order:
        if name == "net_628_ens":
            continue
        r = np.log(M[name]["p0"] / tr["p"])
        noise = (tr["p_se"] / tr["p"]) ** 2
        e = [np.sqrt(max(np.sum((w * r ** 2)[:, z == k]) / np.sum(w[:, z == k]) - np.sum((w * noise)[:, z == k]) / np.sum(w[:, z == k]), 0))
             for k in range(8)]
        ax.plot((np.arange(8) + 0.5) / 8, e, marker=mk, color=col, ms=3, label=lab)
    ax.set_xlabel("depth from bottom face  z / Z  (guide 0 at z/Z = 1)"); ax.set_ylabel("p-weighted rms error")
    ax.set_yscale("log"); ax.set_ylim(0.018, 0.15); Sty.logpct(ax, ticks=(0.02, 0.03, 0.05, 0.1))
    Sty.panel(ax, "d", "Error vs depth")

    ax = axs[1, 1]
    d = guide_distance(ids)
    db = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 5])
    for name, lab, col, mk in order:
        if name == "net_628_ens":
            continue
        r = np.log(M[name]["p0"] / tr["p"])
        noise = (tr["p_se"] / tr["p"]) ** 2
        e = []
        for a, b in zip(db[:-1], db[1:]):
            m = (d >= a) & (d < b)
            e.append(np.sqrt(max(np.sum((w * r ** 2)[m]) / np.sum(w[m]) - np.sum((w * noise)[m]) / np.sum(w[m]), 0)))
        ax.plot(0.5 * (db[:-1] + db[1:]), e, marker=mk, color=col, ms=3, label=lab)
    ax.set_xlabel("lateral distance to guide 0 (cm)"); ax.set_ylabel("p-weighted rms error")
    ax.set_yscale("log"); ax.set_ylim(0.018, 0.15); Sty.logpct(ax, ticks=(0.02, 0.03, 0.05, 0.1))
    Sty.panel(ax, "e", "Error vs distance to guide")

    ax = axs[1, 2]
    comps = [("p", "rms", "p"), ("p", "offset_rms", "p\noffset"), ("p", "shape_rms", "p\nshape"),
             ("t", "rms", "mean\ntime"), ("sd", "rms", "time\ns.d."), ("geo_t", "rms", "total\nmean t")]
    width = 0.8 / len(order)
    for j, (name, lab, col, mk) in enumerate(order):
        vals = [T[name][a][b] for a, b, _ in comps]
        ax.bar(np.arange(len(comps)) + (j - (len(order) - 1) / 2) * width, vals, width * 0.88, color=col, label=lab)
    ax.set_xticks(range(len(comps))); ax.set_xticklabels([c[2] for c in comps], fontsize=5.8)
    ax.set_yscale("log"); Sty.logpct(ax); ax.set_ylabel("rms error vs 200k (noise-subtracted)")
    ax.grid(axis="x", visible=False)
    Sty.panel(ax, "f", "Error components")
    h, l = axs[1, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(order), fontsize=6.5)
    Sty.save(fig, "fig5_residual_statistics")
    return T


if __name__ == "__main__":
    T = plot()
    json.dump(T, open(C.ST / "residual_stats.json", "w"), indent=1)
    for k, v in T.items():
        print(f"{k:12} p {100*v['p']['rms']:.2f}% (bias {100*v['p']['bias']:+.2f}%, offset {100*v['p']['offset_rms']:.2f}%, shape "
              f"{100*v['p']['shape_rms']:.2f}%, geo {100*v['p']['geo_rel_rms']:.2f}%) | t {100*v['t']['rms']:.2f}% | sd {100*v['sd']['rms']:.2f}%")
