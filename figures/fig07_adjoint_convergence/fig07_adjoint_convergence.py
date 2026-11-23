"""Fig. 7 (figM2_adjoint_convergence) on all 80 test geometries: the original 40 (study/adj/) and the 40 new designs
(study/adj_new/, from adj_conv_new.py), each scored against its own 2e5-root reference, pooled per (N, replicate).
Metrics as in fig_adjoint.convergence (C.pmetrics / C.tmetrics, reference noise subtracted).
Panel (b) unchanged (bench.json). Writes study/adj_convergence_80.json and figM2_adjoint_convergence_80.{pdf,png};
the original figure and adj_convergence.json are not touched. Also checks the replicate noise of the new designs
against ref_noise.json (which was measured on the original 40).
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json

import h5py
import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty

Sty.setup()
COLW = 3.4
ADJC = Sty.COL["adjoint"]
BLIND_H5 = C.T / "reference_blind40_2e5roots.h5"
NS = [250, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def truth_new():
    d = {k: [] for k in ("p", "p_se", "mt", "tsd", "pvol", "tvol")}
    with h5py.File(BLIND_H5) as f:
        ids = sorted(g for g in f["geometries"] if g.startswith("test_"))
        for g in ids:
            G = f["geometries"][g]; fl = G["field"]
            d["p"].append(fl["p_per_emitted"][:].ravel()); d["p_se"].append(fl["p_se"][:].ravel())
            d["mt"].append(fl["mean_transport_time_ns"][:].ravel())
            d["tsd"].append(C.tsd_from_h(fl["h_per_ns"][:])[0].ravel())
            d["pvol"].append(float(G.attrs["volume_mean_p"])); d["tvol"].append(float(G.attrs["volume_mean_transport_time_ns"]))
    out = {k: np.array(v) for k, v in d.items()}
    out["ids"] = ids
    return out


def runs(folder, ids):
    return {g: dict(np.load(C.ST / folder / f"{g}.npz")) for g in ids if (C.ST / folder / f"{g}.npz").exists()}


def pick(A, ids, N, r, key):
    out = []
    for g in ids:
        i = np.where((A[g]["N"] == N) & (A[g]["steps"] == 5000) & (A[g]["rep"] == r))[0][0]
        out.append(A[g][key][i].ravel() if A[g][key].ndim > 1 else A[g][key][i])
    return np.array(out)


def score(sets, TN, N, r):
    """sets: list of (truth dict, runs dict). Pooled metrics for one (N, replicate)."""
    tr = {k: np.concatenate([s[0][k] for s in sets]) for k in ("p", "p_se", "mt", "tsd", "pvol", "tvol")}
    get = lambda key: np.concatenate([pick(s[1], s[0]["ids"], N, r, key) for s in sets])
    pp, pt, ps = get("p"), get("mt"), get("tsd")
    return dict(p=C.pmetrics(pp, tr)["rms"], t=C.tmetrics(pt, "mt", tr, TN["mt"])["rms"],
                sd=C.tmetrics(ps, "tsd", tr, TN["tsd"])["rms"])


def main():
    TN = C.truth_time_noise()
    t_old = C.truth(); A_old = runs("adj", t_old["ids"])
    t_new = truth_new(); A_new = runs("adj_new", t_new["ids"])
    assert len(A_old) == 40 and len(A_new) == 40, (len(A_old), len(A_new))
    sets = {"original40": [(t_old, A_old)], "new40": [(t_new, A_new)], "all80": [(t_old, A_old), (t_new, A_new)]}
    rows = []
    for N in NS:
        for r in range(1 if N >= 50000 else 3):
            rows.append(dict(N=N, rep=r, **{k: score(s, TN, N, r) for k, s in sets.items()}))
    # replicate noise of the new designs (no reference involved), compare with the original 40
    noise = {}
    for name, (tr, A) in (("original40", (t_old, A_old)), ("new40", (t_new, A_new))):
        for N in (1000, 5000, 20000):
            v = []
            for gi, g in enumerate(tr["ids"]):
                w = tr["p"][gi] / tr["p"][gi].sum()
                P = [pick(A, [g], N, r, "p")[0] for r in range(3)]
                v += [np.sum(w * np.log(P[a] / P[b]) ** 2) / 2 for a in range(3) for b in range(a + 1, 3)]
            noise.setdefault(name, {})[N] = float(np.sqrt(np.mean(v)))
    json.dump(dict(rows=rows, replicate_noise_p=noise), open(C.ST / "adj_convergence_80.json", "w"), indent=1)
    print("N       p: orig  new   all80 |  tbar: orig  new  all80 |  s_t: all80")
    for N in NS:
        m = lambda s, k: 100 * np.mean([x[s][k] for x in rows if x["N"] == N])
        print(f"{N:6d}  {m('original40','p'):5.2f} {m('new40','p'):5.2f} {m('all80','p'):5.2f} | "
              f"{m('original40','t'):5.2f} {m('new40','t'):5.2f} {m('all80','t'):5.2f} | {m('all80','sd'):5.2f}")
    print("replicate noise of p (no reference):", noise)

    # figure: same layout as paper_figs.m2, panel (a) on the 80 geometries
    B = json.load(open(C.ST / "bench.json"))
    fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.85))
    ax = axs[0]
    for key, lab, col, mk in (("p", r"$p$", ADJC, "o"), ("t", r"$\bar t$", Sty.BLUE[8], "s")):
        m = np.array([np.mean([x["all80"][key] for x in rows if x["N"] == N]) for N in NS])
        ax.plot(NS, m, marker=mk, color=col, ms=2.8, lw=1.1, label=lab)
    xx = np.geomspace(250, 2e5, 20)
    a = np.mean([x["all80"]["p"] for x in rows if x["N"] == 20000]) * np.sqrt(20000)
    ax.plot(xx, a / np.sqrt(xx), color=Sty.INK3, ls="--", lw=0.7, zorder=0)
    ax.text(7e4, a / np.sqrt(7e4) * 1.8, r"$\propto N^{-1/2}$", fontsize=6, color=Sty.INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); Sty.logpct(ax, ticks=(0.01, 0.02, 0.05, 0.1, 0.2))
    ax.set_xlabel("roots $N$"); ax.set_ylabel("voxel error"); ax.legend(fontsize=6, loc="lower left")
    Sty.panel(ax, "a")
    ax = axs[1]
    b = B["adjoint"]
    for gname, col, mk in zip(b["geometries"], (Sty.BLUE[3], Sty.BLUE[6], Sty.BLUE[9]), "os^"):
        ax.plot(b["N"], b["wall_s"][gname], marker=mk, color=col, ms=2.8, lw=1.0, label=f"{b['n_volumes'][gname]} volumes")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("roots $N$"); ax.set_ylabel("wall time per map (s)")
    ax.legend(fontsize=5.8, loc="upper left")
    Sty.panel(ax, "b")
    Sty.save(fig, "figM2_adjoint_convergence_80")


if __name__ == "__main__":
    main()
