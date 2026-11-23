"""Figure: adjoint (numba reverse tracer) convergence with root count, vs the 200k-root maps."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty

NS = [250, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def subset(tr, gids):
    gi = [tr["ids"].index(g) for g in gids]
    return {k: (v[gi] if isinstance(v, np.ndarray) and v.shape[:1] == (40,) else v) for k, v in tr.items()}


def convergence(TN=None):
    TN = TN or dict(mt=0.0, tsd=0.0)
    tr = C.truth()
    A = C.adj_runs()
    gids = list(A)
    sub = subset(tr, gids)
    w = sub["p"] / sub["p"].sum()
    rows = []
    for steps in (5000, 750):
        for N in NS:
            idx = [np.where((A[g]["N"] == N) & (A[g]["steps"] == steps))[0] for g in gids]
            if any(len(i) == 0 for i in idx):
                continue
            for r in range(min(len(i) for i in idx)):
                pick = lambda k: np.array([A[g][k][i[r]].ravel() for g, i in zip(gids, idx)])
                pp, pt, ps, pse = pick("p"), pick("mt"), pick("tsd"), pick("p_se")
                wall = np.array([A[g]["wall"][i[r]] for g, i in zip(gids, idx)])
                pv = np.array([A[g]["pvol"][i[r]] for g, i in zip(gids, idx)])
                tv = np.array([A[g]["tvol"][i[r]] for g, i in zip(gids, idx)])
                m = C.pmetrics(pp, sub)
                pred = float(np.sqrt(np.sum(w * (pse / np.maximum(pp, 1e-12)) ** 2)))
                rows.append(dict(N=N, steps=steps, rep=r, p=m, pred_relse=pred,
                                 t=C.tmetrics(pt, "mt", sub, TN["mt"]), sd=C.tmetrics(ps, "tsd", sub, TN["tsd"]),
                                 pvol_rel=(pv / sub["pvol"] - 1).tolist(), tvol_rel=(tv / sub["tvol"] - 1).tolist(),
                                 wall_mean=float(wall.mean()), wall=wall.tolist()))
    return dict(n_geometries=len(gids), rows=rows)


def pulls(N, rep=0):
    tr = C.truth()
    A = C.adj_runs()
    out = []
    for g, d in A.items():
        gi = tr["ids"].index(g)
        i = np.where((d["N"] == N) & (d["steps"] == 5000) & (d["rep"] == rep))[0][0]
        a, s = d["p"][i].ravel(), d["p_se"][i].ravel()
        b, sb = tr["p"][gi], tr["p_se"][gi]
        ok = (a > 0) & (b > 0)
        out.append((a[ok] - b[ok]) / np.sqrt(s[ok] ** 2 + sb[ok] ** 2))
    return np.concatenate(out)


def agg(rows, N, steps, f):
    v = [f(r) for r in rows if r["N"] == N and r["steps"] == steps]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else (np.nan, np.nan)


def plot(conv, bench=None):
    Sty.setup()
    rows = conv["rows"]
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.5))
    Ns = np.array([N for N in NS if any(r["N"] == N and r["steps"] == 5000 for r in rows)])
    c = Sty.COL["adjoint"]

    def curve(ax, f, label, color=c, marker="o", fill=True, fit=True):
        m = np.array([agg(rows, N, 5000, f) for N in Ns])
        ax.errorbar(Ns, m[:, 0], yerr=m[:, 1], color=color, marker=marker, ms=3.5, lw=1.2,
                    mfc=color if fill else "white", label=label)
        if fit:
            k = Ns >= 1000
            a = np.exp(np.mean(np.log(m[k, 0]) + 0.5 * np.log(Ns[k])))
            xx = np.geomspace(Ns.min(), 2e5, 50)
            ax.plot(xx, a / np.sqrt(xx), color=Sty.INK3, lw=0.8, ls="--", zorder=0)
            return a
        return None

    def cap750(ax, f, label=True):
        v = agg(rows, 20000, 750, f)
        if np.isfinite(v[0]):
            ax.errorbar([20000], [v[0]], yerr=[v[1]], color=Sty.INK, marker="X", ms=5, lw=0, elinewidth=1,
                        label="20k roots, max_steps 750" if label else None, zorder=5)

    ax = axs[0, 0]
    a = curve(ax, lambda r: r["p"]["rms"], "measured (vs 200k)")
    curve(ax, lambda r: r["pred_relse"], "self-reported s.e.", marker="o", fill=False, fit=False)
    cap750(ax, lambda r: r["p"]["rms"])
    ax.text(3e4, a / np.sqrt(3e4) * 1.5, r"$\propto N^{-1/2}$", color=Sty.INK2, fontsize=6.5)
    Sty.panel(ax, "a", "Voxel detection probability")
    ax.set_ylabel("p-weighted rms relative error")
    ax.legend(loc="lower left")

    ax = axs[0, 1]
    curve(ax, lambda r: r["t"]["rms"], "mean transport time")
    curve(ax, lambda r: r["sd"]["rms"], "time s.d. (0-60 ns)", color=Sty.BLUE[8], marker="s")
    cap750(ax, lambda r: r["t"]["rms"])
    cap750(ax, lambda r: r["sd"]["rms"], label=False)
    Sty.panel(ax, "b", "Voxel arrival-time moments")
    ax.set_ylabel("p-weighted rms relative error")
    ax.legend(loc="lower left")

    ax = axs[0, 2]
    curve(ax, lambda r: float(np.sqrt(np.mean(np.square(r["pvol_rel"])))), "whole-volume p")
    curve(ax, lambda r: float(np.sqrt(np.mean(np.square(r["tvol_rel"])))), "whole-volume mean time",
          color=Sty.BLUE[8], marker="s")
    cap750(ax, lambda r: float(np.sqrt(np.mean(np.square(r["pvol_rel"])))))     # rms, like the curves
    cap750(ax, lambda r: float(np.sqrt(np.mean(np.square(r["tvol_rel"])))), label=False)
    Sty.panel(ax, "c", "Per-geometry totals")
    ax.set_ylabel("rms relative error over geometries")
    ax.legend(loc="lower left")
    for ax in axs[0]:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("adjoint roots per geometry  N")
        Sty.logpct(ax)

    ax = axs[1, 0]
    for N, col in ((1000, Sty.BLUE[2]), (20000, Sty.BLUE[8])):
        x = pulls(N)
        ax.hist(np.clip(x, -6, 6), bins=np.linspace(-6, 6, 61), density=True, histtype="step", color=col, lw=1.2,
                label=f"N = {N//1000}k  (mean {x.mean():+.2f}, s.d. {x.std():.2f})")
    xx = np.linspace(-6, 6, 200)
    ax.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color=Sty.INK3, lw=0.8, ls="--", label="N(0,1)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 30)
    ax.set_xlabel(r"voxel pull  $(p_N - p_{200k}) / \sqrt{\sigma_N^2 + \sigma_{200k}^2}$")
    ax.set_ylabel("density")
    Sty.panel(ax, "d", "Error-estimate calibration")
    ax.legend(loc="upper center", fontsize=6, ncol=1)

    ax = axs[1, 1]
    m = np.array([agg(rows, N, 5000, lambda r: r["p"]["bias"]) for N in Ns])
    ax.errorbar(Ns, m[:, 0], yerr=m[:, 1], color=c, marker="o", ms=3.5, label="voxel p (log bias)")
    m = np.array([agg(rows, N, 5000, lambda r: r["t"]["bias"]) for N in Ns])
    ax.errorbar(Ns, m[:, 0], yerr=m[:, 1], color=Sty.BLUE[8], marker="s", ms=3.5, label="mean time")
    m = np.array([agg(rows, N, 5000, lambda r: r["sd"]["bias"]) for N in Ns])
    ax.errorbar(Ns, m[:, 0], yerr=m[:, 1], color=Sty.BLUE[5], marker="^", ms=3.5, label="time s.d.")
    for f, mk in ((lambda r: r["p"]["bias"], "X"), (lambda r: r["t"]["bias"], "X"), (lambda r: r["sd"]["bias"], "X")):
        v = agg(rows, 20000, 750, f)
        ax.plot([20000], [v[0]], color=Sty.INK, marker=mk, ms=5, lw=0)
    ax.axhline(0, color=Sty.INK2, lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("adjoint roots per geometry  N"); ax.set_ylabel("p-weighted mean residual")
    Sty.pct(ax)
    ax.annotate("max_steps 750", xy=(20000, agg(rows, 20000, 750, lambda r: r["t"]["bias"])[0]), xytext=(400, -0.10),
                fontsize=6, color=Sty.INK2, arrowprops=dict(arrowstyle="-", color=Sty.INK3, lw=0.6))
    Sty.panel(ax, "e", "Bias")
    ax.set_ylim(-0.23, 0.06)
    ax.legend(loc="center right", fontsize=6)

    ax = axs[1, 2]
    if bench and "adjoint" in bench:
        b = bench["adjoint"]
        for gname, col, mk in zip(b["geometries"], (Sty.BLUE[3], Sty.BLUE[6], Sty.BLUE[9]), "os^"):
            x = np.array(b["N"]); y = np.array(b["wall_s"][gname])
            ax.plot(x, y, color=col, marker=mk, ms=3.5, label=f"{gname} ({b['n_volumes'][gname]} volumes)")
        ax.set_title("")
        ax.text(0.03, 0.97, f"{b['threads']} threads, {b['cpu']}", transform=ax.transAxes, va="top", fontsize=6, color=Sty.INK2)
    else:
        m = np.array([agg(rows, N, 5000, lambda r: r["wall_mean"]) for N in Ns])
        ax.errorbar(Ns, m[:, 0], yerr=m[:, 1], color=c, marker="o", ms=3.5, label="mean over geometries (shared CPU)")
    xx = np.geomspace(250, 5e4, 10)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("adjoint roots per geometry  N")
    ax.set_ylabel("wall time per geometry map (s)")
    Sty.panel(ax, "f", "Cost")
    ax.legend(loc="lower right", fontsize=6)
    Sty.save(fig, "fig2_adjoint_convergence")


if __name__ == "__main__":
    conv = convergence()                       # raw time errors -> reference-noise estimate
    json.dump(conv, open(C.out("adj_convergence.json"), "w"))
    C.truth_time_noise.cache_clear()
    TN = C.truth_time_noise()
    print("200k reference noise (time):", TN)
    conv = convergence(TN)
    conv["truth_time_noise"] = TN
    json.dump(conv, open(C.out("adj_convergence.json"), "w"))
    bench = json.load(open(C.ST / "bench.json")) if (C.ST / "bench.json").exists() else None
    plot(conv, bench)
