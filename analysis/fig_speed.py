"""Figure: speed vs accuracy of every method for full 20x20x8 detection-probability maps (one geometry, both channels).

Forward Monte Carlo (JAX, Geant4): uniform emission of M photons over the cell gives per-voxel binomial errors;
the p-weighted rms relative error of the map is eps^2 = 3200 * sum(1 - p_v) / (M * sum p_v)  ->  M(eps).
Rates are measured (bench.json). Adjoint: measured error (adj_convergence.json) and wall time (bench.json).
Surrogates: inference time per map; their error does not fall with run time, so they are single points.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty


def photons_for(eps):
    tr = C.truth()
    k = 3200 * (1 - tr["p"]).sum(1) / tr["p"].sum(1)        # per geometry: M * eps^2
    return float(np.median(k)) / eps ** 2


def load():
    B = json.load(open(C.ST / "bench.json"))
    conv = json.load(open(C.out("adj_convergence.json")))
    sc = json.load(open(C.out("scaling.json")))
    return B, conv, sc


def numbers(B, conv, sc):
    out = {}
    # adjoint: error vs N (mean over replicates), time vs N (median geometry of the three)
    Ns = B["adjoint"]["N"]
    g = "B_train_g0180"
    tN = dict(zip(Ns, B["adjoint"]["wall_s"][g]))
    prod = B["adjoint"]["products_s"][g]
    err = {N: np.mean([r["p"]["rms"] for r in conv["rows"] if r["N"] == N and r["steps"] == 5000]) for N in Ns}
    out["adjoint"] = [dict(N=N, s=tN[N] + prod, err=err[N]) for N in Ns]
    # JAX uses ~3 cores per process: scale to the whole 14-core CPU assuming independent processes in parallel
    jr_proc = max(v["photons_per_s"] for v in B["jax"]["batches"].values())
    jr = max(v["photons_per_s"] * 14 / v["cpu_over_wall"] for v in B["jax"]["batches"].values())
    g4r = B["g4"]["parallel14_pol"]["photons_per_s"]
    g4r_unpol = B["g4"]["single_unpol"]["photons_per_s"] * g4r / B["g4"]["single_pol"]["photons_per_s"]
    out["rates"] = dict(jax=jr, jax_one_process=jr_proc, g4_pol_14=g4r, g4_unpol_14=g4r_unpol)
    eps = np.geomspace(0.005, 0.3, 30)
    out["jax"] = [dict(err=float(e), s=photons_for(e) / jr) for e in eps]
    out["g4"] = [dict(err=float(e), s=photons_for(e) / g4r_unpol) for e in eps]
    ens = [r for r in sc["ensembles"] if r["n"] == 628]
    single = [r for r in sc["rows"] if r["n"] == 628 and r["epochs"] == 240 and r["which"] == "final"]
    out["network"] = dict(s=B["network"]["map_s_threads14"], err=float(np.mean([r["p"]["rms"] for r in single])),
                          err_ens=ens[0]["p"]["rms"] if ens else None, n_ens=ens[0]["members"] if ens else None,
                          grad_s=B["network"]["map_plus_grad_s_threads14"])
    gp = [r for r in sc["gp"] if r["n"] == 628][0]
    out["gp"] = dict(s=gp["predict_s"] / 40, err=gp["p"]["rms"], fit_s=gp["fit_s"])
    # amortised cost of the surrogate: 628 adjoint maps at 20k roots + training (240 epochs, measured on this CPU)
    t20 = out["adjoint"][Ns.index(20000)]["s"]
    train_s = 22.62 * 60
    out["surrogate_build_s"] = dict(data=628 * t20, train=train_s, gp_fit=out["gp"]["fit_s"])
    return out


def plot(Nb):
    Sty.setup()
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.2), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    ax = axs[0]
    a = Nb["adjoint"]
    ax.plot([r["s"] for r in a], [r["err"] for r in a], marker="o", color=Sty.COL["adjoint"], ms=3.5, label=Sty.LAB["adjoint"])
    for r in a:
        if r["N"] in (1000, 5000, 20000, 50000):
            ax.annotate(f"{r['N']//1000}k", (r["s"], r["err"]), xytext=(4, 2), textcoords="offset points", fontsize=5.6,
                        color=Sty.COL["adjoint"])
    for key, lab in (("g4", "Geant4 forward, 14 processes"), ("jax", "JAX forward, 14-core equiv.")):
        ax.plot([r["s"] for r in Nb[key]], [r["err"] for r in Nb[key]], color=Sty.COL[key], lw=1.2, label=lab,
                ls="-" if key == "g4" else "--")
    n = Nb["network"]
    ax.plot([n["s"]], [n["err"]], marker="s", color=Sty.COL["network"], ms=5, lw=0, label="Network surrogate (628)")
    if n["err_ens"]:
        ax.plot([n["s"] * n["n_ens"]], [n["err_ens"]], marker="*", color=Sty.ORANGE[5], ms=7, lw=0,
                label=f"Network ensemble of {n['n_ens']}")
    g = Nb["gp"]
    ax.plot([g["s"]], [g["err"]], marker="D", color=Sty.COL["gp"], ms=4.5, lw=0, label="GP surrogate (628)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-4, 1e8); ax.set_ylim(0.005, 0.3)
    for s, lab in ((1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day"), (86400 * 365, "1 yr")):
        ax.axvline(s, color=Sty.GRID, lw=0.8, zorder=0)
        ax.text(s * 1.25, 0.0055, lab, fontsize=5.5, color=Sty.INK3, va="bottom")
    Sty.logpct(ax, ticks=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2))
    ax.set_xlabel("wall time per full map on one 14-core CPU (s)")
    ax.set_ylabel("voxel p error (p-weighted rms vs 200k)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6)
    Sty.panel(ax, "a", "Speed-accuracy frontier")

    ax = axs[1]
    t20 = [r for r in a if r["N"] == 20000][0]
    items = [("Network\nsurrogate", n["s"], Sty.COL["network"]), ("GP\nsurrogate", g["s"], Sty.COL["gp"]),
             ("Adjoint\n20k roots", t20["s"], Sty.COL["adjoint"]),
             ("Geant4 at\nsame error", photons_for(t20["err"]) / Nb["rates"]["g4_unpol_14"], Sty.COL["g4"]),
             ("JAX at\nsame error", photons_for(t20["err"]) / Nb["rates"]["jax"], Sty.COL["jax"])]
    y = np.arange(len(items))[::-1]
    ax.barh(y, [i[1] for i in items], color=[i[2] for i in items], height=0.6)
    for yy, (lab, s, col) in zip(y, items):
        txt = f"{s*1e3:.1f} ms" if s < 1 else (f"{s:.1f} s" if s < 120 else (f"{s/3600:.1f} h" if s < 86400 * 2 else f"{s/86400:.0f} days"))
        ax.text(s * 1.4, yy, txt, va="center", fontsize=6, color=Sty.INK)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=6)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1e9); ax.set_xlabel("wall time per map (s)")
    ax.grid(axis="y", visible=False)
    Sty.panel(ax, "b", f"One map at {100*t20['err']:.1f}% error")

    ax = axs[2]
    b = Nb["surrogate_build_s"]
    K = np.geomspace(1, 1e6, 200)
    ax.plot(K, K * t20["s"] / 3600, color=Sty.COL["adjoint"], label="adjoint 20k per evaluation")
    ax.plot(K, (b["data"] + b["train"] + K * n["s"]) / 3600, color=Sty.COL["network"], label="network: data + training + inference")
    ax.plot(K, (b["data"] + b["gp_fit"] + K * g["s"]) / 3600, color=Sty.COL["gp"], ls="--", label="GP: data + fit + inference")
    kb = (b["data"] + b["train"]) / (t20["s"] - n["s"])
    ax.axvline(kb, color=Sty.INK3, lw=0.6, ls=":")
    ax.text(kb * 1.2, 2e-3, f"break-even\n~{kb:.0f} maps", fontsize=5.8, color=Sty.INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel("number of map evaluations"); ax.set_ylabel("cumulative CPU wall time (h)")
    ax.legend(fontsize=5.5, loc="upper left")
    Sty.panel(ax, "c", "Amortisation")
    Sty.save(fig, "fig7_speed_accuracy")


if __name__ == "__main__":
    Nb = numbers(*load())
    json.dump(Nb, open(C.out("speed_summary.json"), "w"), indent=1, default=float)
    plot(Nb)
    print(json.dumps({k: v for k, v in Nb.items() if k not in ("jax", "g4")}, indent=1, default=float))
