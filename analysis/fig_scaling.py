"""Figure: surrogate accuracy vs training-set size and training length (40 test geometries, 200k reference)."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import csv
import json
import re

import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty


def score(pred, which):
    tr = C.truth()
    TN = C.truth_time_noise()
    p = pred[f"{which}_p0"]
    return dict(p=C.pmetrics(p, tr), t=C.tmetrics(pred[f"{which}_mt"], "mt", tr, TN["mt"]),
                sd=C.tmetrics(pred[f"{which}_tsd"], "tsd", tr, TN["tsd"]))


def collect():
    sp = C.sur_preds()
    rows = []
    for tag, d in sp.items():
        m = d["meta"]
        for which in ("best", "final"):
            if f"{which}_p0" in d:
                rows.append(dict(tag=tag, n=m["n"], epochs=m["epochs"], seed=m["seed"], which=which,
                                 old=tag.startswith("old_"), train_minutes=m.get("train_minutes"), **score(d, which)))
    # ensembles of final weights at each n (all seeds, 240 epochs)
    ens = []
    for n in sorted({r["n"] for r in rows}):
        tags = [t for t, d in sp.items() if d["meta"]["n"] == n and d["meta"]["epochs"] == 240 and not t.startswith("old_")]
        if len(tags) >= 2:
            avg = {f"final_{k}": np.mean([sp[t][f"final_{k}"] for t in tags], 0) for k in ("p0", "mt", "tsd")}
            ens.append(dict(n=n, members=len(tags), **score(avg, "final")))
    gp = []
    tr = C.truth(); TN = C.truth_time_noise()
    for n, d in C.gp_preds().items():
        gp.append(dict(n=n, p=C.pmetrics(d["p0"], tr), t=C.tmetrics(d["mt"], "mt", tr, TN["mt"]),
                       sd=C.tmetrics(d["tsd"], "tsd", tr, TN["tsd"]), fit_s=float(d["fit_s"]), predict_s=float(d["predict_s"])))
    return dict(rows=rows, ensembles=ens, gp=sorted(gp, key=lambda r: r["n"]))


def history(tag):
    for f in (C.ST / "sur" / f"{tag}.history.csv",):
        if f.exists():
            return list(csv.DictReader(open(f)))
    stems = {"n628_e240_s20261003": "sur2_628", "n450_e240_s20261003": "sur2_450", "n278_e240_s20261003": "sur_278_e240",
             "n278_e240_s777": "sur2_278_seed777"}
    f = C.T / "surrogate_lc" / f"{stems.get(tag, tag)}.history.csv"
    return list(csv.DictReader(open(f))) if f.exists() else None


def plot(S):
    Sty.setup()
    conv = json.load(open(C.out("adj_convergence.json")))
    fig, axs = plt.subplots(2, 3, figsize=(7.1, 5.0))
    rows = [r for r in S["rows"] if not r["old"] and r["epochs"] == 240]
    ns = sorted({r["n"] for r in rows})

    def adj_ref(ax, f, Ns=(1000, 5000, 20000)):
        for N, col in zip(Ns, (Sty.BLUE[1], Sty.BLUE[4], Sty.BLUE[8])):
            v = [f(r) for r in conv["rows"] if r["N"] == N and r["steps"] == 5000]
            if v:
                ax.axhline(np.mean(v), color=col, lw=0.8, ls="--", zorder=0, label=f"adjoint, {N//1000}k roots")

    for ax, key, letter, title in ((axs[0, 0], "p", "a", "Voxel detection probability"),
                                   (axs[0, 1], "t", "b", "Voxel mean arrival time"),
                                   (axs[0, 2], "sd", "c", "Voxel arrival-time s.d.")):
        for which, mfc, lab in (("final", None, "network, final weights"), ("best", "white", "network, best-val checkpoint")):
            m = np.array([[np.mean([r[key]["rms"] for r in rows if r["n"] == n and r["which"] == which]),
                           np.std([r[key]["rms"] for r in rows if r["n"] == n and r["which"] == which], ddof=0)] for n in ns])
            ax.errorbar(np.array(ns) * (1.0 if which == "final" else 1.03), m[:, 0], yerr=m[:, 1], color=Sty.COL["network"],
                        marker="s", ms=3.3, mfc=mfc or Sty.COL["network"], lw=1.1 if which == "final" else 0.8,
                        ls="-" if which == "final" else ":", label=lab)
            if which == "final" and key == "p":
                k = np.array(ns) >= 100
                b, a = np.polyfit(np.log(np.array(ns)[k]), np.log(m[k, 0]), 1)
                fitp = dict(a=float(np.exp(a)), b=float(b))
                ax.text(330, np.exp(a) * 330 ** b * 1.3,f"$\\propto n^{{{b:.2f}}}$", fontsize=6.5, color=Sty.COL["network"])
        g = S["gp"]
        ax.plot([r["n"] for r in g], [r[key]["rms"] for r in g], color=Sty.COL["gp"], marker="D", ms=3, label="GP (PCA-48)")
        E = S["ensembles"]
        if E:
            ax.plot([r["n"] for r in E], [r[key]["rms"] for r in E], color=Sty.ORANGE[5], marker="*", ms=6, lw=0,
                    label="network ensemble (final)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(40, 700)
        adj_ref(ax, lambda r: r[key]["rms"])
        Sty.logpct(ax, ticks=(0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2))
        ax.set_xlabel("training geometries  n"); ax.set_ylabel("p-weighted rms error vs 200k")
        Sty.logx_int(ax, (50, 100, 200, 300, 600))
        Sty.panel(ax, letter, title)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6.3)

    # d: epochs at n = 628
    ax = axs[1, 0]
    er = sorted([r for r in S["rows"] if r["n"] == 628 and r["seed"] == 20261003], key=lambda r: r["epochs"])
    for key, col, mk, lab in (("p", Sty.COL["network"], "s", "voxel p"), ("t", Sty.ORANGE[5], "^", "mean time"),
                              ("sd", Sty.ORANGE[1], "v", "time s.d.")):
        for which, ls, mfc in (("final", "-", None), ("best", ":", "white")):
            v = [(r["epochs"], r[key]["rms"]) for r in er if r["which"] == which]
            if v:
                v = np.array(v)
                ax.plot(v[:, 0], v[:, 1], ls=ls, marker=mk, color=col, ms=3, mfc=mfc or col,
                        label=lab if which == "final" else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    Sty.logpct(ax, ticks=(0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2))
    Sty.logx_int(ax, (10, 20, 40, 80, 160, 240, 480))
    ax.set_xlabel("training epochs (n = 628)"); ax.set_ylabel("p-weighted rms error vs 200k")
    ax.legend(fontsize=5.6, loc="upper right", title="solid: final, dotted: best-val", title_fontsize=5.4)
    Sty.panel(ax, "d", "Training length")

    # e: validation curves
    ax = axs[1, 1]
    cmap = {50: Sty.ORANGE[0], 100: Sty.ORANGE[1], 200: Sty.ORANGE[2], 278: Sty.ORANGE[3], 450: Sty.ORANGE[4], 628: Sty.ORANGE[6]}
    for n, col in cmap.items():
        tag = {278: "n278_e240_s20261003", 450: "n450_e240_s20261003", 628: "n628_e240_s20261003"}.get(n, f"n{n}_e240_s1")
        h = history(tag)
        if not h:
            continue
        e = np.array([int(r["epoch"]) for r in h]); v = np.array([float(r["val_logp_med"]) for r in h])
        ax.plot(e, v, color=col, lw=0.9, label=f"n = {n}")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("validation median |ln p error|  (16 geom.)")
    Sty.logpct(ax, ticks=(0.01, 0.02, 0.03, 0.05, 0.1, 0.2))
    ax.legend(fontsize=5.6, ncol=2, loc="upper right")
    Sty.panel(ax, "e", "Validation curves")

    # f: per-geometry totals and training cost
    ax = axs[1, 2]
    for which, mfc in (("final", None), ("best", "white")):
        m = [np.mean([r["p"]["geo_rel_rms"] for r in rows if r["n"] == n and r["which"] == which]) for n in ns]
        ax.plot(ns, m, color=Sty.COL["network"], marker="s", ms=3, mfc=mfc or Sty.COL["network"], ls="-" if which == "final" else ":",
                label="network, " + which)
    ax.plot([r["n"] for r in S["gp"]], [r["p"]["geo_rel_rms"] for r in S["gp"]], color=Sty.COL["gp"], marker="D", ms=3, label="GP")
    if S["ensembles"]:
        ax.plot([r["n"] for r in S["ensembles"]], [r["p"]["geo_rel_rms"] for r in S["ensembles"]], "*", color=Sty.ORANGE[5], ms=6,
                label="ensemble")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(40, 700)
    adj_ref(ax, lambda r: float(np.sqrt(np.mean(np.square(r["pvol_rel"])))))
    Sty.logpct(ax, ticks=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1))
    ax.set_xlabel("training geometries  n"); ax.set_ylabel("rms relative error over geometries")
    Sty.logx_int(ax, (50, 100, 200, 300, 600))
    Sty.panel(ax, "f", "Whole-volume p per geometry")
    Sty.save(fig, "fig6_surrogate_scaling")


if __name__ == "__main__":
    S = collect()
    json.dump(S, open(C.out("scaling.json"), "w"), indent=1, default=float)
    plot(S)
    for r in sorted(S["rows"], key=lambda r: (r["n"], r["epochs"], r["seed"], r["which"])):
        print(f"{r['tag']:24} {r['which']:5} p {100*r['p']['rms']:.2f}% t {100*r['t']['rms']:.2f}% sd {100*r['sd']['rms']:.2f}% geo {100*r['p']['geo_rel_rms']:.2f}%")
    for r in S["gp"]:
        print(f"GP n={r['n']:4} p {100*r['p']['rms']:.2f}% t {100*r['t']['rms']:.2f}% sd {100*r['sd']['rms']:.2f}%")
