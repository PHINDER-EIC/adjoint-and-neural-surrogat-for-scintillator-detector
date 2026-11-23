"""Tables for paper v1 (simplified): no GP, single network as 'the surrogate', narrow columns.
Writes paper_v1/tables/*.tex. table* only where a single column is too narrow."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
from pathlib import Path

import numpy as np

import common as C

OUT = paths.RELEASE / "work" / "tables"
for f in OUT.glob("*.tex"):
    f.unlink()
pc = lambda x, d=1: f"{100 * x:.{d}f}"


def write(name, label, caption, cols, header, rows, wide=False):
    env = "table*" if wide else "table"
    s = [f"\\begin{{{env}}}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{cols}}}", "\\hline", " & ".join(header) + " \\\\", "\\hline"]
    for r in rows:
        s.append("\\hline" if r == "---" else " & ".join(str(x) for x in r) + " \\\\")
    s += ["\\hline", "\\end{tabular}", f"\\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(s) + "\n", encoding="utf-8")
    print("wrote", name)


def tsec(s):
    return f"{s*1e3:.1f} ms" if s < 1 else (f"{s:.1f} s" if s < 120 else (f"{s/60:.0f} min" if s < 7200 else
                                                                      (f"{s/3600:.1f} h" if s < 2 * 86400 else f"{s/86400:.0f} d")))


# ---------------------------------------------------------------- main: agreement with Geant4
A = json.load(open(C.ST / "g4_agreement.json"))
rows = []
for m, lab in (("adjoint", "adjoint ($2\\times10^5$ roots)"), ("jax", "forward engine (JAX)"), ("network", "network")):
    s = A[m]
    rows.append([lab, s["n"], f"{s['mean']:.3f} $\\pm$ {s['mean_se']:.3f}", pc(s["rms_dev"]), f"{s['chi2']:.0f}",
                 f"{100*s['tmean_mean']:+.1f} / {pc(s['tmean_rms'])}", f"{100*s['tsd_mean']:+.1f} / {pc(s['tsd_rms'])}"])
write("tab_geant4", "tab:geant4",
      "Agreement with Geant4 (configuration matching the optical model) for the captured fraction in 5 emission regions $\\times$ "
      "2 channels of 8 test geometries (4 for the forward engine). Ratio: weighted mean of method/Geant4; rms: rms of "
      "(method/Geant4 $-$ 1), to be compared with the median Geant4 statistical error of 2.0\\%; $\\chi^2$ for the given number "
      "of values. Last two columns: mean / rms relative difference (\\%) of the channel-0 mean arrival time and of its "
      "standard deviation.",
      "lrrrrrr", ["method", "values", "ratio", "rms (\\%)", "$\\chi^2$", "$\\bar t$ (\\%)", "$s_t$ (\\%)"], rows, wide=True)

# ---------------------------------------------------------------- main: surrogate accuracy
S = json.load(open(C.ST / "scaling.json"))
Bl = json.load(open(C.ST / "blind.json"))
T = json.load(open(C.ST / "residual_stats.json"))
Rf = [r for r in S["rows"] if r["n"] == 628 and r["epochs"] == 240 and r["which"] == "final" and not r["old"]]
ens = [r for r in S["ensembles"] if r["n"] == 628][0]
mean = lambda R, f: float(np.mean([f(r) for r in R]))
bn = [Bl[f"network_{i}"] for i in range(3)]
rows = [["network, test", pc(mean(Rf, lambda r: r["p"]["rms"]), 2), pc(mean(Rf, lambda r: r["p"]["offset_rms"]), 2),
         pc(mean(Rf, lambda r: r["t"]["rms"]), 2), pc(mean(Rf, lambda r: r["sd"]["rms"]), 2)],
        ["network, new designs", pc(mean(bn, lambda r: r["p"]["rms"]), 2), pc(mean(bn, lambda r: r["p"]["offset_rms"]), 2),
         pc(mean(bn, lambda r: r["t"]["rms"]), 2), pc(mean(bn, lambda r: r["sd"]["rms"]), 2)],
        ["3 networks, test", pc(ens["p"]["rms"], 2), pc(ens["p"]["offset_rms"], 2), pc(ens["t"]["rms"], 2), pc(ens["sd"]["rms"], 2)],
        ["3 networks, new designs", pc(Bl["network_ensemble"]["p"]["rms"], 2), pc(Bl["network_ensemble"]["p"]["offset_rms"], 2),
         pc(Bl["network_ensemble"]["t"]["rms"], 2), pc(Bl["network_ensemble"]["sd"]["rms"], 2)], "---",
        ["adjoint $2\\times10^4$, test", pc(T["adj_20k"]["p"]["rms"], 2), pc(T["adj_20k"]["p"]["offset_rms"], 2), pc(T["adj_20k"]["t"]["rms"], 2),
         pc(T["adj_20k"]["sd"]["rms"], 2)]]
write("tab_surrogate", "tab:surrogate",
      "Voxel-level error (\\%) against $2\\times10^5$-root adjoint maps: detection probability $\\varepsilon_p$ and its "
      "per-geometry offset, mean arrival time $\\varepsilon_{\\bar t}$, arrival-time spread $\\varepsilon_{s_t}$. Network: mean "
      "of three training seeds (628 training geometries); 3 networks: their average. Test: 40 held-out geometries; new designs: "
      "40 designs from an independent Sobol sequence, evaluated once.",
      "lrrrr", ["", "$\\varepsilon_p$", "offset", "$\\varepsilon_{\\bar t}$", "$\\varepsilon_{s_t}$"], rows)

# ---------------------------------------------------------------- appendix: physics options
R = json.load(open(C.ST / "regions_single.json"))["regions"]
import fig_forward as FF
rows = []
for a, b, l in (("g4_pol", "g4_unpol", "polarised / averaged"), ("g4_full", "g4_pol", "re-entry / none")):
    P = FF.pairs(R, a, b)
    for reg in ["all", "whole", "hotspot"]:
        v = [p for p in P if reg == "all" or p["region"] == reg]
        r = np.array([p["ratio"] for p in v]); se = np.array([p["ratio_se"] for p in v]); w = 1 / se ** 2
        rows.append([l if reg == "all" else "", reg, len(r), f"{np.sum(w*r)/w.sum():.4f} $\\pm$ {1/np.sqrt(w.sum()):.4f}"])
hr = [p for p in FF.pairs(R, "g4_full", "g4_pol", chans=("p0",)) if p["region"] == "hotspot"]
r = np.array([p["ratio"] for p in hr]); se = np.array([p["ratio_se"] for p in hr]); w = 1 / se ** 2
rows.append(["", "hotspot, ch. 0", len(r), f"{np.sum(w*r)/w.sum():.4f} $\\pm$ {1/np.sqrt(w.sum()):.4f}"])
write("tab_physics_options", "tab:physics_options",
      "Geant4 ratios measuring the two approximations of the optical model (weighted means over geometries and channels).",
      "llrr", ["ratio", "region", "n", "mean"], rows)

# ---------------------------------------------------------------- appendix: adjoint versions
Ac = json.load(open(C.ST / "adj_convergence.json"))
B = json.load(open(C.ST / "bench.json"))
rows = []
for steps in (5000, 750):
    for N in sorted({r["N"] for r in Ac["rows"] if r["steps"] == steps}):
        Rr = [r for r in Ac["rows"] if r["N"] == N and r["steps"] == steps]
        f = lambda g: float(np.mean([g(r) for r in Rr]))
        wall = tsec(B["adjoint"]["wall_s"]["B_train_g0180"][B["adjoint"]["N"].index(N)]) if (steps == 5000 and N in B["adjoint"]["N"]) else "--"
        Nl = f"{N:,}".replace(",", "\\,") + ("" if steps == 5000 else "$^*$")
        rows.append([Nl, pc(f(lambda r: r["p"]["rms"])), pc(f(lambda r: r["pred_relse"])), pc(f(lambda r: np.sqrt(np.mean(np.square(r["pvol_rel"]))))),
                     pc(f(lambda r: r["t"]["rms"])), pc(f(lambda r: r["sd"]["rms"])), f"{100*f(lambda r: r['t']['bias']):+.1f}", wall])
write("tab_adjoint_versions", "tab:adjoint_versions",
      "Adjoint maps versus roots $N$ (40 test geometries, mean of independent replicates; errors in \\%). Est.: the tracer's own "
      "error estimate; vol.: whole-volume $p$; bias: mean-time bias. Wall time for a 77-volume scene on 14 threads. "
      "$^*$Step cap of 750 instead of 5000.",
      "lrrrrrrr", ["$N$", "$\\varepsilon_p$", "est.", "vol.", "$\\varepsilon_{\\bar t}$", "$\\varepsilon_{s_t}$", "bias", "wall"], rows, wide=True)

# ---------------------------------------------------------------- appendix: split threshold
Sp = json.load(open(C.ST / "split.json"))
rows, ref = [], None
for thr in sorted({r["thr"] for r in Sp}):
    Rr = [r for r in Sp if r["thr"] == thr]
    e = float(np.sqrt(np.mean([r["p_rms"] ** 2 for r in Rr]))); w = float(np.mean([r["wall_s"] for r in Rr]))
    rows.append(["analog" if thr > 1 else thr, pc(e, 2), f"{np.mean([r['splits_per_root'] for r in Rr]):.1f}", f"{w:.2f}", 1 / (e ** 2 * w)])
    if thr == 0.005:
        ref = 1 / (e ** 2 * w)
rows = [r[:-1] + [f"{r[-1]/ref:.1f}"] for r in rows]
write("tab_split", "tab:split",
      "Fresnel split threshold $\\tau$ for $2\\times10^4$-root maps (4 geometries, 2 seed pairs, 14 threads). Efficiency "
      "$\\propto 1/(\\varepsilon_p^2 t)$ relative to $\\tau = 0.005$. All settings are unbiased within 0.3\\%.",
      "lrrrr", ["$\\tau$", "$\\varepsilon_p$ (\\%)", "splits/root", "time (s)", "eff."], rows)

# ---------------------------------------------------------------- appendix: surrogate versions
rows = []
for n in (50, 100, 200, 278, 450, 628):
    Rf = [r for r in S["rows"] if r["n"] == n and r["epochs"] == 240 and r["which"] == "final" and not r["old"]]
    Rb = [r for r in S["rows"] if r["n"] == n and r["epochs"] == 240 and r["which"] == "best" and not r["old"]]
    E = [r for r in S["ensembles"] if r["n"] == n][0]
    rows.append([n, 240, f"{pc(mean(Rf, lambda r: r['p']['rms']), 2)} $\\pm$ {pc(np.std([r['p']['rms'] for r in Rf]), 2)}",
                 pc(mean(Rb, lambda r: r["p"]["rms"]), 2), pc(E["p"]["rms"], 2), pc(mean(Rf, lambda r: r["t"]["rms"]), 2)])
rows.append("---")
for e in (10, 20, 40, 80, 160, 480):
    Rr = {r["which"]: r for r in S["rows"] if r["n"] == 628 and r["epochs"] == e and r["seed"] == 20261003}
    rows.append([628, e, pc(Rr["final"]["p"]["rms"], 2), pc(Rr["best"]["p"]["rms"], 2), "--", pc(Rr["final"]["t"]["rms"], 2)])
abl = np.load(C.ST / "abl/nogru_n628_e240_s1.pred.npz")
tr = C.truth(); TN = C.truth_time_noise()
rows.append("---")
rows.append(["628$^\\dagger$", 240, pc(C.pmetrics(abl["final_p0"], tr)["rms"], 2), pc(C.pmetrics(abl["best_p0"], tr)["rms"], 2), "--",
             pc(C.tmetrics(abl["final_mt"], "mt", tr, TN["mt"])["rms"], 2)])
write("tab_surrogate_versions", "tab:surrogate_versions",
      "Network errors (\\%, 40 test geometries) versus training geometries $n$ and epochs. Final: last-epoch weights, mean "
      "$\\pm$ s.d. over 3 seeds (one seed for the epoch scan); best: lowest validation loss; 3 nets: average of the three final "
      "networks. $^\\dagger$Without the GRU guide encoder (one seed).",
      "rrrrrr", ["$n$", "epochs", "$\\varepsilon_p$ final", "best", "3 nets", "$\\varepsilon_{\\bar t}$"], rows)

# ---------------------------------------------------------------- appendix: cost per map
SP = json.load(open(C.ST / "speed_summary.json"))
E = json.load(open(C.ST / "event_cost.json"))
import fig_speed
M = fig_speed.photons_for([r for r in SP["adjoint"] if r["N"] == 20000][0]["err"])
rows = [["adjoint, $2\\times10^4$ roots", tsec([r for r in SP["adjoint"] if r["N"] == 20000][0]["s"])],
        ["adjoint, $2\\times10^4$, analog", "about 0.9 s"],
        [f"Geant4, ${M/1e8:.1f}\\times10^8$ photons, 14 cores", tsec(M / B["g4"]["parallel14_pol"]["photons_per_s"])],
        ["forward engine (JAX), 3 cores", tsec(M / max(v["photons_per_s"] for v in B["jax"]["batches"].values()))],
        ["network, full map", tsec(SP["network"]["s"])],
        ["network, full map + gradient", tsec(SP["network"]["grad_s"])],
        ["network, 1000 points", tsec(E["network_points"]["1000"])],
        ["superposition, 1000 deposits", tsec(E["superposition"]["1000"])]]
write("tab_cost_per_map", "tab:cost_per_map",
      "Wall time for one complete map ($20\\times20\\times8$ voxels, both channels) at the precision of a $2\\times10^4$-root "
      "adjoint map (2.4\\%), and per-event costs, on the 14-core CPU. ``+ gradient'' includes the backward pass that gives the "
      "derivatives of all outputs with respect to the design.", "lr", ["", "time"], rows)
