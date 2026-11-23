"""Analyse fd/scan_runs.json + fd/time_runs.json (adjoint at u0 + s e_k, 8 seeds each) against fd/scan_net.json.
Per seed: least-squares line and quadratic through all available s; derivative = mean over seeds, error = sd/sqrt(n).
Also the plain central differences at each step h. Writes fd/scan_analysis.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import numpy as np
import fd_scan as FS

A = json.load(open(FS.F.OUT / "time_derivs.json"))
TR = json.load(open(FS.F.OUT / "time_runs.json"))
SR = json.load(open(FS.RUNS))
NET = json.load(open(FS.F.OUT / "scan_net.json"))
out = {}
for key in FS.CASES:
    du = A[key]["du"]
    pts = {}
    for i in range(8):
        for sgn, s in ((1, du), (-1, -du)):
            pts.setdefault(round(s, 3), {})[i] = TR[f"{key}|{sgn}|{i}"]
        for s in FS.S_ADJ:
            r = SR.get(f"{key}|{s:+.2f}|{i}")
            if r and "error" not in r:
                pts.setdefault(round(s, 3), {})[i] = r
    S = np.array(sorted(s for s in pts if len(pts[s]) == 8))
    res = {"s": S.tolist()}
    net = NET[key]; ns = np.array(sorted(map(float, net)))
    for q in ("p", "t", "sd", "t10"):
        Y = np.array([[np.log(pts[s][i][q]) if q == "p" else pts[s][i][q] for s in S] for i in range(8)])  # [seed, s]
        lin = np.array([np.polyfit(S, y, 1)[0] for y in Y])
        quad = np.array([np.polyfit(S, y, 2) for y in Y]) if len(S) >= 4 else None
        cd = {}
        for h in sorted({abs(s) for s in S}):
            if h in S and -h in S:
                d = (Y[:, list(S).index(h)] - Y[:, list(S).index(-h)]) / (2 * h)
                cd[f"{h:g}"] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(8))]
        ny = np.array([np.log(net[f"{s:+.3f}"][0][q]) if q == "p" else net[f"{s:+.3f}"][0][q] for s in ns])
        nsel = (ns >= S.min() - 1e-9) & (ns <= S.max() + 1e-9)
        res[q] = dict(central=cd, line=[float(lin.mean()), float(lin.std(ddof=1) / np.sqrt(8))],
                      quad_slope=None if quad is None else [float(quad[:, 1].mean()), float(quad[:, 1].std(ddof=1) / np.sqrt(8))],
                      quad_curv=None if quad is None else [float(2 * quad[:, 0].mean()), float(2 * quad[:, 0].std(ddof=1) / np.sqrt(8))],
                      mean_curve=Y.mean(0).tolist(), seed_sd=Y.std(0, ddof=1).tolist(),
                      net_line=float(np.polyfit(ns[nsel], ny[nsel], 1)[0]), net_curve=dict(s=ns.tolist(), y=ny.tolist()))
    out[key] = res
json.dump(out, open(FS.F.OUT / "scan_analysis.json", "w"), indent=1)
for key, r in out.items():
    print(f"\n{key}   s = {r['s']}")
    for q in ("t", "sd", "t10", "p"):
        x = r[q]
        cd = "  ".join(f"h={h}: {v[0]:+.3f}±{v[1]:.3f}" for h, v in x["central"].items())
        qs = f"quad {x['quad_slope'][0]:+.3f}±{x['quad_slope'][1]:.3f} curv {x['quad_curv'][0]:+.2f}±{x['quad_curv'][1]:.2f}" if x["quad_slope"] else ""
        print(f"  {q:4} {cd} | line {x['line'][0]:+.3f}±{x['line'][1]:.3f} {qs} | net {x['net_line']:+.3f}")
    print("  t-bar mean curve - mid:", " ".join(f"{s:+.2f}:{v - np.mean(r['t']['mean_curve']):+.3f}" for s, v in zip(r['s'], r['t']['mean_curve'])),
          " seed sd at each s:", " ".join(f"{v:.3f}" for v in r['t']['seed_sd']))
