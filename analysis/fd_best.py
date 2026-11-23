"""Best derivative per case from the line scans: symmetric least-squares line when >= 4 symmetric points (|s| <= 0.16);
otherwise a quadratic least-squares fit through all valid points with |s| <= 0.16 if that is more precise than the
central difference at the original step; otherwise the original central difference. Per seed, mean +- sd/sqrt(8).
Network: the same estimator on the same points (single network sur2_628_final, as in the paper).
Writes fd/best_derivs.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os

import numpy as np

os.environ["ALL"] = "1"
import fd_stencils as ST  # noqa: E402

NET_OLD = json.load(open(ST.FS.F.OUT / "network_time.json"))["single"]
out = {}
for key in ST.A:
    pts = ST.points(key); net = ST.NET[key]
    S_all = np.array(sorted(s for s in pts if abs(s) <= 0.1601))
    NS = np.array(sorted(float(x) for x in net))
    S_sym = np.array([s for s in S_all if -s in S_all])
    du = ST.A[key]["du"]
    row = dict(param=ST.A[key]["param"], s_points=S_all.tolist())
    for q in ST.Q:
        Y = np.array([[ST.f(q, pts[s][i][q]) for s in S_all] for i in range(8)])
        NV = np.array([ST.f(q, net[f"{x:+.3f}"][0][q]) for x in NS])
        lo, hi = S_all.min(), S_all.max()
        sel = (NS >= lo - 1e-9) & (NS <= hi + 1e-9)
        # network: smooth, so its slope at 0 from a quadratic through its own fine scan over the same range
        nslope = float(np.polyfit(NS[sel], NV[sel], 2)[1]) if sel.sum() >= 3 else NET_OLD[key][q]
        cand = {}
        if len(S_sym) >= 4:
            m = np.isin(S_all, S_sym); s = S_all[m]
            cand["line"] = ((Y[:, m] * s).sum(1) / np.sum(s ** 2), nslope)
        if len(S_all) >= 4:
            cand["quad"] = (np.array([np.polyfit(S_all, y, 2)[1] for y in Y]), nslope)
        i1, i0 = list(S_all).index(round(du, 3)), list(S_all).index(round(-du, 3))
        cand["central"] = ((Y[:, i1] - Y[:, i0]) / (2 * du), NET_OLD[key][q])
        se = {k: v[0].std(ddof=1) / np.sqrt(8) for k, v in cand.items()}
        pick = "line" if "line" in cand else min(cand, key=lambda k: se[k])
        row[q] = dict(method=pick, adj=float(cand[pick][0].mean()), se=float(se[pick]), net=float(cand[pick][1]),
                      central=float(cand["central"][0].mean()), central_se=float(se["central"]), net_central=float(cand["central"][1]))
    out[key] = row
json.dump(out, open(ST.FS.F.OUT / "best_derivs.json", "w"), indent=1)
import collections
print("method used (t):", collections.Counter(r["t"]["method"] for r in out.values()))
for q in ST.Q:
    se_old = np.array([r[q]["central_se"] for r in out.values()]); se_new = np.array([r[q]["se"] for r in out.values()])
    a = np.array([r[q]["adj"] for r in out.values()]); n = np.array([r[q]["net"] for r in out.values()])
    a0 = np.array([r[q]["central"] for r in out.values()]); n0 = np.array([r[q]["net_central"] for r in out.values()])
    print(f"{q:4} rms se old {np.sqrt(np.mean(se_old**2)):.4f} new {np.sqrt(np.mean(se_new**2)):.4f} | "
          f"old: corr {np.corrcoef(a0,n0)[0,1]:.2f} rms diff {np.sqrt(np.mean((n0-a0)**2)):.4f} rms deriv {np.sqrt(np.mean(a0**2)):.4f} | "
          f"new: corr {np.corrcoef(a,n)[0,1]:.2f} rms diff {np.sqrt(np.mean((n-a)**2)):.4f} rms deriv {np.sqrt(np.mean(a**2)):.4f} "
          f"chi2 {np.sum(((n-a)/se_new)**2):.0f}/24")
print()
for k, r in out.items():
    print(f"{k:28} {r['t']['method']:7} t {r['t']['adj']:+.3f}±{r['t']['se']:.3f} net {r['t']['net']:+.3f}  (old {r['t']['central']:+.3f}±{r['t']['central_se']:.3f})"
          f" | p {r['p']['adj']:+.3f}±{r['p']['se']:.3f} net {r['p']['net']:+.3f} | t10 {r['t10']['adj']:+.3f}±{r['t10']['se']:.3f} net {r['t10']['net']:+.3f}")
