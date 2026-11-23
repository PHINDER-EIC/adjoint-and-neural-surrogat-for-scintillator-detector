"""All 24 derivatives from the line scans: central differences at every symmetric step h, five-point stencils
[f(-2h) - 8 f(-h) + 8 f(h) - f(2h)] / 12h, and the least-squares line through the symmetric points, per seed
(8 seed pairs, common random numbers across s), mean +- sd/sqrt(8). Network: the same stencils on its scan.
Writes fd/stencils.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os

import numpy as np

os.environ["ALL"] = "1"
import fd_scan as FS  # noqa: E402

A = json.load(open(FS.F.OUT / "time_derivs.json"))
TR = json.load(open(FS.F.OUT / "time_runs.json"))
SR = json.load(open(FS.RUNS))
NET = json.load(open(FS.F.OUT / "scan_net.json"))
Q = ("p", "t", "sd", "t10")
f = lambda q, v: np.log(v) if q == "p" else v


def points(key):
    du = A[key]["du"]; pts = {}
    for i in range(8):
        for sgn in (1, -1):
            pts.setdefault(round(sgn * du, 3), {})[i] = TR[f"{key}|{sgn}|{i}"]
        for s in FS.S_ADJ:
            r = SR.get(f"{key}|{s:+.2f}|{i}")
            if r and "error" not in r:
                pts.setdefault(round(s, 3), {})[i] = r
    return {s: v for s, v in pts.items() if len(v) == 8}


def estimates(val):
    """val(s) -> array over seeds (or scalar). Returns dict name -> per-seed estimate."""
    hs = sorted({abs(s) for s in val.keys() if -s in val})
    out = {}
    for h in hs:
        out[f"c{h:g}"] = (val[h] - val[-h]) / (2 * h)
        if 2 * h in hs or round(2 * h, 3) in hs:
            h2 = round(2 * h, 3)
            out[f"f{h:g}"] = (val[-h2] - 8 * val[-h] + 8 * val[h] - val[h2]) / (12 * h)
    S = np.array([s for s in val if -s in val])
    if len(S) >= 4:
        Y = np.array([val[s] for s in S])
        out["line"] = (Y * S[:, None] if Y.ndim > 1 else Y * S).sum(0) / np.sum(S ** 2)
    return out


res = {}
for key in A:
    pts = points(key)
    net = NET[key]
    r = dict(param=A[key]["param"], du=A[key]["du"], hs=sorted({abs(s) for s in pts if -s in pts}))
    for q in Q:
        adj = estimates({s: np.array([f(q, pts[s][i][q]) for i in range(8)]) for s in pts})
        nv = {s: f(q, net[f"{s:+.3f}"][0][q]) for s in pts if f"{s:+.3f}" in net}
        nest = estimates({s: v for s, v in nv.items() if -s in nv})
        nens = estimates({s: f(q, np.mean([m[q] for m in net[f"{s:+.3f}"]])) if q != "p" else np.log(np.mean([m["p"] for m in net[f"{s:+.3f}"]]))
                          for s in pts if f"{s:+.3f}" in net and f"{-s:+.3f}" in net})
        r[q] = {k: dict(adj=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(8)), net=float(nest.get(k, np.nan)),
                        net_ens=float(nens.get(k, np.nan))) for k, v in adj.items()}
    res[key] = r
json.dump(res, open(FS.F.OUT / "stencils.json", "w"), indent=1)

# consistency of five-point vs central at the same h (truncation) and noise of each estimator
for q in Q:
    print(f"\n=== {q}")
    for name in ("c0.04", "f0.04", "c0.08", "f0.08", "c0.16", "line"):
        rows = [r[q][name] for r in res.values() if name in r[q]]
        if rows:
            print(f"  {name:6} n={len(rows):2d}  median se {np.median([x['se'] for x in rows]):.4f}")
    for h in ("0.04", "0.08"):
        d = [(r[q][f"c{h}"]["adj"] - r[q][f"f{h}"]["adj"]) / r[q][f"c{h}"]["se"] for r in res.values() if f"f{h}" in r[q]]
        if d:
            print(f"  central - five-point at h={h}: n={len(d)} rms in units of central se {np.sqrt(np.mean(np.square(d))):.2f}, mean {np.mean(d):+.2f}")
print("\nper case: available symmetric h")
for k, r in res.items():
    print(f"  {k:28} {r['hs']}")
