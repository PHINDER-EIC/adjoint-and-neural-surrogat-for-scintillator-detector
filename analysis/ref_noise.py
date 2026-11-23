"""Measure the noise of the 2e5-root reference maps without using the reference.

The convergence runs (study/adj/*.npz) contain three independent replicate maps per N for every test geometry.
For two replicates i, j the p-weighted mean square of ln(p_i/p_j) is 2 sigma(N)^2, so sigma(N) is measured directly;
the same for the relative difference of t-bar and s_t, and for the whole-cell totals. The tracer's self-reported s.e.
assumes independent roots; the roots are scrambled-Halton, so it is too large. sigma(N) sqrt(N) = a(N) falls slowly
with N; a power law a(N) = a0 N^-delta fitted for 1e3 <= N <= 2e4 is extrapolated to the reference's 2e5 roots.
Writes study/ref_noise.json (used by common.pmetrics and common.truth_time_noise).
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import glob
import json

import numpy as np

import common as C

NREF = 2e5
tr = C.truth()
acc = {}
for f in sorted(glob.glob(str(C.ST / "adj" / "*.npz"))):
    d = np.load(f)
    gid = f.replace("\\", "/").split("/")[-1][:-4]
    g = tr["ids"].index(gid)
    w = tr["p"][g] / tr["p"][g].sum()
    for N in np.unique(d["N"]):
        sel = np.where((d["N"] == N) & (d["steps"] == 5000))[0]
        for a_ in range(len(sel)):
            for b_ in range(a_ + 1, len(sel)):
                i, j = sel[a_], sel[b_]
                pi, pj = d["p"][i].ravel(), d["p"][j].ravel()
                A = acc.setdefault(int(N), dict(p=[], p_self=[], mt=[], tsd=[], vol=[], vol_self=[]))
                A["p"].append(np.sum(w * np.log(pi / pj) ** 2) / 2)
                A["p_self"].append(np.sum(w * ((d["p_se"][i].ravel() / pi) ** 2 + (d["p_se"][j].ravel() / pj) ** 2)) / 2)
                for key in ("mt", "tsd"):
                    x, y = d[key][i].ravel(), d[key][j].ravel()
                    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
                    r = np.where(ok, (x - y) / np.where(ok, y, 1), 0)
                    A[key].append(np.sum(w * r ** 2) / 2)
                A["vol"].append(np.log(d["pvol"][i] / d["pvol"][j]) ** 2 / 2)
                A["vol_self"].append(((d["pvol_se"][i] / d["pvol"][i]) ** 2 + (d["pvol_se"][j] / d["pvol"][j]) ** 2) / 2)

Ns = np.array(sorted(acc))
table = {int(N): {k: float(np.sqrt(np.mean(v))) for k, v in acc[N].items()} for N in Ns}
fit = (Ns >= 1000) & (Ns <= 20000)
out = dict(n_ref=NREF, per_N=table)
for key in ("p", "mt", "tsd", "vol"):
    s = np.array([table[N][key] for N in Ns])
    a = s * np.sqrt(Ns)
    slope, icpt = np.polyfit(np.log(Ns[fit]), np.log(a[fit]), 1)
    out[f"delta_{key}"] = float(-slope)
    out[key] = float(np.exp(icpt) * NREF ** slope / np.sqrt(NREF))
# self-reported voxel noise of the reference, test set, and the ratio used to rescale it
rel = tr["p_se"] / tr["p"]
w = tr["p"] / tr["p"].sum()
out["p_self_reported"] = float(np.sqrt(np.sum(w * rel ** 2)))
out["k_p"] = out["p"] / out["p_self_reported"]
out["vol_self_reported_median"] = float(np.median(tr["pvol_se"] / tr["pvol"]))
json.dump(out, open(C.ST / "ref_noise.json", "w"), indent=1)
for N in Ns:
    t = table[N]
    print(f"N={N:6d}  p {100*t['p']:.3f}% (self {100*t['p_self']:.3f}%, ratio {t['p']/t['p_self']:.3f})  "
          f"mt {100*t['mt']:.3f}%  tsd {100*t['tsd']:.3f}%  whole cell {100*t['vol']:.3f}% (self {100*t['vol_self']:.3f}%)")
print({k: v for k, v in out.items() if k != "per_N"})
