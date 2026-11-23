"""Compare replicated adjoint FD (fd/time_derivs.json) with the old 2-pair FD (fd/adjoint.json) and the network (fd/network_time.json).
Writes fd/time_compare.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import numpy as np
import common as C
FD = C.ST / "fd"
A = json.load(open(FD / "time_derivs.json"))
O = {k: v for k, v in json.load(open(FD / "adjoint.json")).items() if v.get("ok")}
N = json.load(open(FD / "network_time.json"))
keys = list(A)
res = {"old_vs_new": {}, "network": {}}
for qo, qn in (("dlnp", "p"), ("dt", "t")):
    a = np.array([A[k][qn] for k in keys]); sa = np.array([A[k][qn + "_se"] for k in keys])
    o = np.array([O[k][qo] for k in keys]); so = np.array([O[k][qo + "_se"] for k in keys])
    pull = (o - a) / np.sqrt(so ** 2 + sa ** 2)
    res["old_vs_new"][qn] = dict(median_se_old=float(np.median(so)), median_se_new=float(np.median(sa)),
                                 se_ratio_old_over_new_median=float(np.median(so / sa)), se_ratio_range=[float((so / sa).min()), float((so / sa).max())],
                                 chi2_old_vs_new=float(np.sum(pull ** 2)), pull_sd=float(np.std(pull)), corr=float(np.corrcoef(a, o)[0, 1]))
for w in ("single", "ensemble"):
    res["network"][w] = {}
    for q in ("p", "t", "sd", "t10"):
        a = np.array([A[k][q] for k in keys]); se = np.array([A[k][q + "_se"] for k in keys]); n = np.array([N[w][k][q] for k in keys])
        res["network"][w][q] = dict(chi2=float(np.sum(((n - a) / se) ** 2)), ndf=len(a), corr=float(np.corrcoef(a, n)[0, 1]),
                                   rms_diff=float(np.sqrt(np.mean((n - a) ** 2))), rms_se=float(np.sqrt(np.mean(se ** 2))),
                                   ref_spread=float(np.std(a)), median_snr=float(np.median(np.abs(a) / se)),
                                   n_sig3=int(np.sum(np.abs(a) / se > 3)),
                                   frac_within_2se=float(np.mean(np.abs(n - a) < 2 * se)),
                                   slope=float(np.sum(a * n / se**2) / np.sum(a * a / se**2)))
# per-parameter group: geometry params (cell_z, width) vs guide path (r_over_w, turn2_x, turn2_y)
grp = {"cell": ["cell_z", "width"], "guide": ["r_over_w", "turn2_x", "turn2_y"]}
res["by_group"] = {}
for g, ps in grp.items():
    ks = [k for k in keys if A[k]["param"] in ps]
    res["by_group"][g] = {}
    for q in ("p", "t", "sd", "t10"):
        a = np.array([A[k][q] for k in ks]); se = np.array([A[k][q + "_se"] for k in ks]); n = np.array([N["single"][k][q] for k in ks])
        res["by_group"][g][q] = dict(n=len(ks), chi2=float(np.sum(((n - a) / se) ** 2)), n_sig3=int(np.sum(np.abs(a) / se > 3)),
                                     rms_ref=float(np.sqrt(np.mean(a ** 2))), rms_net=float(np.sqrt(np.mean(n ** 2))),
                                     rms_diff=float(np.sqrt(np.mean((n - a) ** 2))))
res["table"] = {k: {q: [round(A[k][q], 3), round(A[k][q + "_se"], 3), round(N["single"][k][q], 3)] for q in ("p", "t", "sd", "t10")} |
                {"old": {"dlnp": [round(O[k]["dlnp"], 3), round(O[k]["dlnp_se"], 3)], "dt": [round(O[k]["dt"], 3), round(O[k]["dt_se"], 3)]},
                 "base": A[k]["base"]} for k in keys}
json.dump(res, open(FD / "time_compare.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("old_vs_new", "network", "by_group")}, indent=1))
for k in keys:
    t = res["table"][k]
    print(f"{k:26}", " ".join(f"{q}:{t[q][0]:+.3f}±{t[q][1]:.3f}/{t[q][2]:+.3f}" for q in ("p", "t", "sd", "t10")), "old dt", t["old"]["dt"])
