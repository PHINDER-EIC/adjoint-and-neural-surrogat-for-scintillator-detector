"""Agreement of each method with Geant4 (unpolarised, matched configuration) over all emission regions and channels.
Writes study/g4_agreement.json (used by paper_figs.m1 and the Geant4 table). Earlier versions of this file were written
by hand from the same formulas; this script reproduces them (check with --check against the old file).

For each method m and each (geometry, region, channel) with Geant4 data:
  ratio = m / g4,  se = ratio * hypot(se_m / m, se_g4 / g4)          (network: se_m = 0)
  mean  = inverse-variance weighted mean of the ratios +- 1 / sqrt(sum 1/se^2)
  rms_dev = sqrt(mean((ratio - 1)^2)) (unweighted),  chi2 = sum(((ratio - 1) / se)^2), ndf = number of values
  chi2_g4only = the same with the Geant4 error alone
  time: mean and rms of m/g4 - 1 for the channel-0 mean time and 0-60 ns spread
  median_ref_relse = median of hypot(se_m/m, se_g4/g4);  median_g4_relse = median of se_g4/g4
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import numpy as np

import common as C

REF = "g4_unpol"


def stats(regions, method):
    r, s, s_g, tm, ts, rel, relg = [], [], [], [], [], [], []
    for gid, regs in regions.items():
        for name, row in regs.items():
            if method not in row or REF not in row:
                continue
            m, g = row[method], row[REF]
            for ch in (0, 1):
                q, qs = g[f"p{ch}"], g[f"p{ch}_se"]
                v, vs = m[f"p{ch}"], m.get(f"p{ch}_se", 0.0)
                ratio = v / q
                r.append(ratio); s.append(ratio * np.hypot(vs / v, qs / q)); s_g.append(ratio * qs / q)
                rel.append(np.hypot(vs / v, qs / q)); relg.append(qs / q)
            tm.append(m["t_mean"] / g["t_mean"] - 1); ts.append(m["t_sd60"] / g["t_sd60"] - 1)
    r, s, s_g = map(np.array, (r, s, s_g))
    w = 1 / s ** 2
    return dict(n=len(r), mean=float(np.sum(w * r) / w.sum()), mean_se=float(1 / np.sqrt(w.sum())),
                rms_dev=float(np.sqrt(np.mean((r - 1) ** 2))), chi2=float(np.sum(((r - 1) / s) ** 2)),
                chi2_g4only=float(np.sum(((r - 1) / s_g) ** 2)),
                tmean_mean=float(np.mean(tm)), tmean_rms=float(np.sqrt(np.mean(np.square(tm)))),
                tsd_mean=float(np.mean(ts)), tsd_rms=float(np.sqrt(np.mean(np.square(ts)))),
                median_ref_relse=float(np.median(rel)), median_g4_relse=float(np.median(relg)))


def main(path=C.ST / "regions.json", out=C.ST / "g4_agreement.json", check=False):
    regions = json.load(open(path))["regions"]
    res = {m: stats(regions, m) for m in ("adjoint", "jax", "network", "g4_pol")}
    if check:
        old = json.load(open(out))
        for m in res:
            for k in old[m]:
                assert abs(old[m][k] - res[m][k]) < 1e-9 * max(1, abs(old[m][k])), (m, k, old[m][k], res[m][k])
        print("reproduces", out)
        return res
    json.dump(res, open(out, "w"), indent=1)
    for m, v in res.items():
        print(f"{m:8} n={v['n']} mean {v['mean']:.4f}+-{v['mean_se']:.4f} rms {100*v['rms_dev']:.2f}% chi2 {v['chi2']:.1f} "
              f"(g4 only {v['chi2_g4only']:.1f})  median rel se {100*v['median_ref_relse']:.2f}% (g4 {100*v['median_g4_relse']:.2f}%)")
    return res


if __name__ == "__main__":
    main(check="--check" in sys.argv)
