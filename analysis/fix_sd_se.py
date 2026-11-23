"""Standard error of the arrival-time spread (0-60 ns) for the Monte Carlo sources in regions.json.

regions.py used the Gaussian formula se(s) = s / sqrt(2(n-1)). For a distribution with kurtosis kappa (4th central
moment / s^4) the large-sample error is se(s) = s * sqrt((kappa - 1) / (4 n)), which is larger for the long-tailed
arrival-time distributions. kappa is taken from the stored 0.25-ns histogram of each source (regions_hist.npz, bin
centres), so se_new = se_gauss * sqrt((kappa - 1) / 2). Old values kept as t_sd60_se_gauss.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np

import common as C

D = json.load(open(C.ST / "regions.json"))
H = np.load(C.ST / "regions_hist.npz")
tc = (np.arange(240) + 0.5) * 0.25
kap = []
for gid, regs in D["regions"].items():
    for name, row in regs.items():
        for src, v in row.items():
            if "t_sd60_se" not in v:
                continue
            h = H[f"{gid}|{name}|{src}"]
            m = np.sum(h * tc) / h.sum()
            s2 = np.sum(h * (tc - m) ** 2) / h.sum()
            k = np.sum(h * (tc - m) ** 4) / h.sum() / s2 ** 2
            v.setdefault("t_sd60_se_gauss", v["t_sd60_se"])
            v["t_sd60_se"] = v["t_sd60_se_gauss"] * np.sqrt((k - 1) / 2)
            v["t_kurtosis60"] = float(k)
            kap.append(k)
json.dump(D, open(C.ST / "regions.json", "w"))
print(f"kurtosis: median {np.median(kap):.1f}, range {min(kap):.1f}-{max(kap):.1f}; error factor median "
      f"{np.median(np.sqrt((np.array(kap) - 1) / 2)):.2f}")
