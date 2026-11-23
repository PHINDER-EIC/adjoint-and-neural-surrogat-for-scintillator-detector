"""Whole-cell sensitivity-to-error ratios on the 80 test geometries (Sec. VI D): spread of the whole-cell mean arrival
time t-bar (p-weighted mean of voxel mean times, uniform emission) and of ln p-bar across the geometries, against the
network's rms error on them (each of the three networks; mean of the three errors). Also re-reads the guide-length
regression of diag_review2.json. Writes study/ratio80.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np
import torch

import common as C
import diag_review2_lib as D

out = {}
T = D.load(C.TRUTH_H5, D.test_ids); B = D.load(D.BL, D.blind_ids)
tref = np.r_[(T["p"] * T["mt"]).sum(1) / T["p"].sum(1), (B["p"] * B["mt"]).sum(1) / B["p"].sum(1)]
lpref = np.log(np.r_[T["p"].mean(1), B["p"].mean(1)])
et, ep = [], []
for m in D.models:
    tn, lp = [], []
    for atl, ids in ((C.TRUTH_H5, D.test_ids), (D.BL, D.blind_ids)):
        pr = D.predict(atl, ids, m)
        tn.append((pr["p0"] * pr["mt"]).sum(1) / pr["p0"].sum(1)); lp.append(np.log(pr["p0"].mean(1)))
    et.append(float(np.sqrt(np.mean((np.concatenate(tn) - tref) ** 2))))
    ep.append(float(np.sqrt(np.mean((np.concatenate(lp) - lpref) ** 2))))
out = dict(tbar_sd_ns=float(tref.std()), tbar_net_err_ns=et, lnp_sd=float(lpref.std()), lnp_net_err=ep)
out["ratio_t"] = out["tbar_sd_ns"] / np.mean(et); out["ratio_lnp"] = out["lnp_sd"] / np.mean(ep)
g = json.load(open(C.ST / "diag_review2.json"))["tbar_regression"]
out["guide_length_slope_ns_per_cm"] = {k: v[3] for k, v in g["coef"].items()}
json.dump(out, open(C.ST / "ratio80.json", "w"), indent=1)
print(json.dumps(out, indent=1))
