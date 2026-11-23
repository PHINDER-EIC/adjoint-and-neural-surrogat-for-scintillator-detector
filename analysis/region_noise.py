"""Measured standard errors of the adjoint region values used in the Geant4 comparison (regions.json).

Earlier the adjoint error of a region was the mean of its voxel s.e. (whole cell: the tracer's pvol_se). Both are
too large: the mean of voxel errors is not the error of the region mean, and the tracer's s.e. assumes independent
roots while the roots are quasi-random. Here the error is measured from the three independent 2e4-root replicate maps
of each geometry (study/adj/*.npz): for every region (and its mirror image, channel 1) the relative difference of the
region means between replicate pairs gives sigma(2e4); pooled over the 8 geometries per region name, and scaled to
the 2e5-root reference by sqrt(2e4/2e5) (the quasi-random gain beyond N^-1/2 is ignored, which keeps it conservative).
Rewrites the adjoint p0_se/p1_se in regions.json (old values kept as *_se_selfreported) and writes region_noise.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np

import common as C

D = json.load(open(C.ST / "regions.json"))
REG = D["regions"]
acc = {}
for gid, regs in REG.items():
    d = np.load(C.ST / "adj" / f"{gid}.npz")
    sel = np.where((d["N"] == 20000) & (d["steps"] == 5000))[0]
    defs = json.loads(str(np.load(C.ST / "g4/geom" / f"{gid}.npz")["regions_json"]))
    for name, (lo, hi) in defs.items():
        sl = C.region_slices(lo, hi)
        slm = C.region_slices([20 - hi[0], lo[1], 8 - hi[2]], [20 - lo[0], hi[1], 8 - lo[2]])
        for s_ in (sl, slm):
            vals = [d["p"][i][s_].mean() for i in sel]
            for a in range(len(vals)):
                for b in range(a + 1, len(vals)):
                    acc.setdefault(name, []).append(np.log(vals[a] / vals[b]) ** 2 / 2)
rel2e4 = {k: float(np.sqrt(np.mean(v))) for k, v in acc.items()}
scale = np.sqrt(2e4 / 2e5)
rel = {k: v * scale for k, v in rel2e4.items()}
for gid, regs in REG.items():
    for name, row in regs.items():
        a = row["adjoint"]
        for ch in (0, 1):
            a.setdefault(f"p{ch}_se_selfreported", a[f"p{ch}_se"])
            a[f"p{ch}_se"] = a[f"p{ch}"] * rel[name]
json.dump(D, open(C.ST / "regions.json", "w"))
json.dump(dict(rel_se_2e4=rel2e4, rel_se_2e5=rel, n_pairs={k: len(v) for k, v in acc.items()}),
          open(C.ST / "region_noise.json", "w"), indent=1)
for k in rel:
    old = np.median([REG[g][k]["adjoint"]["p0_se_selfreported"] / REG[g][k]["adjoint"]["p0"] for g in REG])
    print(f"{k:14} measured rel. s.e. at 2e5: {100*rel[k]:.3f}%  (2e4: {100*rel2e4[k]:.2f}%; old median {100*old:.2f}%)")
