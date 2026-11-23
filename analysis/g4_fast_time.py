"""Voxel error of the mean arrival time (t-bar) of the Geant4 forward maps in g4_fast.json (B_train_g0180, channel 0),
with the paper's metric (p-weighted rms relative difference, reference noise subtracted), and its prediction from the
number of detected photons per voxel: eps^2 = sum_v w_v (s_t,v / t_v)^2 / n_v, n_v = M p_v / 3200.
Voxels with no detected photon have no t-bar; they are counted with weight but residual 1 (100%) so that the metric
does not improve by dropping them (the fraction is reported). Adds 't_err', 't_pred', 'empty_w' to g4_fast.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import numpy as np
import common as C

GID = "B_train_g0180"
tr = C.truth(); gi = tr["ids"].index(GID)
P, T, S = tr["p"][gi], tr["mt"][gi], tr["tsd"][gi]
w = P / P.sum()
noise = C.truth_time_noise()["mt"]
res = json.load(open(C.ST / "g4_fast.json"))
for key, r in res.items():
    M = int(key); n = 0; cnt = np.zeros(3200); tsum = np.zeros(3200)
    for i in range(14):
        d = np.load(C.ST / "g4" / "fast" / f"m{M}_{i}.npz"); n += int(d["n"])
        sel = d["ch"] == 0
        np.add.at(cnt, d["vox"][sel], 1); np.add.at(tsum, d["vox"][sel], d["t"][sel])
    tb = np.where(cnt > 0, tsum / np.maximum(cnt, 1), np.nan)
    rr = np.where(cnt > 0, (tb - T) / T, 1.0)
    err = float(np.sqrt(max(np.sum(w * rr ** 2) - noise ** 2, 0)))
    nv = n * P / 3200
    pred = float(np.sqrt(np.sum(w * (S / T) ** 2 / nv)))
    r.update(t_err=err, t_pred=pred, empty_w=float(np.sum(w[cnt == 0])))
    print(f"M={n:.2e}: t-bar error {100*err:.2f}% (prediction {100*pred:.2f}%), weight of empty voxels {100*r['empty_w']:.2f}%")
json.dump(res, open(C.ST / "g4_fast.json", "w"), indent=1)
