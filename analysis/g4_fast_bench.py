"""Measured Geant4 cost without per-step Python (g4_fast.py), on all 14 cores, and actual forward maps.

For M = 1e6, 4e6, 1.6e7 photons emitted uniformly over cell B_train_g0180, 14 processes run in parallel (M/14 each).
Wall time = slowest process's transport time (Geant4 initialisation, about 1-2 s per process, reported separately).
The detected photons give a forward map of p (channel 0) whose voxel error against the 2e5-root reference is computed
with the paper's metric (p-weighted rms relative difference, reference noise subtracted). Writes study/g4_fast.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import common as C

GPY = __import__("os").environ.get("G4_PYTHON", __import__("sys").executable)
ENV = dict(os.environ, G4ENSDFSTATEDATA=__import__("os").environ.get("G4ENSDFSTATEDATA", ""))
OUT = C.ST / "g4" / "fast"; OUT.mkdir(parents=True, exist_ok=True)
GID = "B_train_g0180"
NPROC = 14
Ms = [int(float(x)) for x in (sys.argv[1:] or ["1e6", "4e6", "1.6e7"])]

tr = C.truth(); gi = tr["ids"].index(GID)
P = tr["p"][gi]; PSE = tr["p_se"][gi] * C.ref_noise()["k_p"]
w = P / P.sum()
res = json.load(open(C.ST / "g4_fast.json")) if (C.ST / "g4_fast.json").exists() else {}
for M in Ms:
    per = M // NPROC
    t0 = time.perf_counter()
    procs = [subprocess.Popen([GPY, "-u", str(C.ST / "g4_fast.py"), GID, "0,0,0", "20,20,8", str(per), str(1000 + 37 * i + M % 997),
                               str(OUT / f"m{M}_{i}.npz")], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for i in range(NPROC)]
    for p in procs:
        p.wait()
    total = time.perf_counter() - t0
    D = [np.load(OUT / f"m{M}_{i}.npz") for i in range(NPROC)]
    n = sum(int(d["n"]) for d in D)
    beam = max(float(d["wall_s"]) for d in D)
    cnt = np.zeros(3200)
    for d in D:
        sel = d["ch"] == 0
        np.add.at(cnt, d["vox"][sel], 1)
    q = cnt / (n / 3200.0)
    r = (q - P) / P
    err = float(np.sqrt(max(np.sum(w * r ** 2) - np.sum(w * (PSE / P) ** 2), 0)))
    pred = float(np.sqrt(3200 * (1 - P).sum() / (n * P.sum())))
    res[str(M)] = dict(n=n, beam_wall_s=beam, total_wall_s=total, photons_per_s=n / beam, map_err=err, binomial_pred=pred,
                       p_whole=float(cnt.sum() / n), p_whole_ref=float(P.mean()))
    print(f"M={n:.2e}: transport {beam:.1f} s (total {total:.1f} s), {n/beam:.3g} photons/s on {NPROC} cores; "
          f"map error {100*err:.1f}% (binomial {100*pred:.1f}%); whole-cell p {cnt.sum()/n:.5f} vs ref {P.mean():.5f}",
          flush=True)
    json.dump(res, open(C.ST / "g4_fast.json", "w"), indent=1)
