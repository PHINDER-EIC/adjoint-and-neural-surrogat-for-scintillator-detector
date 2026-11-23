"""Geant4 (g4_fast.py, no Python in the photon loop) for the Fig. 13 task on this CPU: N = 1e3 ... 1e6 photons from one
voxel at the cell centre of B_train_g0180 (voxel 10,10,4), one process and 14 processes (N/14 each).
transport_s = BeamOn time (slowest process); total_s = wall time including Geant4 start-up and geometry construction.
Appends 'point_scan_g4' to study/numba_fwd.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os
import subprocess
import time

import numpy as np

import common as C

GPY = __import__("os").environ.get("G4_PYTHON", __import__("sys").executable)
ENV = dict(os.environ, G4ENSDFSTATEDATA=__import__("os").environ.get("G4ENSDFSTATEDATA", ""))
OUT = C.ST / "g4" / "point"; OUT.mkdir(parents=True, exist_ok=True)
GID = "B_train_g0180"
res = {}
for nproc, Ns in ((1, (1000, 10000, 100000, 1000000)), (14, (100000, 1000000))):
    for N in Ns:
        per = int(np.ceil(N / nproc / 1000) * 1000)
        t0 = time.perf_counter()
        ps = [subprocess.Popen([GPY, "-u", str(C.ST / "g4_fast.py"), GID, "10,10,4", "11,11,5", str(per), str(500 + 13 * i),
                                str(OUT / f"p{nproc}_{N}_{i}.npz")], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
              for i in range(nproc)]
        for p in ps:
            p.wait()
        total = time.perf_counter() - t0
        D = [np.load(OUT / f"p{nproc}_{N}_{i}.npz") for i in range(nproc)]
        n = sum(int(d["n"]) for d in D)
        tr = max(float(d["wall_s"]) for d in D)
        caught = sum(int(np.sum(d["ch"] >= 0)) for d in D) / n
        res[f"{nproc}_{N}"] = dict(processes=nproc, N=n, transport_s=tr, total_s=total, caught=caught)
        print(f"Geant4 point source, {nproc:2d} processes, N={n:.0e}: transport {tr:.3f} s, total {total:.2f} s, "
              f"caught {caught:.4f}", flush=True)
f = C.ST / "numba_fwd.json"
d = json.load(open(f)); d["point_scan_g4"] = res; json.dump(d, open(f, "w"), indent=1)
