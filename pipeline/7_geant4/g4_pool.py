"""Run Geant4 chunks (100k photons each) with a worker count read from g4/workers.txt (re-read before every launch).

Tasks (in order): for the 8 JAX geometries, modes pol and unpol over the 5 JAX regions
(whole 4 chunks, blocks 2 chunks each), then mode full (whole 4, hotspot 2); then whole-volume voxel maps
(list in g4/maps.txt: "gid mode chunks" per line). Existing outputs are skipped, so it can be restarted.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import hashlib
import shutil
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
G4 = Path(__import__("os").environ.get("PHADJ_WORK", paths.RELEASE / "work")) / "g4"
(G4 / "logs").mkdir(parents=True, exist_ok=True)
if not (G4 / "maps.txt").exists():
    shutil.copy(H / "maps.txt", G4 / "maps.txt")
RUNS = G4 / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
GPY = os.environ.get("G4_PYTHON", sys.executable)   # Python with geant4_pybind 0.1.3 (Geant4 11.4.1)
ENV = dict(os.environ,   # G4ENSDFSTATEDATA must point at the G4ENSDFSTATE3.0 dataset
           GEANT4_DATA_DIR=str(G4 / "nodata"))
GEOMS = ["B_train_g0180", "A_train_g0013", "B_val_g0025", "B_train_g0195", "A_train_g0052", "A_train_g0042",
         "B_train_g0035", "B_val_g0028"]
CHUNK = 100000


def tasks():
    out = []
    for gid in GEOMS:
        for mode, regs in (("unpol", None), ("pol", None), ("full", ("whole", "hotspot"))):
            R = json.loads(str(np.load(G4 / "geom" / f"{gid}.npz")["regions_json"]))
            for name, (lo, hi) in R.items():
                if regs and name not in regs:
                    continue
                for c in range(4 if name == "whole" else 2):
                    out.append((gid, mode, name, lo, hi, c))
    mp = G4 / "maps.txt"
    if mp.exists():
        for line in mp.read_text().split("\n"):
            if line.strip():
                gid, mode, n = line.split()
                for c in range(int(n)):
                    out.append((gid, mode, "map", [0, 0, 0], [20, 20, 8], c))
    return out


def workers():
    try:
        return max(0, int((G4 / "workers.txt").read_text().strip()))
    except Exception:
        return 2


def main():
    running = {}
    t0 = time.time()
    done = 0
    while True:
        todo = [t for t in tasks() if not (RUNS / f"{t[0]}__{t[1]}__{t[2]}__{t[5]:03d}.npz").exists()
                and f"{t[0]}__{t[1]}__{t[2]}__{t[5]:03d}" not in running]
        for k, p in list(running.items()):
            if p.poll() is not None:
                done += 1
                del running[k]
                if p.returncode:
                    print(f"FAILED {k} rc={p.returncode}", flush=True)
        if not todo and not running:
            break
        while todo and len(running) < workers():
            gid, mode, name, lo, hi, c = todo.pop(0)
            tag = f"{gid}__{mode}__{name}__{c:03d}"
            seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
            log = open(G4 / "logs" / f"{tag}.log", "w")
            running[tag] = subprocess.Popen([GPY, "-u", str(H / "g4_sim.py"), gid, mode, ",".join(map(str, lo)), ",".join(map(str, hi)),
                                             str(CHUNK), str(seed), str(RUNS / f"{tag}.npz")], stdout=log, stderr=subprocess.STDOUT, env=ENV)
        print(f"{time.strftime('%H:%M')} running {len(running)} done {done} todo {len(todo)} elapsed {(time.time()-t0)/60:.0f} min", flush=True)
        time.sleep(30)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    (G4 / "logs").mkdir(exist_ok=True)
    main()
