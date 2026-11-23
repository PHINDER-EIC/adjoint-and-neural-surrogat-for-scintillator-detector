"""Run g4_ev_opt.py (mode unpol) on every photon chunk with a fixed number of parallel workers; restartable."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import os
import subprocess
import sys
import time
from pathlib import Path

H = Path(__import__("os").environ.get("PHADJ_EVENTS", paths.DATA / "derived" / "events"))
GPY = os.environ.get("G4_PYTHON", sys.executable)
ENV = dict(os.environ, G4ENSDFSTATEDATA=__import__("os").environ.get("G4ENSDFSTATEDATA", ""),
           GEANT4_DATA_DIR=str(H.parent / "g4" / "nodata"))
W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
todo = [c for c in sorted((H / "chunks").glob("c*.npz")) if not (H / "opt" / c.name).exists()]
run = {}
t0 = time.time(); done = 0; total = len(todo)
(H / "logs").mkdir(exist_ok=True)
while todo or run:
    for k, p in list(run.items()):
        if p.poll() is not None:
            done += 1; del run[k]
            if p.returncode:
                print("FAILED", k, p.returncode, flush=True)
    while todo and len(run) < W:
        c = todo.pop(0)
        seed = 1000 + int(c.stem[1:])
        log = open(H / "logs" / f"{c.stem}.log", "w")
        run[c.stem] = subprocess.Popen([GPY, "-u", str(Path(__file__).resolve().parent / "g4_ev_opt.py"), "B_train_g0180", "unpol", str(c), "0", "1000000000",
                                        str(seed), str(H / "opt" / c.name)], stdout=log, stderr=subprocess.STDOUT, env=ENV)
    print(f"{time.strftime('%H:%M:%S')} done {done}/{total} running {len(run)} elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    time.sleep(20)
print("ALLDONE", flush=True)
