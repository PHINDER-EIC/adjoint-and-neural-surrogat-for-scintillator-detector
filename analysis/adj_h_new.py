"""2e4-root adjoint maps WITH arrival-time histograms for the 40 new designs (the convergence runs in adj_new/ keep only
moments). Same settings as the atlas (max_steps 5000 as in adj_conv). Seeds independent of the reference and adj_new.
Writes study/adj_new_h/<gid>.npz with p [20,20,8] and h [20,20,8,240] (per emitted photon per ns)."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import time

import numpy as np

import adj_conv as AC

OUT = AC.T / "derived/adj_new_h"
OUT.mkdir(parents=True, exist_ok=True)
bank = [g for g in json.load(open(AC.T / "banks/bank.json"))["geometries"] if g["id"].startswith("test_")]
for gi, raw in enumerate(bank):
    f = OUT / f"{raw['id']}.npz"
    if f.exists():
        continue
    a = AC.args_for(5000)
    ls, ps = 7_800_000 + gi, 9_800_000 + gi
    spec, p0, p1, cfg = AC.atlasmod._geometry_cfg(raw, a, ls, ps)
    tr = AC.na.from_stage6_cfg(cfg)
    t0 = time.perf_counter(); m, _ = tr.run(20000, ls, ps); el = time.perf_counter() - t0
    prod = AC.atlasmod._compute_products(raw, spec, p0, p1, cfg, m, a, "study", ls, ps, el)
    np.savez_compressed(f, p=prod["p"].astype(np.float32), h=prod["h"].astype(np.float32))
    print(raw["id"], f"{el:.1f}s", prod["h"].shape, flush=True)
print("ALLDONE", flush=True)
