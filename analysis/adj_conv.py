"""Adjoint (numba) convergence study on the 40 test geometries.

For each geometry: independent runs (own Halton scramble + physics seed) at N roots in NS, REPS replicates
(one for 50k), plus one 20k run with the repo default max_steps = 750. Each run is reduced with the atlas's own
_compute_products; we keep per voxel p, p_se, mean time, time s.d. (from the 0-60 ns histogram), the whole-volume
values and the wall time. Truth for scoring = the 200,000-root maps (other seeds).

Resumable: one npz per geometry in study/adj/.
usage: python adj_conv.py [gid ...]
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(str(paths.REPO))
T = paths.DATA
sys.path[:0] = [str(ROOT / "Scintillator-Simulator-main/extras/numba_adjoint"), str(ROOT / "phinder_bridge/scripts"),
                str(ROOT / "Scintillator-Simulator-main/src")]
import numba  # noqa: E402
import numba_adjoint as na  # noqa: E402
import phinder_stage6_adjoint_geometry_atlas as atlasmod  # noqa: E402

OUT = T / "derived/adj"
OUT.mkdir(parents=True, exist_ok=True)
NS = [250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
REPS = 3
TC = (np.arange(240) + 0.5) * 0.25


def args_for(steps):
    return types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0,
                                 split_min_child_weight=0.005, max_live_branches=64, max_branch_nodes=1024,
                                 max_steps=int(steps), max_flight_time_ns=80.0, start_epsilon_cm=1e-7)


def tsd(h):
    m0 = h.sum(-1)
    m1 = (h * TC).sum(-1)
    m2 = (h * TC ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = m1 / m0
        return np.sqrt(np.maximum(m2 / m0 - mu ** 2, 0.0)), mu


def one(raw, n, ls, ps, steps):
    a = args_for(steps)
    spec, p0, p1, cfg = atlasmod._geometry_cfg(raw, a, ls, ps)
    tr = na.from_stage6_cfg(cfg)
    t0 = time.perf_counter()
    m, _ = tr.run(n, ls, ps)
    el = time.perf_counter() - t0
    prod = atlasmod._compute_products(raw, spec, p0, p1, cfg, m, a, "study", ls, ps, el)
    sd, mu_h = tsd(prod["h"])
    s = prod["summary"]
    return dict(p=prod["p"].astype(np.float32), p_se=prod["p_se"].astype(np.float32),
                mt=prod["mean_t"].astype(np.float32), tsd=sd.astype(np.float32), mt_h=mu_h.astype(np.float32),
                pvol=s["volume_mean_p"], pvol_se=s["volume_mean_p_se"], tvol=s["volume_mean_transport_time_ns"],
                wall=el, failed=int(m["failed"]) if "failed" in m else 0, unresolved=s["unresolved_weight_per_root"])


def main():
    bank = json.load(open(T / "banks/bank_test40.json"))["geometries"]
    want = set(sys.argv[1:])
    print(f"threads={numba.get_num_threads()}", flush=True)
    t_all = time.perf_counter()
    for gi, raw in enumerate(bank):
        gid = raw["id"]
        if (want and gid not in want) or (OUT / f"{gid}.npz").exists():
            continue
        tg = time.perf_counter()
        runs = []   # (N, rep, steps)
        for n in NS:
            for r in range(1 if n >= 50000 else REPS):
                runs.append((n, r, 5000))
        runs.append((20000, 0, 750))
        rec = {k: [] for k in ("N", "rep", "steps", "pvol", "pvol_se", "tvol", "wall", "unresolved")}
        maps = {k: [] for k in ("p", "p_se", "mt", "tsd", "mt_h")}
        for (n, r, steps) in runs:
            ls, ps = 7_000_000 + 1000 * r + gi, 9_000_000 + 1000 * r + gi + n   # independent of truth seeds
            o = one(raw, n, ls, ps, steps)
            for k in rec:
                rec[k].append({"N": n, "rep": r, "steps": steps}.get(k, o.get(k)))
            for k in maps:
                maps[k].append(o[k])
        np.savez_compressed(OUT / f"{gid}.npz", **{k: np.array(v) for k, v in rec.items()},
                            **{k: np.stack(v) for k, v in maps.items()})
        w = np.array(rec["wall"]); N = np.array(rec["N"])
        print(f"{gid} done in {time.perf_counter()-tg:.0f}s; wall at 1k {w[N==1000].mean():.2f}s 20k {w[(N==20000)][:3].mean():.2f}s "
              f"50k {w[N==50000].mean():.2f}s; elapsed {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
