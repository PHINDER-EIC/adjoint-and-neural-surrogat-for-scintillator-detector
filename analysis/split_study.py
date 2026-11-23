"""Sensitivity of the reverse tracer to the Fresnel-split threshold (split when both children weigh >= threshold,
else one analog child). threshold > 0.5 never splits (pure analog). 20k roots, 2 independent seed pairs, 4 test
geometries; scored against the 200k-root reference (default threshold 0.005). Writes study/split.json.
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
sys.path[:0] = [str(T / "derived"), str(ROOT / "Scintillator-Simulator-main/extras/numba_adjoint"), str(ROOT / "phinder_bridge/scripts"),
                str(ROOT / "Scintillator-Simulator-main/src")]
import numba_adjoint as na  # noqa: E402
import phinder_stage6_adjoint_geometry_atlas as A  # noqa: E402
import common as C  # noqa: E402

GEOMS = ["A_train_g0013", "B_train_g0180", "B_val_g0025", "B_train_g0195"]
THR = [0.0005, 0.005, 0.05, 0.25, 1.01]
SEEDS = [(5_100_001, 5_200_001), (5_100_002, 5_200_002)]


def main():
    tr0 = C.truth()
    bank = {g["id"]: g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"]}
    out = []
    for gid in GEOMS:
        gi = tr0["ids"].index(gid)
        pt, pse = tr0["p"][gi], tr0["p_se"][gi]
        w = pt / pt.sum()
        for thr in THR:
            for si, seeds in enumerate(SEEDS):
                a = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=thr,
                                          max_live_branches=64, max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0,
                                          start_epsilon_cm=1e-7)
                spec, p0, p1, cfg = A._geometry_cfg(bank[gid], a, *seeds)
                tr = na.from_stage6_cfg(cfg)
                tr.run(200, *seeds)                      # warm caches
                t0 = time.perf_counter()
                m, _ = tr.run(20000, *seeds)
                el = time.perf_counter() - t0
                prod = A._compute_products(bank[gid], spec, p0, p1, cfg, m, a, "split", seeds[0], seeds[1], el)
                p = prod["p"].ravel()
                r = np.log(np.maximum(p, 1e-12) / pt)
                err2 = float(np.sum(w * r ** 2) - np.sum(w * (pse / pt) ** 2))
                s = prod["summary"]
                row = dict(gid=gid, thr=thr, seed=si, wall_s=el, p_rms=float(np.sqrt(max(err2, 0))), p_bias=float(np.sum(w * r)),
                           pvol_rel=s["volume_mean_p"] / tr0["pvol"][gi] - 1, splits_per_root=s["deterministic_splits_per_root"],
                           fallbacks_per_root=s["analog_fallbacks_per_root"], selfse=s["voxel_relative_se_median"])
                out.append(row)
                print(f"{gid} thr={thr:<7} seed{si} {el:6.2f}s p_rms {100*row['p_rms']:.2f}% bias {100*row['p_bias']:+.2f}% "
                      f"vol {100*row['pvol_rel']:+.2f}% splits/root {row['splits_per_root']:.1f}", flush=True)
                json.dump(out, open(T / "derived/split.json", "w"), indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
