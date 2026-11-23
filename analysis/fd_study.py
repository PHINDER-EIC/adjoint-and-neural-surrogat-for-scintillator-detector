"""Derivative fidelity: d ln(whole-volume p0) / d u for design parameters, adjoint finite differences vs surrogates.

Designs are perturbed in the atlas's 13-D unit design space u (the GP input) and mapped to physical geometry with the
atlas's own _candidate_from_u, so adjoint, network and GP see identical perturbed designs. Adjoint: central
differences with common random numbers (same Halton launch states and keyed physics stream for +/- runs), two
independent seed pairs -> derivative and its standard error. Results: study/fd/adjoint.json and fd_atlas.h5 (one
geometry group per perturbed design, seed pair A) for the surrogate evaluation (fd_surrogates.py).
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(str(paths.REPO))
T = paths.DATA
sys.path[:0] = [str(ROOT / "Scintillator-Simulator-main/extras/numba_adjoint"), str(ROOT / "phinder_bridge/scripts"),
                str(ROOT / "Scintillator-Simulator-main/src")]
import numba  # noqa: E402
import numba_adjoint as na  # noqa: E402
import phinder_stage6_adjoint_geometry_atlas as A  # noqa: E402

OUT = T / "derived/fd"
OUT.mkdir(parents=True, exist_ok=True)
GEOMS = ["A_train_g0013", "B_train_g0180", "B_val_g0025", "B_train_g0195", "A_train_g0042", "B_train_g0035"]
PARAMS = {"cell_z": 2, "width": 3, "r_over_w": 4, "turn2_x": 6, "turn2_y": 10}
DU = 0.04
import os
DU = float(os.environ.get("FD_DU", DU))
ROOTS = 100000
SEEDS = [(7_700_001, 9_900_001), (7_700_002, 9_900_002)]
ARGS = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005,
                             max_live_branches=64, max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0,
                             start_epsilon_cm=1e-7)
RANGES = A._load_ranges(ROOT / "phinder_bridge/config/stage6B_geometry_atlas_ranges.json")


def raw_from_u(u, gid):
    r = A._candidate_from_u(np.asarray(u, float), RANGES, 0)
    r["id"] = gid
    r["split"] = "test"
    return r


def run(raw, seeds):
    spec, p0, p1, cfg = A._geometry_cfg(raw, ARGS, *seeds)
    tr = na.from_stage6_cfg(cfg)
    t0 = time.perf_counter()
    m, _ = tr.run(ROOTS, *seeds)
    prod = A._compute_products(raw, spec, p0, p1, cfg, m, ARGS, "fd", seeds[0], seeds[1], time.perf_counter() - t0)
    return spec, p0, p1, prod


def main():
    bank = {g["id"]: g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"]}
    res = json.load(open(OUT / "adjoint.json")) if (OUT / "adjoint.json").exists() else {}
    print("threads", numba.get_num_threads(), flush=True)
    for gid in GEOMS:
        u0 = np.asarray(bank[gid]["spacefill_vector"], float)
        chk = raw_from_u(u0, gid)
        assert np.allclose(chk["cell_cm"], bank[gid]["cell_cm"]) and np.allclose(chk["turns_xy_cm"], bank[gid]["turns_xy_cm"]), gid
        for pname, k in PARAMS.items():
            key = f"{gid}|{pname}"
            if key in res and not (os.environ.get("FD_RETRY") and "skipped" in res[key]):
                continue
            us = {}
            for sgn in (+1, -1):
                u = u0.copy(); u[k] += sgn * DU
                if not (0 <= u[k] <= 1):
                    us = None; break
                if k >= 9 and not np.all(np.diff(u[9:13]) >= 0):      # keep the y order statistics ordered
                    us = None; break
                us[sgn] = u
            if us is None:
                res[key] = dict(skipped="out of range / order"); continue
            row = dict(gid=gid, param=pname, k=k, du=DU, u0=u0.tolist(), pvol={}, tvol={}, ok=True)
            try:
                for sgn, u in us.items():
                    raw = raw_from_u(u, f"{gid}__{pname}__{'p' if sgn > 0 else 'm'}")
                    raw["spacefill_vector"] = u.tolist()
                    for si, seeds in enumerate(SEEDS):
                        spec, p0, p1, prod = run(raw, seeds)
                        s = prod["summary"]
                        row["pvol"][f"{sgn}_{si}"] = s["volume_mean_p"]
                        row["tvol"][f"{sgn}_{si}"] = s["volume_mean_transport_time_ns"]
                        if si == 0:
                            with h5py.File(OUT / "fd_atlas.h5", "a") as f:
                                A._write_geometry_group(f, raw, spec, p0, p1, prod, ARGS)
            except Exception as e:  # invalid perturbed geometry (fillet overlap etc.)
                row = dict(skipped=repr(e)[:200])
            if row.get("ok"):
                d = [(np.log(row["pvol"][f"1_{si}"]) - np.log(row["pvol"][f"-1_{si}"])) / (2 * DU) for si in range(2)]
                dt = [(row["tvol"][f"1_{si}"] - row["tvol"][f"-1_{si}"]) / (2 * DU) for si in range(2)]
                row.update(dlnp=float(np.mean(d)), dlnp_se=float(abs(d[0] - d[1]) / 2),
                           dt=float(np.mean(dt)), dt_se=float(abs(dt[0] - dt[1]) / 2))
                print(f"{key:28} dlnp/du {row['dlnp']:+.4f} +- {row['dlnp_se']:.4f}   dt/du {row['dt']:+.3f} +- {row['dt_se']:.3f} ns", flush=True)
            else:
                print(key, "skipped", row.get("skipped"), flush=True)
            res[key] = row
            json.dump(res, open(OUT / "adjoint.json", "w"), indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
