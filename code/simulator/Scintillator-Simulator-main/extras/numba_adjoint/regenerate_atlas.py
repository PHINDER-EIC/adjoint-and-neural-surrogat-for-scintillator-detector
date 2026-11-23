"""Regenerate the Stage-6B adjoint geometry atlas with the numba reverse tracer.

For every geometry in the bank:
  1. trace N roots with max_steps = STEPS (default 5000) and write the atlas group
     (the atlas's own _compute_products / _write_geometry_group, so the file format is unchanged);
  2. trace the SAME roots with the atlas default max_steps = 750. Draws are keyed by
     (root, branch, step), so run 2 is run 1 cut at step 750 and the shift between them
     is the step-cap bias with almost no noise.

Resumable: geometries already in the output file are skipped.
Self-pausing: exits with code 3 ("PAUSED") when the projected total exceeds
--pause-total-min, or one geometry takes longer than --pause-geometry-s.

Usage:
  python regenerate_atlas.py <out_dir> [--bank bank.json] [--baseline-steps 750|0] [--roots 20000] [--steps 5000] [--pause-total-min 40] [--pause-geometry-s 120]
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "phinder_bridge/scripts"), str(ROOT / "Scintillator-Simulator-main/src")]

import h5py  # noqa: E402
import numba  # noqa: E402
import numba_adjoint as na  # noqa: E402
import phinder_stage6_adjoint_geometry_atlas as atlasmod  # noqa: E402

BANK = ROOT / "phinder_bridge/config/stage6B_geometry_bank_default.json"
SEEDS = {"train": (606061, 808081), "val": (606171, 808191), "test": (606271, 808291)}   # atlas defaults


def atlas_args(max_steps):
    return types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0,
                                 split_min_child_weight=0.005, max_live_branches=64, max_branch_nodes=1024,
                                 max_steps=int(max_steps), max_flight_time_ns=80.0, start_epsilon_cm=1e-7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--roots", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--baseline-steps", type=int, default=750, help="0 skips the paired baseline pass")
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--pause-total-min", type=float, default=40.0)
    ap.add_argument("--pause-geometry-s", type=float, default=120.0)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = a.out_dir / "stage6B_adjoint_geometry_atlas_numba.h5"
    log = a.out_dir / "shift_vs_750.jsonl"

    bank = atlasmod._load_bank(a.bank)
    geoms = bank["geometries"]
    done = set()
    if out_h5.exists():
        with h5py.File(out_h5, "r") as f:
            done = set(f["geometries"].keys()) if "geometries" in f else set()
    todo = [g for g in geoms if str(g["id"]) not in done]
    print(f"bank={a.bank.name} geometries={len(geoms)} done={len(done)} todo={len(todo)} roots={a.roots} "
          f"steps={a.steps} baseline={a.baseline_steps} threads={numba.get_num_threads()}", flush=True)

    args_main, args_base = atlas_args(a.steps), atlas_args(max(a.baseline_steps, 1))
    t_start = time.perf_counter()
    for i, raw in enumerate(todo, 1):
        gid, split = str(raw["id"]), str(raw["split"])
        ls, ps = SEEDS[split]
        t0 = time.perf_counter()
        spec, p0, p1, cfg = atlasmod._geometry_cfg(raw, args_main, ls, ps)
        tr = na.from_stage6_cfg(cfg)
        t_build = time.perf_counter() - t0
        t1 = time.perf_counter()
        m, _ = tr.run(a.roots, ls, ps)
        t_main = time.perf_counter() - t1
        fp = atlasmod._fingerprint(raw, cfg, args_main, ls, ps)
        prod = atlasmod._compute_products(raw, spec, p0, p1, cfg, m, args_main, fp, ls, ps, t_main)

        # paired baseline: same roots, atlas default step cap
        if a.baseline_steps > 0:
            cfg_b = dict(cfg, max_steps=int(a.baseline_steps))
            tr_b = na.from_stage6_cfg(cfg_b)
            t2 = time.perf_counter()
            mb, _ = tr_b.run(a.roots, ls, ps)
            t_base = time.perf_counter() - t2
            prod_b = atlasmod._compute_products(raw, spec, p0, p1, cfg_b, mb, args_base, fp, ls, ps, t_base)
        else:                                   # no baseline: shift columns read 0
            mb, prod_b, t_base = m, prod, 0.0

        with h5py.File(out_h5, "a") as f:
            f.require_group("geometries")
            atlasmod._write_geometry_group(f, raw, spec, p0, p1, prod, args_main)
            g = f["geometries"][gid]
            g.attrs["tracer"] = "extras/numba_adjoint"
            g.attrs["max_steps"] = int(a.steps)
            atlasmod._rebuild_index(f)

        s, sb = prod["summary"], prod_b["summary"]
        p, pb = prod["p"], prod_b["p"]
        ok = p > 0
        vox = (pb[ok] - p[ok]) / p[ok]
        row = dict(
            id=gid, split=split, roots=a.roots, failed=m["failed"], failed_baseline=mb["failed"],
            p_vol=s["volume_mean_p"], p_vol_se=s["volume_mean_p_se"],
            p_vol_750=sb["volume_mean_p"], p_shift_rel=sb["volume_mean_p"] / s["volume_mean_p"] - 1.0,
            t_vol_ns=s["volume_mean_transport_time_ns"], t_vol_750_ns=sb["volume_mean_transport_time_ns"],
            t_shift_rel=sb["volume_mean_transport_time_ns"] / s["volume_mean_transport_time_ns"] - 1.0,
            voxel_shift_median=float(np.median(vox)) if vox.size else float("nan"),
            voxel_shift_p05=float(np.quantile(vox, 0.05)) if vox.size else float("nan"),
            voxel_relSE_median=s["voxel_relative_se_median"], voxel_relSE_p90=s["voxel_relative_se_p90"],
            unresolved=s["unresolved_weight_per_root"], unresolved_750=sb["unresolved_weight_per_root"],
            reach_scint=s["reached_scintillator_fraction"], splits_per_root=s["deterministic_splits_per_root"],
            events_per_root=m["boundary_events"] / max(m["n"], 1),
            cell_cm=[float(x) for x in spec.cell_cm], width_cm=float(spec.width_cm), radius_cm=float(spec.radius_cm),
            n_volumes=len(tr.scene.volumes),
            s_build=round(t_build, 2), s_main=round(t_main, 2), s_baseline=round(t_base, 2))
        with open(log, "a") as fh:
            fh.write(json.dumps(row) + "\n")

        el = time.perf_counter() - t_start
        dt_geo = time.perf_counter() - t0
        proj = el / i * len(todo)
        print(f"[{len(done)+i}/{len(geoms)}] {gid:>10} {split:5} vols={row['n_volumes']:3d} "
              f"geo={dt_geo:5.1f}s (main {t_main:4.1f}, base {t_base:4.1f})  p={row['p_vol']:.5f} "
              f"shift750 p={100*row['p_shift_rel']:+.2f}% t={100*row['t_shift_rel']:+.2f}%  "
              f"noise={100*row['voxel_relSE_median']:.1f}%  elapsed={el/60:.1f} min proj={proj/60:.1f} min", flush=True)
        if i < len(todo) and (proj / 60 > a.pause_total_min or dt_geo > a.pause_geometry_s):
            print(f"PAUSED after {gid}: projected total {proj/60:.1f} min (limit {a.pause_total_min}), "
                  f"this geometry {dt_geo:.0f} s (limit {a.pause_geometry_s:.0f}). Re-run to resume.", flush=True)
            sys.exit(3)
    print(f"DONE {len(todo)} geometries in {(time.perf_counter()-t_start)/60:.1f} min -> {out_h5}", flush=True)


if __name__ == "__main__":
    main()
