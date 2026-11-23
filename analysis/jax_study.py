"""Forward JAX (engine B) runs for the speed/accuracy study.

For one geometry of the 40-geometry test set, photons are emitted isotropically at t = 0, uniformly inside
five voxel-aligned regions of the 20x20x8 atlas grid. For each region the script stores the caught counts per
batch (for an empirical standard error), the arrival times of caught photons in both channels, the number of
truncated photons, and wall/CPU time. Analysis (adjoint, surrogate) is done elsewhere from the voxel ranges.

usage: python jax_study.py <geometry_id> [n_whole] [n_block] [batch]
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
OUT = T / "derived/jax"
sys.path[:0] = [str(ROOT / "phinder_bridge/scripts"), str(ROOT / "Scintillator-Simulator-main/src"),
                str(ROOT / "Scintillator-Simulator-main/extras/jax_compaction")]
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import h5py  # noqa: E402
import phinder_stage6_adjoint_geometry_atlas as atlasmod  # noqa: E402
import phinder_stage6_adjoint_prototype as proto  # noqa: E402
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch  # noqa: E402
from Photon_Simulator.core.photonsEngines.JaxEngine import JaxApproximateEngine, JaxTrackingConfig  # noqa: E402
from compaction import track_photon_batch_compacted  # noqa: E402

gid = sys.argv[1]
n_whole = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
n_block = int(sys.argv[3]) if len(sys.argv) > 3 else 25000
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
OUT.mkdir(parents=True, exist_ok=True)
TRUTH = T / "reference_test40_2e5roots.h5"

raw = next(g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"] if g["id"] == gid)
args = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005,
                             max_live_branches=64, max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0,
                             start_epsilon_cm=1e-7)
spec, _, _, cfg = atlasmod._geometry_cfg(raw, args, 1, 2)
scene, _, _, _ = proto._build_from_cfg(cfg)
X, Y, Z = map(float, cfg["cell_cm"])
sp = np.array([X / 20, Y / 20, Z / 8])

with h5py.File(TRUTH) as f:
    pA = f["geometries"][gid]["field/p_per_emitted"][:]
    gp = f["geometries"][gid]["guide0_points_cm"][:]
# hotspot: 2x2 block at the top layer around the highest-p voxel
ix, iy = np.unravel_index(np.argmax(pA[:, :, 7]), (20, 20))
ix, iy = int(min(max(ix, 0), 18)), int(min(max(iy, 0), 18))
# corner: the 2x2 xy corner farthest from every guide-0 point, top two layers
best, cx, cy = -1.0, 0, 0
for (a, b) in ((0, 0), (0, 18), (18, 0), (18, 18)):
    c = np.array([(a + 1) * sp[0], (b + 1) * sp[1]])
    d = np.min(np.linalg.norm(gp[:, :2] - c, axis=1))
    if d > best:
        best, cx, cy = d, a, b
REGIONS = {   # name: (voxel lo, voxel hi) exclusive, photons
    "whole": ((0, 0, 0), (20, 20, 8), n_whole),
    "hotspot": ((ix, iy, 7), (ix + 2, iy + 2, 8), n_block),
    "centre": ((9, 9, 3), (11, 11, 5), n_block),
    "below_hotspot": ((ix, iy, 0), (ix + 2, iy + 2, 1), n_block),
    "far_corner": ((cx, cy, 6), (cx + 2, cy + 2, 8), n_block),
}

eng = JaxApproximateEngine(scene, JaxTrackingConfig(max_steps=200000), dtype=jnp.float64)
rng = np.random.default_rng(int.from_bytes(gid.encode(), "little") % 2**32)

res = dict(geometry=gid, cell_cm=[X, Y, Z], width_cm=float(spec.width_cm), radius_cm=float(spec.radius_cm),
           n_volumes=len(scene.volumes), batch=BATCH, regions={})
arrays = {}
t_all = time.perf_counter()
for ri, (name, (lo, hi, n)) in enumerate(REGIONS.items()):
    lo_cm, hi_cm = np.array(lo) * sp, np.array(hi) * sp
    w0, c0b, c1b, t0s, t1s, trunc = time.perf_counter(), [], [], [], [], 0
    cpu0 = time.process_time()
    for k, a in enumerate(range(0, n, BATCH)):
        m = min(BATCH, n - a)
        pts = lo_cm + rng.random((m, 3)) * (hi_cm - lo_cm)
        v = rng.normal(size=(m, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        o = track_photon_batch_compacted(eng, PhotonBatch(pts, v, np.zeros(m), np.full(m, -1, np.int32)),
                                         key=100000 * (ri + 1) + k)
        s = np.asarray(o.statuses); zf = np.asarray(o.interaction_points)[:, 2]; tt = np.asarray(o.times)
        c0 = (s == 2) & (zf > Z - 1e-6); c1 = (s == 2) & (zf < 1e-6)
        c0b.append([m, int(c0.sum())]); c1b.append([m, int(c1.sum())])
        t0s.append(tt[c0]); t1s.append(tt[c1])
        trunc += int(np.asarray(getattr(o, "truncated", np.zeros(m, bool))).sum())
    wall, cpu = time.perf_counter() - w0, time.process_time() - cpu0
    t0s, t1s = np.concatenate(t0s), np.concatenate(t1s)
    arrays[f"{name}_t0"], arrays[f"{name}_t1"] = t0s.astype(np.float64), t1s.astype(np.float64)
    arrays[f"{name}_b0"], arrays[f"{name}_b1"] = np.array(c0b), np.array(c1b)
    C0, C1 = int(t0s.size), int(t1s.size)
    res["regions"][name] = dict(lo=list(lo), hi=list(hi), photons=n, caught0=C0, caught1=C1, truncated=trunc,
                                wall_s=round(wall, 2), cpu_s=round(cpu, 2), photons_per_s=n / wall)
    print(f"{gid} {name:14} n={n} {wall/60:.1f} min (cpu/wall {cpu/wall:.1f}) p0 {C0/n:.5f} p1 {C1/n:.5f} "
          f"t0 {t0s.mean() if C0 else float('nan'):.2f} ns trunc={trunc}", flush=True)
res["wall_total_s"] = round(time.perf_counter() - t_all, 1)
json.dump(res, open(OUT / f"{gid}.json", "w"), indent=1)
np.savez_compressed(OUT / f"{gid}.npz", **arrays)
print("wrote", OUT / f"{gid}.json", flush=True)
