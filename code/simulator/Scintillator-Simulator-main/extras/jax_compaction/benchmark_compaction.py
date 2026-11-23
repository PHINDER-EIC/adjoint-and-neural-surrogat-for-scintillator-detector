"""Compare engine B with and without active-photon compaction.

Run from the repo root (Scintillator-Simulator-main):

    python extras/jax_compaction/benchmark_compaction.py --scene v2 --photons 20000
    python extras/jax_compaction/benchmark_compaction.py --scene reference --photons 20000
    python extras/jax_compaction/benchmark_compaction.py --scene v2 --photons 100000 --skip-baseline

Prints, for each mode: wall time, ms per photon, bounce statistics, the share
of useful work, and grouped final statuses with a z-score between the modes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))   # repo_root/src, if not installed

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)            # before any JAX array exists
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from Photon_Simulator import AIR, CLAD, CORE, BoxVolume, Scene  # noqa: E402
from Photon_Simulator.config.defaults import EJ200, DX  # noqa: E402
from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import ConvexPolyhedronVolume  # noqa: E402
from Photon_Simulator.core.detectorSetup.referenceScene import build_reference_scene, make_source_batch  # noqa: E402
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch  # noqa: E402
from Photon_Simulator.core.photonsEngines.JaxEngine import JaxApproximateEngine, JaxTrackingConfig  # noqa: E402
from Photon_Simulator.core.physics.physics import ParticleInteraction  # noqa: E402

from compaction import track_photon_batch_compacted  # noqa: E402

# Design 0 of fast_engine_for_lorenzo/studies/v2/configs.npz:
# [bottom_w, bottom_h, depth, taper, x_shear, fiber_t]
V2_THETA = (3.5517564015462995, 7.656830024905503, 1.7254351805895567,
            0.6187300961464643, 0.0839672802016139, 0.19017563592642545)


def build_v2_scene(theta=V2_THETA):
    """Same geometry as fast_engine_for_lorenzo/studies/v2/sim_v2.py:build_scene_v2."""
    bw, bh, d, taper, shear, ft = theta
    tw = taper * bw
    tx0 = 0.5 * (1.0 - shear) * (bw - tw)
    tx1 = tx0 + tw
    v = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0],
                  [tx0, 0, d], [tx1, 0, d], [tx1, bh, d], [tx0, bh, d]], dtype=float)
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    clad_t, overhang = 0.001, 0.2
    fy1 = bh + overhang
    return Scene(volumes=[
        ConvexPolyhedronVolume("scintillator", EJ200, v, faces, priority=0),
        BoxVolume("fiber_clad", CLAD, [tx0, 0, d], [tx1, fy1, d + ft], priority=10),
        BoxVolume("fiber_core", CORE, [tx0, 0, d + clad_t], [tx1, fy1, d + ft - clad_t], priority=20),
    ], outside_material=AIR, scintillator_material=EJ200)


def v2_photons(scene, n, seed):
    """Photons from one MIP track through the middle of the v2 scintillator."""
    bw, bh, d = V2_THETA[0], V2_THETA[1], V2_THETA[2]
    inter = ParticleInteraction(scene)
    rng = np.random.default_rng(seed)
    dedx = 2.0 * EJ200.density_g_cm3
    length = min(1.2 / dedx, 0.95 * d)
    light_yield = 1.2 * n / (dedx * length)        # ~20% more photons than needed, then cut to n
    pts, t0 = [], []
    while sum(len(p) for p in pts) < n:
        p, t = inter.generate_photon_source(np.array([0.5 * bw, 0.5 * bh, 1e-4]), np.array([0, 0, 1.0]),
                                            length, DX, 0.998, dedx, light_yield, rng)
        pts.append(p); t0.append(t)
    pts = np.concatenate(pts)[:n]; t0 = np.concatenate(t0)[:n]
    dirs, times = inter.photon_generator(pts, t0, EJ200.rise_time_ns, EJ200.decay_time_ns, rng)
    return PhotonBatch(pts, dirs, times, np.full(n, -1, dtype=np.int32))


def summarize(name, out, seconds, n):
    s = np.asarray(out.statuses)
    st = np.asarray(out.steps)
    groups = {
        "caught": int((s == 2).sum()),
        "bulk_absorbed": int(np.isin(s, [0, 7, 8]).sum()),
        "lost": int(np.isin(s, [1, 9, 10]).sum()),
        "reflector_absorbed": int(np.isin(s, [4, 5, 6]).sum()),
        "edge": int((s == 3).sum()),
        "truncated": int(np.asarray(out.truncated).sum()),
    }
    print(f"\n{name}: {seconds:8.2f} s   {seconds / n * 1e3:7.3f} ms/photon")
    print(f"  bounces: mean {st.mean():.1f}  median {np.median(st):.0f}  p99 {np.percentile(st, 99):.0f}  "
          f"max {st.max()}  (lock-step useful work {st.mean() / max(1, st.max()) * 100:.1f}%)")
    print("  " + "  ".join(f"{k} {v}" for k, v in groups.items()))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=["v2", "reference"], default="v2")
    ap.add_argument("--photons", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--segment-steps", type=int, default=16)
    ap.add_argument("--min-bucket", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--skip-baseline", action="store_true", help="only run the compacted engine")
    args = ap.parse_args()

    if args.scene == "v2":
        scene = build_v2_scene()
        batch = v2_photons(scene, args.photons, args.seed)
    else:
        scene = build_reference_scene()
        batch = make_source_batch(args.photons, args.seed)
    n = batch.size

    engine = JaxApproximateEngine(
        scene, JaxTrackingConfig(max_steps=args.max_steps, mixed_precision=True), dtype=jnp.float64)
    print(f"scene={args.scene}  photons={n}  backend={jax.default_backend()}  "
          f"segment_steps={args.segment_steps}  min_bucket={args.min_bucket}")

    def compacted(key):
        out = track_photon_batch_compacted(engine, batch, key=key, segment_steps=args.segment_steps,
                                           min_bucket=args.min_bucket)
        jax.block_until_ready(out)
        return out

    t0 = time.perf_counter(); compacted(0); warm = time.perf_counter() - t0   # compiles every bucket size
    t0 = time.perf_counter(); out_c = compacted(1); dt_c = time.perf_counter() - t0
    print(f"(compacted first call incl. compilation: {warm:.2f} s)")
    g_c = summarize("COMPACTED", out_c, dt_c, n)

    if args.skip_baseline:
        return
    t0 = time.perf_counter()
    out_b = engine.track_photon_batch(batch, key=1)
    jax.block_until_ready(out_b)
    dt_b = time.perf_counter() - t0
    g_b = summarize("BASELINE (lock-step, incl. one compilation)", out_b, dt_b, n)

    print(f"\nspeed-up: {dt_b / dt_c:.1f}x")
    print("agreement (z = difference / combined binomial sigma; |z| < 3 is expected):")
    for k in g_c:
        p1, p2 = g_b[k] / n, g_c[k] / n
        sigma = np.sqrt(max(p1 * (1 - p1) / n + p2 * (1 - p2) / n, 1e-12))
        print(f"  {k:18s} baseline {g_b[k]:7d}  compacted {g_c[k]:7d}  z = {(p2 - p1) / sigma:+.2f}")


if __name__ == "__main__":
    main()
