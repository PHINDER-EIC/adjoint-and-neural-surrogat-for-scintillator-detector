"""Validate NumbaEngine against engines A (ApproximateEngine) and B (JAX), and time it.

Run from the repo root (Scintillator-Simulator-main):

    python extras/numba_engine/validate_numba_engine.py            # everything (~3 min, A is the slow part)
    python extras/numba_engine/validate_numba_engine.py --quick    # smaller A samples

Checks
  1. photon-by-photon equality with A on a scene without randomness
  2. status fractions + caught-time quantiles vs A and B on four scenes (z-scores)
  3. speed (photons per second), all threads and one thread
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numba  # noqa: E402
import numpy as np  # noqa: E402

from Photon_Simulator.core.photonsEngines.batchTransport import BatchTransport  # noqa: E402

import scenes as S  # noqa: E402
from numba_engine import NumbaEngine  # noqa: E402

GROUPS = {"caught": [2], "bulk_abs": [0, 7, 8], "refl_abs": [4, 5, 6], "lost": [1, 9, 10], "edge": [3]}


def fractions(statuses):
    s = np.asarray(statuses)
    return {k: float(np.isin(s, v).mean()) for k, v in GROUPS.items()}, len(s)


def zscore(a, na, b, nb):
    se = np.sqrt(a * (1 - a) / na + b * (1 - b) / nb)
    return 0.0 if se == 0 else (b - a) / se


def compare(title, ref_name, ref, others):
    """ref/others: (statuses, times) per engine."""
    print(f"\n=== {title}")
    fr, nr = fractions(ref[0])
    head = f"  {'':10s} {ref_name + f' (n={nr})':>18s}" + "".join(f" {k + f' (n={len(v[0])})':>24s}" for k, v in others.items())
    print(head)
    worst = 0.0
    for g in GROUPS:
        row = f"  {g:10s} {fr[g] * 100:17.2f}%"
        for name, (s, _t) in others.items():
            fo, no = fractions(s)
            z = zscore(fr[g], nr, fo[g], no)
            worst = max(worst, abs(z))
            row += f" {fo[g] * 100:13.2f}% (z {z:+5.2f})"
        print(row)
    def q(s, t):
        t = np.asarray(t)[np.asarray(s) == 2]
        return np.percentile(t, [10, 50, 90]).round(3).tolist() if len(t) else "-"
    print(f"  caught time q10/q50/q90 ns: {ref_name} {q(*ref)}  " + "  ".join(f"{k} {q(*v)}" for k, v in others.items()))
    print(f"  largest |z| = {worst:.2f}  ({'OK' if worst < 3.5 else 'CHECK'}: |z| < 3.5 expected for agreement)")
    return worst


def run_a(scene, batch):
    r = BatchTransport(scene, engine="approximate", seed=7, max_steps=200000).run(batch)
    return r.batch.statuses, r.batch.times


def run_b(scene, batch):
    r = BatchTransport(scene, engine="jax", seed=7, max_steps=200000, chunk_size=batch.size).run(batch)
    return r.batch.statuses, r.batch.times


def run_d(scene, batch, seed=7):
    r = NumbaEngine(scene).track_photon_batch(batch, seed=seed)
    return r.batch.statuses, r.batch.times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_a = 1500 if args.quick else 4000
    worst = []

    # 1. exact, photon by photon ------------------------------------------------
    sc = S.deterministic_scene()
    b = S.uniform_batch(2000, 1)
    ra = BatchTransport(sc, engine="approximate", seed=1, max_steps=200000).run(b)
    rd = NumbaEngine(sc).track_photon_batch(b, seed=99)
    same_status = np.array_equal(ra.batch.statuses, rd.batch.statuses)
    same_steps = np.array_equal(ra.steps, rd.steps)
    dp = np.abs(ra.batch.interaction_points - rd.batch.interaction_points).max()
    dt = np.abs(ra.batch.times - rd.batch.times).max()
    print("=== 1. no-randomness scene, 2000 photons, A vs Numba photon by photon")
    print(f"  identical statuses: {same_status}   identical step counts: {same_steps}   "
          f"max |position diff| {dp:.2e} cm   max |time diff| {dt:.2e} ns   "
          f"statuses: {dict(zip(*np.unique(rd.batch.statuses, return_counts=True)))}")

    # 2. statistics ---------------------------------------------------------------
    sc = S.build_reference_scene()
    worst.append(compare("2a. reference scene (specular reflector)", "A",
                         run_a(sc, S.make_source_batch(n_a, 3)),
                         {"B": run_b(sc, S.make_source_batch(40000, 4)),
                          "Numba": run_d(sc, S.make_source_batch(400000, 5))}))
    sc = S.v2_scene()
    worst.append(compare("2b. v2 scene (frustum + clad + core, no wrap)", "A",
                         run_a(sc, S.v2_photons(max(1000, n_a // 2), 3)),
                         {"B": run_b(sc, S.v2_photons(20000, 4)),
                          "Numba": run_d(sc, S.v2_photons(400000, 5))}))
    sc = S.lambertian_scene(0.3)
    worst.append(compare("2c. Lambertian wrap (70% diffuse) - A cannot run this", "B",
                         run_b(sc, S.make_source_batch(40000, 4)),
                         {"Numba": run_d(sc, S.make_source_batch(400000, 5))}))
    sc = S.air_gap_scene(with_air_volume=True)
    worst.append(compare("2d. 1 mm air gap filled by an explicit air volume", "A",
                         run_a(sc, S.uniform_batch(n_a, 3)),
                         {"B": run_b(sc, S.uniform_batch(40000, 4)),
                          "Numba": run_d(sc, S.uniform_batch(400000, 5))}))

    # 3. speed --------------------------------------------------------------------
    print("\n=== 3. speed (after compilation)")
    for name, sc, mk in (("reference", S.build_reference_scene(), lambda n: S.make_source_batch(n, 8)),
                         ("v2", S.v2_scene(), lambda n: S.v2_photons(n, 8))):
        eng = NumbaEngine(sc)
        b = mk(200000)
        eng.track_photon_batch(mk(1000), seed=0)
        for threads in (numba.config.NUMBA_NUM_THREADS, 1):
            numba.set_num_threads(threads)
            t0 = time.perf_counter(); r = eng.track_photon_batch(b, seed=1); dtm = time.perf_counter() - t0
            print(f"  {name:9s} 200000 photons, {threads:2d} thread(s): {dtm:7.3f} s = {dtm / b.size * 1e6:6.2f} us/photon"
                  f"   (mean {r.steps.mean():.1f} steps)")
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    print(f"\nlargest |z| over all statistical comparisons: {max(worst):.2f}")


if __name__ == "__main__":
    main()
