"""Validate fast_engine (numba) against the real ApproximateEngine.

Protocol (same yardstick as matched_event_test): for CONFIG_IDS designs, same
single-MIP truth, R seeds per engine. The fast engine differs from the real
one only in the propagation RNG stream, so if the physics port is faithful its
features must sit within the real engine's own seed-to-seed noise:
    median |mean_fast - mean_real| / real_seed_std  ~ 1
Also compared: caught fraction, caught-time quantiles, status budget, and the
wall-time per event (the point of the exercise).

Run:  .venv-srso/Scripts/python.exe studies/v2/validate_fast_engine.py
      [--ly 2000] [--seeds 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
import sim_v2  # noqa: E402
import common  # noqa: E402
from common import set_seed  # noqa: E402

import numpy as np  # noqa: E402
from sim_v2 import run_event_v2  # noqa: E402
from fast_engine import ST_ABSORBED, ST_CAUGHT, ST_EDGE, ST_LOST, \
    run_event_v2_fast  # noqa: E402

CONFIG_IDS = [0, 7, 21]
WORKERS = 6


def real_job(args):
    theta, mips, seed, ly = args
    t0 = time.perf_counter()
    feats, t, ng = run_event_v2(theta, mips, seed, ly)
    return feats, t, ng, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ly", type=float, default=2000.0)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    set_seed(0)

    cfg = np.load(V2 / "configs.npz")
    thetas = cfg["thetas"]
    rng = np.random.default_rng(123)
    cases = []
    for c in CONFIG_IDS:
        u = rng.uniform(0.1, 0.9, 1); v = rng.uniform(0.1, 0.9, 1)
        E = rng.uniform(0.8, 1.6, 1); t0 = np.zeros(1)
        cases.append((c, thetas[c], (u, v, E, t0)))

    # ---- real engine (parallel, it is the slow side) ----------------------
    jobs = [(theta, mips, 500 + s, args.ly)
            for _c, theta, mips in cases for s in range(args.seeds)]
    tw = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        real = list(ex.map(real_job, jobs))
    real_wall = time.perf_counter() - tw
    real_percall = np.mean([r[3] for r in real])

    # ---- fast engine (serial; includes one-time numba compile) ------------
    t_c = time.perf_counter()
    run_event_v2_fast(cases[0][1], cases[0][2], 12345, 200.0)  # warm compile
    compile_s = time.perf_counter() - t_c
    fast, fast_times = [], []
    for _c, theta, mips in cases:
        for s in range(args.seeds):
            t1 = time.perf_counter()
            fast.append(run_event_v2_fast(theta, mips, 900 + s, args.ly))
            fast_times.append(time.perf_counter() - t1)

    # ---- comparison --------------------------------------------------------
    ns = args.seeds
    print(f"LY {args.ly:.0f}, {len(cases)} configs x {ns} seeds/engine")
    print(f"{'config':>6} {'caught real':>12} {'caught fast':>12} "
          f"{'|d_feat|/seed-noise':>20}")
    ratios, all_r_t, all_f_t = [], [], []
    for i, (c, _theta, _mips) in enumerate(cases):
        Freal = np.array([real[i * ns + s][0] for s in range(ns)])
        Ffast = np.array([fast[i * ns + s][0] for s in range(ns)])
        nr = np.mean([len(real[i * ns + s][1]) for s in range(ns)])
        nf = np.mean([len(fast[i * ns + s][1]) for s in range(ns)])
        for s in range(ns):
            all_r_t.append(real[i * ns + s][1]); all_f_t.append(fast[i * ns + s][1])
        noise = Freal.std(0) + 1e-9
        ratio = float(np.median(np.abs(Freal.mean(0) - Ffast.mean(0)) / noise))
        ratios.append(ratio)
        print(f"{c:6d} {nr:12.0f} {nf:12.0f} {ratio:20.2f}")
    print(f"\nmedian ratio across configs: {np.median(ratios):.2f}  "
          f"(~1 => fast engine statistically indistinguishable)")

    rt = np.concatenate(all_r_t); ft = np.concatenate(all_f_t)
    q = [10, 50, 90]
    print(f"caught-time quantiles ns  real {np.percentile(rt, q).round(3)}  "
          f"fast {np.percentile(ft, q).round(3)}")

    print(f"\nreal engine : {real_percall:8.2f} s/event "
          f"(pool wall {real_wall:.1f} s)")
    print(f"fast engine : {np.mean(fast_times):8.3f} s/event "
          f"(numba compile once: {compile_s:.1f} s)")
    print(f"SPEEDUP     : {real_percall / np.mean(fast_times):8.1f}x "
          f"per event, single-thread vs single-thread")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
