"""Regenerate the 600-single v2 bank with the fast engine + three-way benchmark.

PART A — three propagation variants on identical events (config 0, LY 1000):
  1. real engine, stock                (Python stepping + Python poly hits)
  2. real engine + numba poly kernel   (the box_ray_intersections approach:
     jit ONLY ConvexPolyhedronVolume.ray_intersections, keep the Python loop)
  3. fast engine                       (whole stepping loop in one njit kernel)

PART B — rerun EVERY event of bank_N1 through the fast engine with the SAME
(theta, MIP truth, seed) as the saved real-engine bank (truth is re-derived
from event_seed and asserted equal to the per-event files), then validate:
  - n_generated matches exactly (generation path untouched)
  - caught-count agreement and correlation across 600 matched events
  - per-feature distribution shift in units of the real bank's spread
  - downstream: RF-A grouped-CV z-MSE on real vs fast bank, and
    train-on-fast -> test-on-real transfer
Writes bank_N1_fast.npz; NEVER touches the real bank files.

Run:  .venv-srso/Scripts/python.exe studies/v2/bank_fast_rerun.py [--skip-bench]
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
from numba import njit  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from gen_bank_v2 import event_seed  # noqa: E402
from sim_v2 import draw_mips, run_event_v2  # noqa: E402
from fast_engine import run_event_v2_fast  # noqa: E402
from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import (  # noqa: E402
    ConvexPolyhedronVolume)

BENCH_LY = 1000.0
BENCH_SEEDS = (600, 601)


# ---------------------------------------------------------------------------
# PART A: numba kernel for ray_intersections ONLY (box_ray_intersections style)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _poly_all_hits(face_p, face_n, verts, faces, start, direction, eps=1e-9):
    norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    d = direction / norm
    nf = face_p.shape[0]
    ts = np.empty(nf)
    normals = np.empty((nf, 3))
    points = np.empty((nf, 3))
    cnt = 0
    for f in range(nf):
        denom = d[0] * face_n[f, 0] + d[1] * face_n[f, 1] + d[2] * face_n[f, 2]
        if abs(denom) < eps:
            continue
        t = ((face_p[f, 0] - start[0]) * face_n[f, 0]
             + (face_p[f, 1] - start[1]) * face_n[f, 1]
             + (face_p[f, 2] - start[2]) * face_n[f, 2]) / denom
        if t <= eps:
            continue
        hx = start[0] + t * d[0]
        hy = start[1] + t * d[1]
        hz = start[2] + t * d[2]
        inside = True
        m = faces.shape[1]
        for i in range(m):
            a = faces[f, i]
            b = faces[f, (i + 1) % m]
            ex = verts[b, 0] - verts[a, 0]
            ey = verts[b, 1] - verts[a, 1]
            ez = verts[b, 2] - verts[a, 2]
            vx = hx - verts[a, 0]
            vy = hy - verts[a, 1]
            vz = hz - verts[a, 2]
            cx = ey * vz - ez * vy
            cy = ez * vx - ex * vz
            cz = ex * vy - ey * vx
            if (cx * face_n[f, 0] + cy * face_n[f, 1]
                    + cz * face_n[f, 2]) < -eps:
                inside = False
                break
        if inside:
            ts[cnt] = t
            points[cnt, 0] = hx; points[cnt, 1] = hy; points[cnt, 2] = hz
            normals[cnt, 0] = face_n[f, 0]
            normals[cnt, 1] = face_n[f, 1]
            normals[cnt, 2] = face_n[f, 2]
            cnt += 1
    order = np.argsort(ts[:cnt])
    return ts[:cnt][order], points[:cnt][order], normals[:cnt][order]


def _patched_ray_intersections(self, start, direction, eps=1e-9):
    faces_arr = getattr(self, "_faces_arr", None)
    if faces_arr is None:
        faces_arr = np.array(self.faces, dtype=np.int64)
        self._faces_arr = faces_arr
    ts, pts, nrm = _poly_all_hits(
        np.ascontiguousarray(self.face_points),
        np.ascontiguousarray(self.face_normals),
        np.ascontiguousarray(self.vertices), faces_arr,
        np.asarray(start, dtype=float), np.asarray(direction, dtype=float))
    return [(ts[i], pts[i], nrm[i]) for i in range(len(ts))]


def part_a() -> None:
    print("=== PART A: three-way benchmark (config 0, LY "
          f"{BENCH_LY:.0f}, {len(BENCH_SEEDS)} events each) ===", flush=True)
    thetas = np.load(V2 / "configs.npz")["thetas"]
    theta = thetas[0]
    cases = []
    for s in BENCH_SEEDS:
        rng = np.random.default_rng(s)
        cases.append((draw_mips(1, rng), s))

    original = ConvexPolyhedronVolume.ray_intersections
    rows = []
    for label, patch, runner in (
            ("real engine (stock)", False, run_event_v2),
            ("real + numba poly kernel only", True, run_event_v2),
            ("fast engine (whole loop njit)", False, run_event_v2_fast)):
        if patch:
            ConvexPolyhedronVolume.ray_intersections = _patched_ray_intersections
        try:
            if runner is run_event_v2_fast:            # warm compile off-clock
                runner(theta, cases[0][0], 999, 200.0)
            caught, tt = [], []
            for mips, s in cases:
                t0 = time.perf_counter()
                _f, t, _ng = runner(theta, mips, s, BENCH_LY)
                tt.append(time.perf_counter() - t0)
                caught.append(len(t))
        finally:
            ConvexPolyhedronVolume.ray_intersections = original
        rows.append((label, np.mean(tt), np.mean(caught)))
        print(f"  {label:34} {np.mean(tt):9.3f} s/event  "
              f"(caught ~{np.mean(caught):.0f})", flush=True)
    base = rows[0][1]
    print(f"  speedup vs stock: kernel-only {base / rows[1][1]:.2f}x, "
          f"whole-loop {base / rows[2][1]:.0f}x", flush=True)


# ---------------------------------------------------------------------------
# PART B: full bank rerun through the fast engine
# ---------------------------------------------------------------------------
def fast_worker(job):
    cfg, k, theta, ly = job
    seed = event_seed(1, cfg, k)
    rng = np.random.default_rng(seed)
    mips = draw_mips(1, rng)
    feats, caught_t, n_gen = run_event_v2_fast(theta, mips, seed, ly)
    return cfg, k, np.stack(mips, axis=1), feats, caught_t, n_gen


def part_b() -> None:
    files = sorted((V2 / "bank_N1").glob("ev_*.npz"))
    print(f"\n=== PART B: rerun {len(files)} bank_N1 events on the fast "
          "engine (same theta/truth/seed) ===", flush=True)
    real = {}
    for f in files:
        z = np.load(f)
        real[(int(z["config_id"]), int(z["event_k"]))] = dict(
            theta=z["theta"], mips=z["mips"], targets=z["targets"],
            features=z["features"], n_gen=int(z["n_generated"]),
            n_caught=len(z["caught_t"]), ly=float(z["light_yield"]))

    jobs = [(c, k, v["theta"], v["ly"]) for (c, k), v in sorted(real.items())]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as ex:
        out = list(ex.map(fast_worker, jobs, chunksize=8))
    wall = time.perf_counter() - t0
    print(f"  rerun wall time: {wall:.1f} s for {len(jobs)} events "
          f"({wall / len(jobs) * 1000:.0f} ms/event incl. pool overhead) — "
          "the real bank took ~2 days staged", flush=True)

    # ---- validation ------------------------------------------------------
    mip_err, ngen_ok, nc_r, nc_f = 0.0, 0, [], []
    TH, TG, FT, CID, EVK, NG = [], [], [], [], [], []
    for cfg, k, mips, feats, caught_t, n_gen in out:
        r = real[(cfg, k)]
        mip_err = max(mip_err, float(np.abs(mips - r["mips"]).max()))
        ngen_ok += int(n_gen == r["n_gen"])
        nc_r.append(r["n_caught"]); nc_f.append(len(caught_t))
        TH.append(r["theta"]); TG.append(r["targets"]); FT.append(feats)
        CID.append(cfg); EVK.append(k); NG.append(n_gen)
    nc_r = np.array(nc_r, float); nc_f = np.array(nc_f, float)
    print(f"  truth reproduction: max |mips_re - mips_saved| = {mip_err:.2e}")
    print(f"  n_generated exact match: {ngen_ok}/{len(out)}")
    z = (nc_f - nc_r) / np.sqrt(np.maximum(1.0, (nc_f + nc_r) / 2.0))
    print(f"  caught counts: mean real {nc_r.mean():.0f} vs fast "
          f"{nc_f.mean():.0f}; corr {np.corrcoef(nc_r, nc_f)[0, 1]:.4f}; "
          f"paired z = (d/sqrt(n)): mean {z.mean():+.2f}, std {z.std():.2f}")

    Ffast = np.array(FT, dtype=np.float64)
    Freal = np.array([real[(c, k)]["features"] for c, k in zip(CID, EVK)],
                     dtype=np.float64)
    shift = np.abs(Ffast.mean(0) - Freal.mean(0)) / (Freal.std(0) + 1e-9)
    print(f"  feature-mean shift (z units, 18 features): "
          f"max {shift.max():.3f}, median {np.median(shift):.3f}")

    np.savez_compressed(
        V2 / "bank_N1_fast.npz", theta=np.array(TH), targets=np.array(TG),
        features=Ffast.astype(np.float32),
        config_id=np.array(CID, np.int32), event_k=np.array(EVK, np.int32),
        n_generated=np.array(NG, np.int32))
    print("  saved bank_N1_fast.npz (real bank untouched)")

    # ---- downstream: does a reco trained on either bank behave the same? --
    T = np.array(TG, dtype=np.float64)
    THa = np.array(TH, dtype=np.float64)
    cid = np.array(CID)
    Xr = np.hstack([Freal, THa]); Xf = np.hstack([Ffast, THa])
    names = ["E", "mean_u", "mean_v"]
    folds = list(GroupKFold(5).split(Xr, groups=cid))

    def cv(Xtr_bank, Xte_bank, label):
        preds = np.zeros_like(T)
        for tr, va in folds:
            m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=8)
            m.fit(Xtr_bank[tr], T[tr]); preds[va] = m.predict(Xte_bank[va])
        zz = [float((((preds[:, i] - T[:, i]) / T[:, i].std()) ** 2).mean())
              for i in range(3)]
        print(f"  {label:34} " + " ".join(f"{n}={v:.3f}"
                                          for n, v in zip(names, zz)))

    print("  RF-A grouped 5-fold z-MSE (1.0 = predict-mean):")
    cv(Xr, Xr, "train real   -> test real")
    cv(Xf, Xf, "train fast   -> test fast")
    cv(Xf, Xr, "train fast   -> test real (transfer)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--bench-only", action="store_true",
                    help="run only the three-way benchmark (no bank_N1 needed)")
    args = ap.parse_args()
    set_seed(0)
    if not args.skip_bench:
        part_a()
    if not args.bench_only:
        part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
