"""Speed of the numba forward engine (extras/numba_engine) on this 14-core CPU, for Fig. 11 and Fig. 13.

Scene: B_train_g0180, built exactly as for the adjoint atlas (same optical model; photons entering air are terminated).
(1) Fig. 11: M photons emitted uniformly over the cell (isotropic, t = 0), all threads, in chunks of 1e6. Detected
    photons (status CAUGHT) are binned by emission voxel; channel from the final position (port 0 above the top face,
    port 1 below the bottom face). Errors of the channel-0 maps of p and of the mean arrival time t-bar with the paper's
    metric against the 2e5-root reference (as g4_fast_bench.py / g4_fast_time.py). Timing: the transport call only
    (photon generation and binning reported separately).
(2) Fig. 13: N = 1e3 ... 1e6 photons from one voxel at the cell centre (voxel 10,10,4), all threads and one thread,
    after compilation and warm-up; best of 3 for N <= 1e4.
Writes study/numba_fwd.json.
usage: python numba_fwd_bench.py [M ...]
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
sys.path[:0] = [str(ROOT / "phinder_bridge/scripts"), str(ROOT / "Scintillator-Simulator-main/src"),
                str(ROOT / "Scintillator-Simulator-main/extras/numba_engine")]
import numba  # noqa: E402
import phinder_stage6_adjoint_geometry_atlas as atlasmod  # noqa: E402
import phinder_stage6_adjoint_prototype as proto  # noqa: E402
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch  # noqa: E402
from numba_engine import NumbaEngine  # noqa: E402

import common as C  # noqa: E402

GID = "B_train_g0180"
CAUGHT = 2
CHUNK = 1_000_000
raw = next(g for g in json.load(open(C.T / "banks/bank_test40.json"))["geometries"] if g["id"] == GID)
ta = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005,
                           max_live_branches=64, max_branch_nodes=1024, max_steps=750, max_flight_time_ns=80.0,
                           start_epsilon_cm=1e-7)
_, _, _, cfg = atlasmod._geometry_cfg(raw, ta, 606271, 808291)
scene, _, _, _ = proto._build_from_cfg(cfg)
X, Y, Z = map(float, cfg["cell_cm"])
SP = np.array([X / 20, Y / 20, Z / 8])
eng = NumbaEngine(scene)
NT = numba.config.NUMBA_NUM_THREADS
rng = np.random.default_rng(11)


def batch(n, lo=(0, 0, 0), hi=(20, 20, 8)):
    lo = np.array(lo) * SP; hi = np.array(hi) * SP
    p = lo + rng.random((n, 3)) * (hi - lo)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return PhotonBatch(p, v, np.zeros(n), np.full(n, -1, np.int32))


t0 = time.perf_counter(); eng.track_photon_batch(batch(2000), seed=0); compile_s = time.perf_counter() - t0
out = json.load(open(C.ST / "numba_fwd.json")) if (C.ST / "numba_fwd.json").exists() else {}
out.update(gid=GID, threads=NT, compile_and_first_call_s=compile_s)

# ---------------------------------------------------------------- (1) uniform emission, maps
tr = C.truth(); gi = tr["ids"].index(GID)
P, T = tr["p"][gi], tr["mt"][gi]
PSE = tr["p_se"][gi] * C.ref_noise()["k_p"]; w = P / P.sum(); tnoise = C.truth_time_noise()["mt"]
maps = out.get("maps", {})
for M in [int(float(x)) for x in (sys.argv[1:] or ["1e6", "4e6", "1.6e7", "6.4e7"])]:
    cnt = np.zeros((2, 3200)); tsum = np.zeros(3200); t_tr = 0.0; t_all0 = time.perf_counter(); n = 0; steps = 0.0
    for k in range(int(np.ceil(M / CHUNK))):
        b = batch(min(CHUNK, M - n))
        t0 = time.perf_counter(); r = eng.track_photon_batch(b, seed=1000 + M % 997, index_offset=n); t_tr += time.perf_counter() - t0
        ijk = np.minimum((b.interaction_points / SP).astype(int), [19, 19, 7])
        vox = (ijk[:, 0] * 20 + ijk[:, 1]) * 8 + ijk[:, 2]
        sel = r.batch.statuses == CAUGHT
        ch = (r.batch.interaction_points[sel, 2] < Z / 2).astype(int)        # final z: port 0 above, port 1 below
        np.add.at(cnt, (ch, vox[sel]), 1)
        s0 = sel.copy(); s0[sel] = ch == 0
        np.add.at(tsum, vox[s0], r.batch.times[s0])
        n += b.size; steps += float(r.steps.sum())
    q = cnt[0] / (n / 3200.0)
    perr = float(np.sqrt(max(np.sum(w * ((q - P) / P) ** 2) - np.sum(w * (PSE / P) ** 2), 0)))
    tb = np.where(cnt[0] > 0, tsum / np.maximum(cnt[0], 1), np.nan)
    rr = np.where(cnt[0] > 0, (tb - T) / T, 1.0)
    terr = float(np.sqrt(max(np.sum(w * rr ** 2) - tnoise ** 2, 0)))
    maps[str(M)] = dict(n=n, transport_s=t_tr, total_s=time.perf_counter() - t_all0, photons_per_s=n / t_tr,
                        steps_per_photon=steps / n, p_err=perr, t_err=terr, empty_w=float(np.sum(w[cnt[0] == 0])),
                        p_whole_ch0=float(cnt[0].sum() / n), p_whole_ch1=float(cnt[1].sum() / n), p_whole_ref=float(P.mean()))
    m = maps[str(M)]
    print(f"M={n:.2e}: transport {t_tr:.1f} s ({n/t_tr:.3g} ph/s, {NT} threads), total {m['total_s']:.1f} s; "
          f"p err {100*perr:.1f}%, t-bar err {100*terr:.1f}%; p ch0 {m['p_whole_ch0']:.5f} ch1 {m['p_whole_ch1']:.5f} "
          f"ref {P.mean():.5f}", flush=True)
    out["maps"] = maps
    json.dump(out, open(C.ST / "numba_fwd.json", "w"), indent=1)

# ---------------------------------------------------------------- (2) point-like source, N scan
pt = {}
for nthr in (NT, 1):
    numba.set_num_threads(nthr)
    for N in (1000, 10000, 100000, 1000000):
        best = np.inf
        for rep in range(3 if N <= 10000 else 1):
            b = batch(N, (10, 10, 4), (11, 11, 5))
            t0 = time.perf_counter(); r = eng.track_photon_batch(b, seed=7 + rep); best = min(best, time.perf_counter() - t0)
        pt[f"{nthr}_{N}"] = dict(threads=nthr, N=N, transport_s=best, caught=float(np.mean(r.batch.statuses == CAUGHT)))
        print(f"point source, {nthr:2d} threads, N={N:.0e}: {best:.4f} s ({N/best:.3g} ph/s)", flush=True)
numba.set_num_threads(NT)
out["point_scan"] = pt
json.dump(out, open(C.ST / "numba_fwd.json", "w"), indent=1)
