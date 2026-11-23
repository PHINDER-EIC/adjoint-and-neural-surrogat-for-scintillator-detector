"""v2 simulation: x-asymmetric frustum scintillator, multi-MIP events.

Design theta (6 continuous, EJ-200 fixed):
  [bottom_w, bottom_h, depth, taper, x_shear, fiber_t]
  taper   = top_w / bottom_w                        in [0.4, 1.0]
  x_shear = 0 -> top base centered (x-symmetric);   in [0, 0.6]
            s -> top-base left edge at (1-s)/2 * slack (pushed toward x=0)

Geometry: ConvexPolyhedronVolume with the validated face winding (probe_wedge);
flat bottom entry face at z=0 (full rectangle), flat top at z=depth carrying
the v1 fiber layout (clad+core boxes sized to the TOP base; core +y face =
readout — all `_is_caught` constraints preserved).

Events: N MIPs; per MIP (u, v) fractional on the bottom face, E ~ U[E_LO,E_HI]
MeV, arrival t0 (first = 0 = trigger, others U[0, T0_MAX] ns). Track along +z,
length min(E/(dE/dx*rho), 0.95 * local_depth(x)) with the analytic local depth
from the two slanted x-wall planes.

Targets by N (homogeneous per batch, no masking needed):
  N=1: [E, u, v]                                                   (3)
  N=2: [E_tot, mean_u, mean_v, var_u, var_v, cov_uv, t_range]      (7)
  N=3: same + t_gap_avg (= t_range / 2 — collinearity documented)  (8)
Spreads are population (ddof=0) statistics over the event's MIPs, in
fractional coordinates (design-independent).

Features (winner set, ABSOLUTE times, t=0 at first-MIP arrival):
  f_abs(8) + f_early(7) + [log1p(n_caught), caught_fraction,
  log1p(n_generated)]  -> d_feat = 18
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "studies"))
import common  # noqa: E402
from common import CLAD_T, E_HI, E_LO, FIBER_Y_OVERHANG, U_HI, U_LO  # noqa: E402

import numpy as np  # noqa: E402
from feature_lab import f_abs, f_early  # noqa: E402
from sim import scaled_scintillator  # noqa: E402
from Photon_Simulator import AIR, CLAD, CORE, BoxVolume, Scene  # noqa: E402
from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import (  # noqa: E402
    ConvexPolyhedronVolume)
from Photon_Simulator.core.photonsEngines.approximateEngine import (  # noqa: E402
    ApproximateEngine, ParallelConfig, PhotonBatch, PhotonBatchSimulator,
    PhotonStatus)
from Photon_Simulator.core.physics.particles import MIP  # noqa: E402
from Photon_Simulator.core.physics.physics import ParticleInteraction  # noqa: E402
from Photon_Simulator.config.defaults import DX  # noqa: E402

LY = 10000.0                 # true EJ-200 light yield (position needs it)
T0_MAX = 10.0                # ns, arrival window for MIPs after the first
BETA = 0.998
BOUNDS = np.array([[1.0, 4.0],     # bottom_w
                   [3.0, 8.0],     # bottom_h
                   [1.0, 3.0],     # depth
                   [0.4, 1.0],     # taper
                   [0.0, 0.6],     # x_shear
                   [0.05, 0.30]])  # fiber_t
PARAM_NAMES = ["bottom_w", "bottom_h", "depth", "taper", "x_shear", "fiber_t"]
D_FEAT = 18
# validated winding: identical-shape poly box reproduced BoxVolume physics
FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
         [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]


def top_base(theta):
    bw, _bh, _d, taper, shear, _ft = theta
    tw = taper * bw
    slack = bw - tw
    x0 = 0.5 * (1.0 - shear) * slack     # shear=0 -> centered (symmetric)
    return float(x0), float(x0 + tw)


def local_depth(theta, x):
    """Available z-extent above entry x before a slanted x-wall is crossed."""
    bw, _bh, d, _taper, _shear, _ft = theta
    tx0, tx1 = top_base(theta)
    z_left = d * x / tx0 if tx0 > 1e-12 and x < tx0 else d
    z_right = d * (bw - x) / (bw - tx1) if (bw - tx1) > 1e-12 and x > tx1 else d
    return float(min(d, z_left, z_right))


def build_scene_v2(theta, material) -> Scene:
    bw, bh, d, _taper, _shear, ft = theta
    tx0, tx1 = top_base(theta)
    v = np.array([
        [0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0],       # bottom
        [tx0, 0, d], [tx1, 0, d], [tx1, bh, d], [tx0, bh, d],  # top
    ], dtype=float)
    fy1 = bh + FIBER_Y_OVERHANG
    vols = [
        ConvexPolyhedronVolume("scintillator", material, v, FACES, priority=0),
        BoxVolume("fiber_clad", CLAD, [tx0, 0, d], [tx1, fy1, d + ft],
                  priority=10),
        BoxVolume("fiber_core", CORE, [tx0, 0, d + CLAD_T],
                  [tx1, fy1, d + ft - CLAD_T], priority=20),
    ]
    return Scene(volumes=vols, outside_material=AIR,
                 scintillator_material=material)


def draw_mips(n_mips: int, rng: np.random.Generator):
    """Per-MIP truth: (u, v, E, t0). First MIP defines the trigger (t0=0)."""
    u = rng.uniform(U_LO, U_HI, n_mips)
    v = rng.uniform(U_LO, U_HI, n_mips)
    E = rng.uniform(E_LO, E_HI, n_mips)
    t0 = np.concatenate([[0.0], rng.uniform(0.0, T0_MAX, n_mips - 1)]) \
        if n_mips > 1 else np.zeros(1)
    return u, v, E, t0


def targets_v2(u, v, E, t0):
    """Per-batch target vector (see module docstring)."""
    n = len(u)
    if n == 1:
        return np.array([E[0], u[0], v[0]], dtype=np.float64)
    base = [E.sum(), u.mean(), v.mean(),
            u.var(), v.var(), float(np.cov(u, v, ddof=0)[0, 1]),
            float(t0.max() - t0.min())]
    if n == 2:
        return np.array(base, dtype=np.float64)
    return np.array(base + [float((t0.max() - t0.min()) / (n - 1))],
                    dtype=np.float64)


def run_event_v2(theta, mips, seed: int, light_yield: float = LY):
    """Simulate one multi-MIP event. Returns (features(18), caught_times,
    n_generated)."""
    u, v, E, t0 = mips
    mat = scaled_scintillator("EJ200", light_yield)
    scene = build_scene_v2(theta, mat)
    bw, bh = theta[0], theta[1]
    dedx = 2.0 * mat.density_g_cm3
    np.random.seed(seed % 2**32)
    rng = np.random.default_rng(seed)
    inter = ParticleInteraction(scene, MIP())

    pts_all, dirs_all, times_all = [], [], []
    for i in range(len(u)):
        x, y = float(u[i] * bw), float(v[i] * bh)
        L = min(float(E[i]) / dedx, 0.95 * local_depth(theta, x))
        pts, ptimes = inter.generate_photon_source(
            np.array([x, y, 1e-4]), np.array([0.0, 0.0, 1.0]), L, DX,
            BETA, dedx, light_yield, rng)
        dirs, phot_t = inter.photon_generator(
            pts, ptimes, mat.rise_time_ns, mat.decay_time_ns, rng)
        pts_all.append(pts); dirs_all.append(dirs)
        times_all.append(phot_t + t0[i])          # arrival offset of MIP i
    pts = np.concatenate(pts_all); dirs = np.concatenate(dirs_all)
    ptimes = np.concatenate(times_all)
    st = np.full(ptimes.shape, PhotonStatus.RUNNING_INSIDE, dtype=np.int8)
    batch = PhotonBatch(interaction_points=pts, directions=dirs,
                        times=ptimes, statuses=st)
    eng = ApproximateEngine(scene=scene)
    simr = PhotonBatchSimulator(eng, parallel_config=ParallelConfig(n_workers=1))
    end = simr.run(batch, parallel=False, show_progress=False)
    stz = np.asarray(end.statuses)
    t = np.asarray(end.times)[stz == int(PhotonStatus.CAUGHT)]
    n_gen = len(ptimes)
    feats = np.array(
        f_abs(t) + f_early(t)
        + [np.log1p(len(t)), len(t) / max(1, n_gen), np.log1p(n_gen)],
        dtype=np.float32)
    assert feats.shape == (D_FEAT,)
    return feats, t.astype(np.float32), n_gen
