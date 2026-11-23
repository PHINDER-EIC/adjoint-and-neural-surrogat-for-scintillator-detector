"""ScintSim: the Simulate+Reconstruct data source for the SRSO loop.

Implements srso_kit's `Sim` contract (labels.py):

    sim(theta (B, theta_dim), conditions (B, 3), rng) -> (features (B, d_feat),
                                                          truth   (B, 3))

Each row of theta is one detector design; each row of conditions is one
particle (u, v, E): fractional entry position on the z=0 face and deposited
energy in MeV. For every row we build the Scene, generate scintillation
photons with ParticleInteraction, transport them with ApproximateEngine via
PhotonBatchSimulator (sequential, no progress bars), and summarize the
photons CAUGHT at the fiber-core +y readout face into a fixed feature vector.
The reconstruction net (srso_kit MLPReco) then learns features -> (u, v, E).

The optimizer never differentiates through this code — labels are detached
by srso_kit — so plain numpy/Monte Carlo is fine here.

Seeding: the srso_kit engine passes its np.random.default_rng(seed); we draw
one child seed per row from it, so results are reproducible per run seed.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import common  # noqa: F401  (path bootstrap; also silences sim logging)
from common import (
    BETA,
    CLAD_T,
    D_FEAT,
    DEMO_LIGHT_YIELD,
    FIBER_Y_OVERHANG,
    ZERO_CAUGHT_SENTINEL,
)

from Photon_Simulator import AIR, CLAD, CORE, EJ200, EJ228, BoxVolume, Scene
from Photon_Simulator.config.defaults import DX
from Photon_Simulator.core.photonsEngines.approximateEngine import (
    ApproximateEngine,
    ParallelConfig,
    PhotonBatch,
    PhotonBatchSimulator,
    PhotonStatus,
)
from Photon_Simulator.core.physics.particles import MIP
from Photon_Simulator.core.physics.physics import ParticleInteraction

# Categorical option order — MUST match problem.make_space()'s declaration.
MATERIAL_OPTIONS = ("EJ200", "EJ228")

# Continuous theta layout (declaration order in problem.make_space()):
#   0: width (x, cm)   1: height (y, cm)   2: depth (z, cm)   3: fiber_t (cm)
N_CONT = 4


def scaled_scintillator(option: str, light_yield: float):
    """Preset material with the demo-scaled light yield (statistics knob)."""
    base = {"EJ200": EJ200, "EJ228": EJ228}[option]
    return replace(base, light_yield_photons_per_MeV=light_yield)


def build_scene(width: float, height: float, depth: float, fiber_t: float,
                material) -> Scene:
    """Mirror scripts/examples/example_1.py: box scintillator with a fiber
    (clad + core) glued on the back (+z) face; core's +y face is the
    photodetector (`_is_caught` requires the names and a BoxVolume core)."""
    fiber_y1 = height + FIBER_Y_OVERHANG
    volumes = [
        BoxVolume("scintillator", material,
                  [0.0, 0.0, 0.0], [width, height, depth], priority=0),
        BoxVolume("fiber_clad", CLAD,
                  [0.0, 0.0, depth], [width, fiber_y1, depth + fiber_t],
                  priority=10),
        BoxVolume("fiber_core", CORE,
                  [0.0, 0.0, depth + CLAD_T],
                  [width, fiber_y1, depth + fiber_t - CLAD_T], priority=20),
    ]
    return Scene(volumes=volumes, outside_material=AIR,
                 scintillator_material=material)


def run_one_event(scene: Scene, x: float, y: float, length: float,
                  rng: np.random.Generator, light_yield: float):
    """One particle -> transported photon batch. Returns (end_batch, n_gen)."""
    scint = scene.get_scintillator()
    interaction = ParticleInteraction(scene, MIP())
    dedx_cm = 2.0 * scint.density_g_cm3        # DE_DX * rho, MeV/cm (see event.py)
    source_points, times = interaction.generate_photon_source(
        np.array([x, y, 0.0]), np.array([0.0, 0.0, 1.0]), length, DX,
        BETA, dedx_cm, light_yield, rng)
    directions, photon_times = interaction.photon_generator(
        source_points, times, scint.rise_time_ns, scint.decay_time_ns, rng)
    statuses = np.full(photon_times.shape, PhotonStatus.RUNNING_INSIDE,
                       dtype=np.int8)
    batch = PhotonBatch(interaction_points=source_points,
                        directions=directions,
                        times=photon_times,
                        statuses=statuses)
    engine = ApproximateEngine(scene=scene)
    simulator = PhotonBatchSimulator(engine,
                                     parallel_config=ParallelConfig(n_workers=1))
    end_batch = simulator.run(batch, parallel=False, show_progress=False)
    return end_batch, batch.size


def extract_features(end_batch: PhotonBatch, n_generated: int) -> np.ndarray:
    """Fixed-width photon summary (see common.FEATURE_NAMES)."""
    statuses = np.asarray(end_batch.statuses)
    times = np.asarray(end_batch.times)
    caught = statuses == int(PhotonStatus.CAUGHT)
    n_caught = int(caught.sum())
    out = np.empty(D_FEAT, dtype=np.float32)
    out[0] = np.log1p(n_caught)
    out[1] = n_caught / max(1, n_generated)
    if n_caught > 0:
        t = times[caught]
        out[2] = t.mean()
        out[3] = t.std()
        out[4], out[5], out[6] = np.quantile(t, [0.1, 0.5, 0.9])
    else:
        out[2:7] = ZERO_CAUGHT_SENTINEL
    out[7] = np.log1p(n_generated)
    return out


def evaluate_design(theta_row: np.ndarray, cond_row: np.ndarray, seed: int,
                    light_yield: float) -> np.ndarray:
    """One design + one particle -> one feature row. Module-level so it can
    run in worker processes (Windows spawn pickles the function reference)."""
    width, height, depth, fiber_t = theta_row[:N_CONT]
    mat_idx = int(np.argmax(theta_row[N_CONT:N_CONT + len(MATERIAL_OPTIONS)]))
    material = scaled_scintillator(MATERIAL_OPTIONS[mat_idx], light_yield)
    u, v, energy = cond_row
    # physical entry point; track length from deposited energy
    x = float(u) * float(width)
    y = float(v) * float(height)
    dedx_cm = 2.0 * material.density_g_cm3
    length = min(float(energy) / dedx_cm, 0.95 * float(depth))

    scene = build_scene(float(width), float(height), float(depth),
                        float(fiber_t), material)
    # Photon generation uses the passed Generator, but ApproximateEngine's
    # interaction sampling uses the GLOBAL np.random (its own parallel worker
    # init seeds it the same way, approximateEngine.py:1169) — seed both from
    # the row seed for full determinism, sequential and pooled alike.
    np.random.seed(seed % 2**32)
    row_rng = np.random.default_rng(seed)
    end_batch, n_gen = run_one_event(scene, x, y, length, row_rng, light_yield)
    return extract_features(end_batch, n_gen)


class ScintSim:
    """srso_kit `Sim` callable over Scintillator-Simulator Monte Carlo.

    Parameters
    ----------
    light_yield : photons/MeV used for every design (demo statistics knob).
    save_dir    : if set, every call dumps theta/conditions/features/truth to
                  save_dir/data/call_NNN_iterMMM.npz (all generated data kept).
    n_workers   : >1 evaluates designs in a persistent ProcessPoolExecutor
                  (the MC tracker is pure Python and CPU-bound; per-design
                  parallelism beats the simulator's per-batch pool because a
                  single event is only a few hundred photons). Deterministic:
                  per-row seeds are drawn from `rng` before dispatch and
                  results come back in submission order.
    """

    def __init__(self, light_yield: float = DEMO_LIGHT_YIELD,
                 save_dir: Optional[Path] = None, n_workers: int = 1):
        self.light_yield = float(light_yield)
        self.save_dir = Path(save_dir) if save_dir else None
        self.n_workers = int(n_workers)
        self.iteration = 0            # updated via begin_iteration hook
        self._call_idx = 0
        self._pool = None             # lazy, persists across iterations
        self.last_timing_s: float = float("nan")
        if self.save_dir:
            (self.save_dir / "data").mkdir(parents=True, exist_ok=True)

    # srso_kit forwards the engine's IterationState here each outer iteration.
    def begin_iteration(self, state) -> None:
        self.iteration = int(getattr(state, "iteration", 0))

    def _get_pool(self):
        if self._pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __call__(self, theta: torch.Tensor, conditions: torch.Tensor,
                 rng: np.random.Generator):
        t0 = time.perf_counter()
        th = theta.detach().cpu().numpy().astype(np.float64)
        cond = conditions.detach().cpu().numpy().astype(np.float64)
        B = th.shape[0]
        # draw ALL row seeds up front -> deterministic under parallelism
        seeds = [int(s) for s in rng.integers(0, 2**63 - 1, size=B)]

        if self.n_workers > 1:
            pool = self._get_pool()
            rows = list(pool.map(evaluate_design, list(th), list(cond), seeds,
                                 [self.light_yield] * B))
            feats = np.stack(rows).astype(np.float32)
        else:
            feats = np.empty((B, D_FEAT), dtype=np.float32)
            for i in range(B):
                feats[i] = evaluate_design(th[i], cond[i], seeds[i],
                                           self.light_yield)

        truth = cond.astype(np.float32)          # reco target = (u, v, E)
        self.last_timing_s = time.perf_counter() - t0

        if self.save_dir:
            np.savez_compressed(
                self.save_dir / "data" /
                f"call_{self._call_idx:03d}_iter_{self.iteration:03d}.npz",
                theta=th, conditions=cond, features=feats, truth=truth,
                light_yield=self.light_yield, timing_s=self.last_timing_s)
        self._call_idx += 1
        return (torch.from_numpy(feats), torch.from_numpy(truth))
