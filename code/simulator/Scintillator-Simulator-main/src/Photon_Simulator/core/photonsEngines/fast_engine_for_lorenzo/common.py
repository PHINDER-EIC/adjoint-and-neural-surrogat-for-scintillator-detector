"""Path bootstrap + shared constants for the scintillator-SRSO coupling.

Import this FIRST in every script here: it puts both vendored packages on
sys.path so `import srso_kit` and `import Photon_Simulator` work from the
`.venv-srso` interpreter without installing either package.

NOTE on dtype: srso_kit is float32 throughout (its validated AIDO recipe);
we deliberately stay float32 here instead of the workspace float64 default.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent            # experiments/scintillator_srso
WORKSPACE = PKG_DIR.parents[1]                       # Lorenzo simulation/
SCINT_SRC = WORKSPACE / "Scintillator-Simulator-main" / "src"

for p in (str(WORKSPACE), str(SCINT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Keep the Photon_Simulator logger quiet: at INFO it writes per-call records
# and would drown hundreds of MC evaluations. Must be set before/after import.
os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.getLogger().setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Physics/demo constants (single source of truth for sim.py and problem.py)
# ---------------------------------------------------------------------------
# Reduced light yield for the ~10-min demo budget: real plastics are ~10_000
# photons/MeV; we scale down so one event is a few hundred photons. This is a
# statistics/speed knob for demonstrating the optimization loop, NOT a
# physics claim (relative resolutions change with photon counts).
DEMO_LIGHT_YIELD = 400.0          # photons / MeV

BETA = 0.998                      # MIP speed (v/c), matches example_1.py
CLAD_T = 0.001                    # cm, fiber cladding wall thickness
FIBER_Y_OVERHANG = 0.2            # cm, fiber extends past scintillator in +y

# Condition ranges: (u, v) fractional entry position on the z=0 face and
# deposited energy E in MeV. E_HI is chosen so the track length
# L = E / (dE/dx * rho) never exceeds 0.95 * min(depth) for any design in
# bounds (depth lo = 1.0 cm, dE/dx ~ 2.05 MeV/cm -> L(1.8 MeV) ~ 0.88 cm),
# keeping truth == requested condition (no clipping ambiguity).
U_LO, U_HI = 0.05, 0.95
E_LO, E_HI = 0.4, 1.8             # MeV deposited

# Objective scales: position targets are O(1) fractions; E spans ~1.4 MeV.
E_SCALE = 1.4

# Feature vector layout produced by sim.extract_features (d_feat entries)
FEATURE_NAMES = [
    "log1p_n_caught",
    "caught_fraction",
    "t_mean_ns",
    "t_std_ns",
    "t_q10_ns",
    "t_q50_ns",
    "t_q90_ns",
    "log1p_n_generated",
]
D_FEAT = len(FEATURE_NAMES)
D_LABEL = 3                       # (u_hat, v_hat, E_hat)
D_CONDITION = 3                   # (u, v, E)
ZERO_CAUGHT_SENTINEL = -1.0       # fills time stats when no photon is caught


def set_seed(seed: int) -> dict:
    """Set AND log every seed source (ml.md rule, hook-enforced).

    Covers: PYTHONHASHSEED, random.seed, numpy.random.seed (legacy global),
    torch.manual_seed. numpy Generators used by srso_kit/sim code are created
    separately from the same seed (np.random.default_rng(seed)) by callers.
    Returns the seed record so run manifests can persist it.
    """
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    record = {
        "seed": seed,
        "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
        "random.seed": seed,
        "numpy.random.seed": seed,
        "torch.manual_seed": seed,
    }
    logging.getLogger("scintillator_srso").warning("seeds set: %s", record)
    return record
