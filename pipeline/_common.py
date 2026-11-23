"""Shared settings of the pipeline scripts: where the simulator code lives and where results are written.

Results go to work/ (override with PHADJ_WORK), never to data/, so a rebuild can be compared with the released files."""
import os
import subprocess
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE / "code"))
import paths  # noqa: E402

SIM = paths.REPO
WORK = Path(os.environ.get("PHADJ_WORK", RELEASE / "work"))
BANKS = WORK / "banks"
ATLAS_TOOL = SIM / "phinder_bridge/scripts/phinder_stage6_adjoint_geometry_atlas.py"
REGENERATE = SIM / "Scintillator-Simulator-main/extras/numba_adjoint/regenerate_atlas.py"
SURROGATE = SIM / "phinder_bridge/scripts/phinder_stage6B_surrogate.py"
PYTHONPATH = os.pathsep.join([str(SIM / "Scintillator-Simulator-main/src"), str(SIM / "phinder_bridge/scripts"),
                              str(SIM / "Scintillator-Simulator-main/extras/numba_adjoint")])

# geometry banks: name -> (train, val, test, Sobol seed); split seed = Sobol seed + 1
BANK_SPECS = {"bank_default84": (64, 12, 8, 629100),     # = phinder_bridge/config/stage6B_geometry_bank_default.json
              "bank250": (200, 30, 20, 731200),
              "bank350": (310, 25, 15, 842300),
              "bank_blind40": (1, 0, 40, 953400)}        # 40 additional test designs (+ train_g0000)
# training atlas: bank -> id prefix in the combined file
ATLAS_PARTS = [("bank_default84", "A_"), ("bank250", "B_"), ("bank350", "C_")]
N_TRAIN, N_VAL, N_TEST, LABEL_SEED = 628, 16, 40, 20261028
SEEDS = [20261003, 1, 2]                                  # the three networks
EPOCHS, SAMPLES_PER_EPOCH = 240, 148032                   # = round(65536 * 628 / 278)


def run(cmd, **kw):
    env = dict(os.environ, PYTHONPATH=PYTHONPATH + os.pathsep + os.environ.get("PYTHONPATH", ""))
    print(">", " ".join(str(c) for c in cmd), flush=True)
    kw.setdefault("cwd", SIM)                 # the tools resolve phinder_bridge/config/... from the repository root
    r = subprocess.run([str(c) for c in cmd], env=env, **kw)
    if r.returncode:
        raise SystemExit(f"command failed with exit code {r.returncode}")
    return r
