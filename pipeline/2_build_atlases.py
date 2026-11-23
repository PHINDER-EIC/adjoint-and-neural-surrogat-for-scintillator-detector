"""Step 2: adjoint maps of the three training banks, 2e4 roots per geometry, max_steps 5000 (numba reverse tracer).

Writes work/atlas_<bank>/stage6B_adjoint_geometry_atlas_numba.h5.  Resumable: geometries already in a file are skipped.
The launch and physics seeds depend only on a geometry's split, so every map is reproduced bit for bit
(checked on single geometries against the released atlas).  About 6 s per geometry on 14 threads: 684 geometries, ~70 min.
usage: python 2_build_atlases.py [--roots 20000]
"""
import argparse
import sys

from _common import ATLAS_PARTS, BANKS, REGENERATE, WORK, run

ap = argparse.ArgumentParser()
ap.add_argument("--roots", type=int, default=20000)
a = ap.parse_args()
for bank, _ in ATLAS_PARTS:
    run([sys.executable, REGENERATE, WORK / f"atlas_{bank}", "--bank", BANKS / f"{bank}.json", "--roots", a.roots,
         "--steps", 5000, "--baseline-steps", 0, "--pause-total-min", 1e9, "--pause-geometry-s", 1e9])
