"""Step 5: reference maps, 2e5 roots per geometry: the 40 test geometries and the 40 additional test designs.

Writes work/reference_test40/ and work/reference_blind40/ (stage6B_adjoint_geometry_atlas_numba.h5 in each).
Bit-identical to the released references (checked on one geometry).  About 40-80 s per geometry: ~1 h per set.
usage: python 5_build_references.py
"""
import sys

from _common import BANKS, REGENERATE, WORK, run

for bank, out in (("bank_test40", "reference_test40"), ("bank_blind40", "reference_blind40")):
    run([sys.executable, REGENERATE, WORK / out, "--bank", BANKS / f"{bank}.json", "--roots", 200000, "--steps", 5000,
         "--baseline-steps", 0, "--pause-total-min", 1e9, "--pause-geometry-s", 1e9])
