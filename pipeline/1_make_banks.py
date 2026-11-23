"""Step 1: the four geometry banks (13-D Sobol designs, preflighted), written to work/banks/.

Each bank is identical to the one in data/banks/ (checked for all four).  Takes about 1 min in total.
usage: python 1_make_banks.py
"""
import sys

from _common import ATLAS_TOOL, BANK_SPECS, BANKS, run

BANKS.mkdir(parents=True, exist_ok=True)
for name, (tr, va, te, seed) in BANK_SPECS.items():
    run([sys.executable, ATLAS_TOOL, "make-bank", "--output", BANKS / f"{name}.json", "--train", tr, "--val", va,
         "--test", te, "--sobol-seed", seed, "--split-seed", seed + 1])
