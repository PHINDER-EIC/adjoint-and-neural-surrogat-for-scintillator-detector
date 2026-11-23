"""Step 4: the bank of the 40 test geometries (the 'test' split of the training atlas), for the 2e5-root references.

Each entry is the geometry's bank entry with the atlas id (A_/B_ prefix) and split 'test', so the references use the
test seeds.  Writes work/banks/bank_test40.json (identical to data/banks/bank_test40.json).
usage: python 4_make_test_bank.py [atlas.h5]   (default: work/atlas_684_geometries_2e4roots.h5)
"""
import copy
import json
import sys

import h5py

from _common import ATLAS_PARTS, BANKS, WORK

atlas = sys.argv[1] if len(sys.argv) > 1 else WORK / "atlas_684_geometries_2e4roots.h5"
with h5py.File(atlas, "r") as f:
    test = sorted(g for g in f["geometries"] if f["geometries"][g].attrs["split"] == "test")
src = {pre: json.load(open(BANKS / f"{bank}.json")) for bank, pre in ATLAS_PARTS}
out = copy.deepcopy(src["A_"])
out["description"] = ("the 40 held-out test geometries of the surrogate learning curve "
                      "(ids prefixed A_ = default bank, B_ = bank250)")
out["geometries"] = []
for gid in test:
    pre, raw_id = gid[:2], gid[2:]
    e = copy.deepcopy(next(g for g in src[pre]["geometries"] if g["id"] == raw_id))
    e["id"], e["split"] = gid, "test"
    out["geometries"].append(e)
json.dump(out, open(BANKS / "bank_test40.json", "w"), indent=1)
print(len(out["geometries"]), "test geometries ->", BANKS / "bank_test40.json")
