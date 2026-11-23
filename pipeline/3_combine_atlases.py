"""Step 3: combine the three atlases into the training atlas and assign the splits used in the paper.

Ids are prefixed A_ (default bank), B_ (bank250), C_ (bank350).  The 40 test and 16 validation geometries are drawn
from A_/B_ with a fixed permutation (seed 20261028); the first 628 of the remaining pool are 'train'
(all 684 geometries are used).  Writes work/atlas_684_geometries_2e4roots.h5.
usage: python 3_combine_atlases.py            (or --check FILE: compare the split labels of FILE with this rule)
"""
import argparse

import h5py
import numpy as np

from _common import ATLAS_PARTS, LABEL_SEED, N_TEST, N_TRAIN, N_VAL, WORK

OUT = WORK / "atlas_684_geometries_2e4roots.h5"


def labels(all_ids):
    ids = [i for i in sorted(all_ids) if i[:2] in ("A_", "B_")]          # test/val drawn only from A/B
    extra = [i for i in sorted(all_ids) if i[:2] not in ("A_", "B_")]
    perm = np.random.default_rng(LABEL_SEED).permutation(len(ids))
    lab = {ids[i]: "test" for i in perm[:N_TEST]}
    lab.update({ids[i]: "val" for i in perm[N_TEST:N_TEST + N_VAL]})
    pool = [ids[i] for i in perm[N_TEST + N_VAL:]] + extra
    for k, gid in enumerate(pool):
        lab[gid] = "train" if k < N_TRAIN else "unused"
    return lab


ap = argparse.ArgumentParser()
ap.add_argument("--check", default="")
a = ap.parse_args()
if a.check:
    with h5py.File(a.check, "r") as f:
        stored = {g: f["geometries"][g].attrs["split"] for g in f["geometries"]}
    lab = labels(stored)
    bad = [g for g in stored if stored[g] != lab[g]]
    print(f"{len(stored)} geometries, {len(bad)} with a different split label")
    raise SystemExit(1 if bad else 0)

with h5py.File(OUT, "w") as f:
    f.create_dataset("u_edges", data=np.linspace(0.0, 1.0, 21))
    f.create_dataset("v_edges", data=np.linspace(0.0, 1.0, 21))
    f.create_dataset("s_edges", data=np.linspace(0.0, 1.0, 9))
    f.create_dataset("time_edges_ns", data=np.arange(241) * 0.25)
    g = f.create_group("geometries")
    for bank, pre in ATLAS_PARTS:
        with h5py.File(WORK / f"atlas_{bank}" / "stage6B_adjoint_geometry_atlas_numba.h5", "r") as s:
            for gid in s["geometries"]:
                s.copy(s["geometries"][gid], g, name=pre + gid)
    lab = labels(list(g.keys()))
    for gid, s_ in lab.items():
        g[gid].attrs["split"] = s_
    print({s_: sum(v == s_ for v in lab.values()) for s_ in ("train", "val", "test", "unused")}, "->", OUT)
