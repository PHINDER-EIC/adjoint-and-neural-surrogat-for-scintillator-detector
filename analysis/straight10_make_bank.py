"""Ten straight-guide designs (no bends), as xgeom/make_bank.py for X_test_g0000 but spread over the design family:
cell D_x, D_y in [4, 6] cm, D_z in [0.7, 1.3] cm, width w in [0.4, 0.8] cm, R/w in [1.5, 2.5] (Sobol, scipy seed 20261111).
Start and end follow the family's fixed fractions (start at 0.10 D_x, 0.16 D_y; end at 0.78 D_x, D_y + 0.25 cm), so in
this family a straight guide always runs along the same diagonal; the four 'turning points' lie on that line at sorted
random fractions kept inside the internal-vertex ranges (x 12-88% of D_x, y 22-88% of D_y). Collinear points get no
fillet, so the guide has no bends. Writes bank_s10.json (split test -> 2e5-root reference seeds) and bank_s10_val.json
(split val -> independent 2e4-root seeds)."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import copy
import json

import numpy as np
from scipy.stats import qmc

base = json.load(open("../bank_x.json"))
u = qmc.Sobol(d=9, scramble=True, seed=20261111).random(16)[:10]
geoms = []
for i, v in enumerate(u):
    X, Y, Z = 4 + 2 * v[0], 4 + 2 * v[1], 0.7 + 0.6 * v[2]
    W = 0.4 + 0.4 * v[3]; RW = 1.5 + 1.0 * v[4]
    s = np.array([0.10 * X, 0.16 * Y]); e = np.array([0.78 * X, Y + 0.25])
    fy = lambda f: (s[1] + f * (e[1] - s[1])) / Y
    lo = max(0.06 / ((e[1] - s[1]) / Y), 0.03) + 0.01; hi = (0.88 - 0.16) / ((e[1] - s[1]) / Y) - 0.01
    fr = np.sort(lo + (hi - lo) * np.asarray(v[5:9]))
    fr = np.maximum.accumulate(fr + np.arange(4) * 1e-3)
    turns = [(s + f * (e - s)).tolist() for f in fr]
    assert all(0.12 <= t[0] / X <= 0.88 and 0.22 <= t[1] / Y <= 0.88 for t in turns), (i, turns)
    g = dict(id=f"S_test_g{i:04d}", cell_cm=[X, Y, Z], width_cm=W, radius_cm=RW * W, start_xy_cm=s.tolist(), turns_xy_cm=turns,
             end_xy_cm=e.tolist(), mirrored_guides=True, n_segments=5, r_over_w=RW, candidate_index=-1,
             design_vector=[X, Y, Z, W, RW * W] + [c for t in turns for c in t], spacefill_vector=[0.5] * 13, split="test")
    geoms.append(g)
    print(g["id"], f"cell {X:.2f} x {Y:.2f} x {Z:.2f}  w {W:.3f}  R/w {RW:.2f}  fractions {np.round(fr, 2)}")
b = copy.deepcopy(base); b["geometries"] = geoms; b["description"] = "ten straight-guide designs (no bends)"
json.dump(b, open("bank_s10.json", "w"), indent=1)
bv = copy.deepcopy(b)
for g in bv["geometries"]:
    g["split"] = "val"
json.dump(bv, open("bank_s10_val.json", "w"), indent=1)
