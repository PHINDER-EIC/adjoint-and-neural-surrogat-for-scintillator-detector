"""Geometry 00: two straight guides that cross as an X (no bends). 5 x 5 x 1 cm cell, width and radius of ref_5x5x1.
The four 'turn' points lie on the straight line, so the rounding code adds no fillet (collinear points are skipped)."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json, numpy as np
b = json.load(open("bank_test40.json"))
X, Y, Z = 5.0, 5.0, 1.0
W, R = 0.5329505205154419, 1.075294521457457
s = np.array([0.1 * X, 0.16 * Y]); e = np.array([0.78 * X, Y + 0.25])
turns = [(s + f * (e - s)).tolist() for f in (0.2, 0.4, 0.6, 0.8)]
g = dict(id="X_test_g0000", cell_cm=[X, Y, Z], width_cm=W, radius_cm=R, start_xy_cm=s.tolist(), turns_xy_cm=turns,
         end_xy_cm=e.tolist(), mirrored_guides=True, n_segments=5, r_over_w=R / W, candidate_index=-1,
         design_vector=[X, Y, Z, W, R] + [c for t in turns for c in t], spacefill_vector=[0.5] * 13, split="test")
b["geometries"] = [g]; b["description"] = "geometry 00: straight guides crossing as an X"
json.dump(b, open("xgeom/bank_x.json", "w"), indent=1)
print(g)
