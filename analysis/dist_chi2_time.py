"""Where in time is the network's chi2 excess against Geant4? Same bins and merging as dist_more.py; each merged
group is assigned to t < 2 ns or t >= 2 ns by its first bin. Prints the share of chi2 and of chi2 - ndf below 2 ns."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import numpy as np
import torch

sys.argv = sys.argv[:1]
import dist_more as D  # noqa: E402

C, J, RG = D.C, D.J, D.RG
R = json.load(open(C.ST / "regions.json"))["regions"]
H = dict(np.load(C.ST / "regions_hist_single.npz"))
models = [RG.S.load_model(p, torch.device("cpu"))[0].eval() for p in J.PATHS]
edges = J.IDX * D.DT
acc = {m: np.zeros(4) for m in ("adjoint", "network")}   # chi2 early, ndf early, chi2 late, ndf late
for gid, regs in R.items():
    defs = json.loads(str(np.load(C.ST / "g4/geom" / f"{gid}.npz")["regions_json"]))
    nets = [RG.net_voxels(gid, [m]) for m in models]
    for name, (lo, hi) in defs.items():
        sl = C.region_slices(lo, hi); row = regs[name]
        n = D.coarse_counts(np.round(H[f"{gid}|{name}|g4_unpol"] * D.DT * row["g4_unpol"]["n"]))[0]
        shapes = {"adjoint": [D.norm(H[f"{gid}|{name}|adjoint"])[0]],
                  "network": [D.norm((mn[sl] * p0[sl][..., None]).reshape(-1, 240).sum(0))[0] for p0, _, mn in nets]}
        for m, ss in shapes.items():
            for s in ss:
                q = D.norm(D.coarse_counts(s))[0]; gi = D.groups(q * n.sum())
                for j in range(gi.max() + 1):
                    sel = gi == j; e = q[sel].sum() * n.sum(); c = (n[sel].sum() - e) ** 2 / e
                    early = edges[np.argmax(sel)] < 2.0
                    acc[m] += np.array([c, 1, 0, 0]) if early else np.array([0, 0, c, 1])
for m, (ce, ne, cl, nl) in acc.items():
    k = 3 if m == "network" else 1   # network summed over 3 seeds
    print(f"{m:8s} t<2ns: chi2 {ce/k:.0f} for {ne/k:.0f} groups   t>=2ns: chi2 {cl/k:.0f} for {nl/k:.0f} groups   "
          f"share of excess below 2 ns {(ce-ne)/((ce-ne)+(cl-nl)):.2f}")
