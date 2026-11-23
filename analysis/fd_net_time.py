"""Network derivatives of whole-cell p, mean arrival time, arrival-time s.d. and t10 at the perturbed designs of fd_study.py.
Whole-cell distribution = voxel mean of p0 * timing mass (uniform emission), same definitions as fd_time.py.
Single network (the one in the paper) and the 3-network average. Writes fd/network_time.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import common as C
from fd_time import EDGES, TC

REPO = Path(str(paths.REPO))
sys.path.insert(0, str(REPO / "phinder_bridge/scripts"))
import phinder_stage6B_surrogate as S  # noqa: E402

FD = C.ST / "fd"
MODELS = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
torch.set_num_threads(4)


def stats(M, p):
    m = M.sum()
    mu = float((M * TC).sum() / m)
    sd = float(np.sqrt((M * TC ** 2).sum() / m - mu ** 2))
    c = np.cumsum(M) / m
    j = int(np.searchsorted(c, 0.1)); c0 = c[j - 1] if j > 0 else 0.0
    return dict(p=float(p), t=mu, sd=sd, t10=float(EDGES[j] + (0.1 - c0) / (c[j] - c0) * 0.25))


def main():
    rows = {k: v for k, v in json.load(open(FD / "adjoint.json")).items() if v.get("ok")}
    models = [S.load_model(m, torch.device("cpu"))[0].eval() for m in MODELS]
    out = {"single": {}, "ensemble": {}}
    for key, r in rows.items():
        st = {"single": {}, "ensemble": {}}
        for s in ("p", "m"):
            d = S.load_split_data(FD / "fd_atlas.h5", [f"{r['gid']}__{r['param']}__{s}"])
            Ms, ps = [], []
            with torch.no_grad():
                geom, nodes, uvs, *_ = S._batch_from_indices(d, np.arange(d.n_vox), torch.device("cpu"))
                for mdl in models:
                    o = mdl(geom, nodes, uvs)
                    mass = mdl.timing_mass(o["mix0_logits"]).numpy(); mass = mass / mass.sum(1, keepdims=True)
                    p0 = o["p0"].numpy()
                    Ms.append((p0[:, None] * mass).mean(0)); ps.append(p0.mean())
            st["single"][s] = stats(Ms[0], ps[0])
            st["ensemble"][s] = stats(np.mean(Ms, 0), np.mean(ps))
        for w in ("single", "ensemble"):
            a, b = st[w]["p"], st[w]["m"]
            out[w][key] = {q: float(((np.log(a[q]) - np.log(b[q])) if q == "p" else (a[q] - b[q])) / (2 * r["du"]))
                           for q in ("p", "t", "sd", "t10")}
            out[w][key]["base"] = {q: 0.5 * (a[q] + b[q]) for q in ("p", "t", "sd", "t10")}
    json.dump(out, open(FD / "network_time.json", "w"), indent=1)
    for key in rows:
        print(key, {q: round(out["single"][key][q], 3) for q in ("p", "t", "sd", "t10")})


if __name__ == "__main__":
    main()
