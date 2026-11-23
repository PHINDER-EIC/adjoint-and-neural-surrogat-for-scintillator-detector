"""Voxel-level Jensen-Shannon divergence of the channel-0 arrival-time shape f_0(t), network and 2e4-root adjoint maps
vs the 2e5-root reference, on all 80 test geometries (Sec. VI C, Fig. 8).

Per voxel: shapes normalized over 0-60 ns on the Fig. 6 coarse bins (41 bins), JS in bits; summary = p-weighted mean
over voxels (as the other voxel errors), and the JS of the whole-cell distribution (uniform emission) per geometry.
The reference is noisy too. The 2e4-root map is unbiased, so its JS to the reference is noise from both maps, which
scales as 1/N: the reference's own share is about JS(2e4, ref) / 11. Network = mean over the three seeds of the
per-seed JS (each network evaluated on its own). Writes study/voxel_js.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.argv = sys.argv[:1]
import common as C  # noqa: E402
import js_regions as J  # noqa: E402

S = J.S
BLIND = C.T / "reference_blind40_2e5roots.h5"
ATL = {"A": C.T / "atlas_numba/stage6B_adjoint_geometry_atlas_numba.h5", "B": C.T / "atlas_numba250/stage6B_adjoint_geometry_atlas_numba.h5"}
IDX = J.IDX


def coarse_v(h):
    c = np.add.reduceat(h.reshape(-1, 240).astype(float), IDX[:-1], axis=1)
    s = c.sum(1, keepdims=True)
    return c / np.where(s > 0, s, 1), s[:, 0] > 0


def js_v(p, q):
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0); b = np.where(q > 0, q * np.log2(q / m), 0)
    return 0.5 * (a.sum(1) + b.sum(1))


def js1(p, q):
    return float(js_v(p[None], q[None])[0])


def network(path, gid, models):
    d = S.load_split_data(path, [gid])
    out = []
    with torch.no_grad():
        geom, nodes, uvs, *_ = S._batch_from_indices(d, np.arange(d.n_vox), torch.device("cpu"))
        for m in models:
            o = m(geom, nodes, uvs)
            p0 = o["p0"].numpy(); mass = m.timing_mass(o["mix0_logits"]).numpy()
            out.append((p0, mass * p0[:, None]))
    return out


def main():
    models = [S.load_model(p, torch.device("cpu"))[0].eval() for p in J.PATHS]
    sets = []
    with h5py.File(C.TRUTH_H5) as f:
        for gid in sorted(f["geometries"]):
            pre, g = gid.split("_", 1)
            with h5py.File(ATL[pre]) as a:
                h20 = a["geometries"][g]["field/h_per_ns"][:]
            sets.append(("original", C.TRUTH_H5, gid, f["geometries"][gid]["field/p_per_emitted"][:].ravel(),
                         f["geometries"][gid]["field/h_per_ns"][:], h20))
    with h5py.File(BLIND) as f:
        for gid in sorted(g for g in f["geometries"] if g.startswith("test_")):
            z = np.load(C.ST / "adj_new_h" / f"{gid}.npz")
            sets.append(("new", BLIND, gid, f["geometries"][gid]["field/p_per_emitted"][:].ravel(),
                         f["geometries"][gid]["field/h_per_ns"][:], z["h"]))
    rows = []
    for name, path, gid, p, href, h20 in sets:
        ref, ok = coarse_v(href); a20, ok2 = coarse_v(h20)
        w = np.where(ok & ok2, p, 0); w = w / w.sum()
        whole = lambda h: (lambda c: c / c.sum())(np.add.reduceat(h.reshape(-1, 240).sum(0).astype(float), IDX[:-1]))
        r = dict(set=name, gid=gid, adj20k=float(np.sum(w * js_v(a20, ref))), adj20k_whole=js1(whole(h20), whole(href)))
        nv, nw = [], []
        for p0, M in network(path, gid, models):
            nm, _ = coarse_v(M)
            nv.append(float(np.sum(w * js_v(nm, ref)))); nw.append(js1(whole(M), whole(href)))
        r.update(net=float(np.mean(nv)), net_seeds=nv, net_whole=float(np.mean(nw)))
        rows.append(r)
        print(f"{name:8} {gid:15} voxel JS: net {1e3*r['net']:.2f}  adj20k {1e3*r['adj20k']:.2f}   whole cell: net {1e3*r['net_whole']:.3f} adj20k {1e3*r['adj20k_whole']:.3f}", flush=True)
    summ = {}
    for sel in ("original", "new", "all80"):
        R = [r for r in rows if sel == "all80" or r["set"] == sel]
        m = lambda k: float(np.mean([r[k] for r in R]))
        summ[sel] = dict(n=len(R), net=m("net"), adj20k=m("adj20k"), ref_floor=m("adj20k") / 11,
                         net_whole=m("net_whole"), adj20k_whole=m("adj20k_whole"),
                         net_whole_median=float(np.median([r["net_whole"] for r in R])))
    json.dump(dict(units="bits", bins="Fig. 6 coarse bins (41) over 0-60 ns", summary=summ, rows=rows),
              open(C.ST / "voxel_js.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
