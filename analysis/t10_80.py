"""Channel asymmetry dt10 = t10(ch1) - t10(ch0) on all 80 test geometries (original 40 + 40 new designs), network
(each of the three seeds) and independent 2e4-root adjoint maps, against the 2e5-root references; same definitions as
t10_study.py (rms over voxels, sign agreement where |dt10_ref| > 0.25 ns). Also the crossed straight-guide geometry
X_test_g0000 (xgeom/): dt10 and the voxel error of p. Writes study/t10_80.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import h5py
import numpy as np
import torch

import common as C
from t10_study import quantile, mirror, S

PATHS = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
BLIND = C.T / "reference_blind40_2e5roots.h5"
XG = C.T / "xgeom"
models = [S.load_model(p, torch.device("cpu"))[0].eval() for p in PATHS]


def net_maps(path, gid):
    d = S.load_split_data(path, [gid])
    out = []
    with torch.no_grad():
        geom, nodes, uvs, *_ = S._batch_from_indices(d, np.arange(d.n_vox), torch.device("cpu"))
        for m in models:
            o = m(geom, nodes, uvs)
            M0 = (m.timing_mass(o["mix0_logits"]).numpy() * o["p0"].numpy()[:, None]).reshape(20, 20, 8, 240)
            M1 = (m.timing_mass(o["mix1_logits"]).numpy() * o["p1"].numpy()[:, None]).reshape(20, 20, 8, 240)
            out.append((o["p0"].numpy().reshape(20, 20, 8), quantile(M1, 0.1) - quantile(M0, 0.1)))
    return out


def dtq(h):
    return quantile(mirror(h), 0.1) - quantile(h, 0.1)


def metrics(d, dr):
    big = np.abs(dr) > 0.25
    return dict(rms_ns=float(np.sqrt(np.mean((d - dr) ** 2))), sign=float(np.mean(np.sign(d[big]) == np.sign(dr[big]))),
                n_sign=int(big.sum()))


rows = []
sets = []
with h5py.File(C.TRUTH_H5) as fr, h5py.File(C.T / "atlas_684_geometries_2e4roots.h5") as fa:
    for gid in sorted(fr["geometries"]):
        indep = "_test_" not in gid    # 2e4 maps of originally-test geometries share roots with the reference
        sets.append(("original", C.TRUTH_H5, gid, fr["geometries"][gid]["field/h_per_ns"][:],
                     fa["geometries"][gid]["field/h_per_ns"][:] if indep else None))
with h5py.File(BLIND) as fb:
    for gid in sorted(g for g in fb["geometries"] if g.startswith("test_")):
        sets.append(("new", BLIND, gid, fb["geometries"][gid]["field/h_per_ns"][:], np.load(C.ST / "adj_new_h" / f"{gid}.npz")["h"]))
D = {"ref": [], "adj": [], "net": [[], [], []], "set": []}
for name, path, gid, h, ha in sets:
    dr = dtq(h)
    D["ref"].append(dr); D["set"].append(name)
    D["adj"].append(dtq(ha) if ha is not None else np.full_like(dr, np.nan))
    for i, (_, d) in enumerate(net_maps(path, gid)):
        D["net"][i].append(d)
    print(name, gid, flush=True)
ref = np.array(D["ref"]); adj = np.array(D["adj"]); net = [np.array(x) for x in D["net"]]; st = np.array(D["set"])
res = {}
for sel_name, sel in (("original40", st == "original"), ("new40", st == "new"), ("all80", np.ones(len(st), bool))):
    ok = sel & ~np.isnan(adj).any(axis=(1, 2, 3))
    per = [metrics(n[sel], ref[sel]) for n in net]
    res[sel_name] = dict(net_seeds=per, net_rms_ns=float(np.mean([p["rms_ns"] for p in per])),
                         net_sign=float(np.mean([p["sign"] for p in per])),
                         adj20k=dict(metrics(adj[ok], ref[ok]), n_geometries=int(ok.sum())),
                         ref_range_ns=[float(np.quantile(ref[sel], 0.01)), float(np.quantile(ref[sel], 0.99))])

# crossed straight guides
with h5py.File(XG / "ref/stage6B_adjoint_geometry_atlas_numba.h5") as f, h5py.File(XG / "adj20k/stage6B_adjoint_geometry_atlas_numba.h5") as g:
    gid = "X_test_g0000"
    h = f["geometries"][gid]["field/h_per_ns"][:]; p = f["geometries"][gid]["field/p_per_emitted"][:]
    ha = g["geometries"][gid]["field/h_per_ns"][:]; pa = g["geometries"][gid]["field/p_per_emitted"][:]
    seeds = list(f["geometries"][gid].attrs.get("launch_seed", [])) if False else None
dr = dtq(h); w = p / p.sum()
perr = lambda q: float(np.sqrt(np.sum(w * ((q - p) / p) ** 2)))
xn = net_maps(XG / "ref/stage6B_adjoint_geometry_atlas_numba.h5", "X_test_g0000")
res["xgeom"] = dict(net_dt10=[metrics(d, dr) for _, d in xn], net_p_err=[perr(q) for q, _ in xn],
                    net_p_err_mean3=perr(np.mean([q for q, _ in xn], 0)),
                    adj20k_dt10=metrics(dtq(ha), dr), adj20k_p_err=perr(pa),
                    ref_rel_se_p=float(np.sqrt(np.sum(w * (f_se := 0) ** 2))) if False else None)
json.dump(res, open(C.ST / "t10_80.json", "w"), indent=1)
print(json.dumps(res, indent=1))
# arrays for the scatter of Fig. 14 (all 80 test geometries): reference, independent 2e4 map (NaN where not independent),
# first network (sur2_628_final)
np.savez_compressed(C.ST / "t10_80_arrays.npz", ref=ref, adj=adj, net0=net[0], sets=st)
print("saved t10_80_arrays.npz")
