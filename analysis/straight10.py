"""Network on ten straight-guide designs (xgeom/straight10, no bends; absent from training) plus X_test_g0000:
voxel errors of p, t-bar, s_t (C.pmetrics / C.tmetrics: p-weighted, reference noise subtracted, as for the 80 test
geometries), per-geometry offset, and dt10 (rms, sign where |dt10_ref| > 0.25 ns), for each of the three networks and an
independent 2e4-root adjoint map. Writes study/straight10.json."""
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
SETS = {"straight10": C.T / "xgeom/straight10", "X": C.T / "xgeom"}
models = [S.load_model(p, torch.device("cpu"))[0].eval() for p in PATHS]
TN = C.truth_time_noise()


def load(path):
    out = {k: [] for k in ("p", "p_se", "mt", "tsd", "h", "ids")}
    with h5py.File(path) as f:
        for g in sorted(f["geometries"]):
            fl = f["geometries"][g]["field"]
            h = fl["h_per_ns"][:]
            out["ids"].append(g); out["h"].append(h)
            out["p"].append(fl["p_per_emitted"][:].ravel()); out["p_se"].append(fl["p_se"][:].ravel())
            out["mt"].append(fl["mean_transport_time_ns"][:].ravel()); out["tsd"].append(C.tsd_from_h(h)[0].ravel())
    return {k: (np.array(v) if k != "ids" else v) for k, v in out.items()}


def dt10(h):
    return quantile(mirror(h), 0.1) - quantile(h, 0.1)


def score(pp, mt, sd, d10, tr):
    m = C.pmetrics(pp, tr)
    dr = np.array([dt10(h) for h in tr["h"]]); big = np.abs(dr) > 0.25
    return dict(p=m["rms"], p_offset=m["offset_rms"], p_geo_max=m["geo_rel_max"], t=C.tmetrics(mt, "mt", tr, TN["mt"])["rms"],
                sd=C.tmetrics(sd, "tsd", tr, TN["tsd"])["rms"], dt10_ps=1e3 * float(np.sqrt(np.mean((d10 - dr) ** 2))),
                dt10_sign=float(np.mean(np.sign(d10[big]) == np.sign(dr[big]))))


res = {}
for name, base in SETS.items():
    ref_path = base / "ref/stage6B_adjoint_geometry_atlas_numba.h5"
    tr = load(ref_path); a = load(base / "adj20k/stage6B_adjoint_geometry_atlas_numba.h5")
    assert a["ids"] == tr["ids"]
    r = dict(n=len(tr["ids"]), adj20k=score(a["p"], a["mt"], a["tsd"], np.array([dt10(h) for h in a["h"]]), tr))
    seeds = []
    for mdl in models:
        P, MT, SD, D = [], [], [], []
        for gid in tr["ids"]:
            d = S.load_split_data(ref_path, [gid])
            with torch.no_grad():
                geom, nodes, uvs, *_ = S._batch_from_indices(d, np.arange(d.n_vox), torch.device("cpu"))
                o = mdl(geom, nodes, uvs)
                m0 = mdl.timing_mass(o["mix0_logits"]).numpy(); m1 = mdl.timing_mass(o["mix1_logits"]).numpy()
                p0 = o["p0"].numpy()
            m0n = m0 / m0.sum(1, keepdims=True)
            mu = (m0n * C.TC).sum(1); sdv = np.sqrt(np.maximum((m0n * C.TC ** 2).sum(1) - mu ** 2, 0))
            P.append(p0); MT.append(mu); SD.append(sdv)
            D.append(quantile(m1.reshape(20, 20, 8, 240), 0.1) - quantile(m0.reshape(20, 20, 8, 240), 0.1))
        seeds.append(score(np.array(P), np.array(MT), np.array(SD), np.array(D), tr))
    r["net_seeds"] = seeds
    r["net_mean"] = {k: float(np.mean([s[k] for s in seeds])) for k in seeds[0]}
    res[name] = r
json.dump(res, open(C.ST / "straight10.json", "w"), indent=1)
for name, r in res.items():
    n, a = r["net_mean"], r["adj20k"]
    print(f"{name:10} n={r['n']:2d}  network: p {100*n['p']:.1f}% (offset {100*n['p_offset']:.1f}%, worst geometry {100*n['p_geo_max']:.1f}%)  "
          f"t {100*n['t']:.1f}%  s_t {100*n['sd']:.1f}%  dt10 {n['dt10_ps']:.0f} ps sign {100*n['dt10_sign']:.1f}%")
    print(f"{'':10}       adj 2e4: p {100*a['p']:.1f}%  t {100*a['t']:.1f}%  s_t {100*a['sd']:.1f}%  dt10 {a['dt10_ps']:.0f} ps sign {100*a['dt10_sign']:.1f}%")
    print(f"{'':10}       per seed p: {[round(100*s['p'],2) for s in r['net_seeds']]}")
