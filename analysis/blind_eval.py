"""One-shot blind evaluation on 40 fresh geometries (new Sobol seed 953400, never seen during any training,
model selection or analysis). Frozen models: 3-network ensemble and single networks (628 training geometries, 240
epochs, final weights) and the GP (PCA-48, 628 training geometries). Reference: 200,000-root adjoint maps.
Writes study/blind.json.
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

import common as C

REPO = Path(str(paths.REPO))
sys.path.insert(0, str(REPO / "phinder_bridge/scripts"))
import phinder_stage6B_surrogate as S  # noqa: E402

ATL = C.T / "reference_blind40_2e5roots.h5"
MODELS = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
TC = torch.as_tensor(C.TC, dtype=torch.float32)


def main():
    with h5py.File(ATL, "a") as f:
        for k, v in (("u_edges", np.linspace(0, 1, 21)), ("v_edges", np.linspace(0, 1, 21)), ("s_edges", np.linspace(0, 1, 9)),
                     ("time_edges_ns", np.arange(241) * 0.25)):
            if k not in f:
                f.create_dataset(k, data=v)
        ids = sorted(g for g in f["geometries"] if g.startswith("test_"))
        tr = dict(p=[], p_se=[], mt=[], tsd=[], ids=ids, pvol=[], tvol=[])
        X = []
        for g in ids:
            G = f["geometries"][g]; fl = G["field"]
            tr["p"].append(fl["p_per_emitted"][:].ravel()); tr["p_se"].append(fl["p_se"][:].ravel())
            tr["mt"].append(fl["mean_transport_time_ns"][:].ravel())
            sd, _ = C.tsd_from_h(fl["h_per_ns"][:]); tr["tsd"].append(sd.ravel())
            X.append(G["spacefill_vector"][:])
    tr = {k: (np.array(v) if k != "ids" else v) for k, v in tr.items()}
    TN = C.truth_time_noise()
    data = S.load_split_data(ATL, ids)
    preds = []
    for m in MODELS:
        mdl, _ = S.load_model(m, torch.device("cpu")); mdl.eval()
        out = {k: np.zeros((len(ids), 3200), np.float32) for k in ("p0", "mt", "tsd")}
        with torch.no_grad():
            for a in range(0, len(ids) * 3200, 8192):
                ii = np.arange(a, min(a + 8192, len(ids) * 3200))
                geom, nodes, uvs, *_ = S._batch_from_indices(data, ii, torch.device("cpu"))
                o = mdl(geom, nodes, uvs)
                mass = mdl.timing_mass(o["mix0_logits"]); mass = mass / mass.sum(-1, keepdim=True)
                mu = mass @ TC
                out["p0"].flat[ii] = o["p0"].numpy(); out["mt"].flat[ii] = mdl.mean_time(o["mix0_logits"]).numpy()
                out["tsd"].flat[ii] = torch.sqrt(torch.clamp(mass @ TC ** 2 - mu ** 2, min=0)).numpy()
        preds.append(out)
    ens = {k: np.mean([p[k] for p in preds], 0) for k in ("p0", "mt", "tsd")}
    # GP refit on the same 628 training geometries
    sys.argv = sys.argv[:1]
    import gp_study as G
    test, pool = G.pools()
    trn = pool[:628]
    P, M, SD = G.maps(G.R.ATLAS, trn)
    Xtr = np.array([G.R.spacefill[i] for i in trn])
    (lp, mt, sd), tfit, tpred = G.gp_predict(Xtr, [np.log(np.maximum(P, 1e-6)), M, SD], np.array(X))
    gp = dict(p0=np.exp(lp), mt=mt, tsd=sd)
    # distance of each blind design to its nearest training design (13-D unit space)
    Xall = np.array([G.R.spacefill[i] for i in trn])
    dmin = np.min(np.linalg.norm(np.array(X)[:, None] - Xall[None], axis=-1), axis=1)
    res = dict(n_geometries=len(ids),
               nearest_training_distance=dict(median=float(np.median(dmin)), min=float(dmin.min()), max=float(dmin.max())))
    for name, pr in [("network_ensemble", ens), ("gp", gp)] + [(f"network_{i}", p) for i, p in enumerate(preds)]:
        pm = C.pmetrics(pr["p0"], tr)
        res[name] = dict(p=pm, t=C.tmetrics(pr["mt"], "mt", tr, TN["mt"]), sd=C.tmetrics(pr["tsd"], "tsd", tr, TN["tsd"]),
                         geo_t_rms=float(np.sqrt(np.mean(C.geo_time(pr["p0"], pr["mt"], tr) ** 2))))
        print(f"{name:18} p {100*pm['rms']:.2f}% (offset {100*pm['offset_rms']:.2f}, shape {100*pm['shape_rms']:.2f}, whole-vol {100*pm['geo_rel_rms']:.2f}) "
              f"t {100*res[name]['t']['rms']:.2f}% sd {100*res[name]['sd']['rms']:.2f}%", flush=True)
    json.dump(res, open(C.ST / "blind.json", "w"), indent=1)


if __name__ == "__main__":
    main()
