"""Recompute the network metrics of blind.json (40 new designs) with the current common.pmetrics / tmetrics.
Same frozen models and reference as blind_eval.py; the GP entry (not used in paper v1) is kept from the old file
because refitting it takes 40 min. Also stores the per-voxel predictions in study/blind_preds.npz."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np

import common as C
import blind_eval as B

old = json.load(open(C.ST / "blind.json"))
import h5py  # noqa: E402
with h5py.File(B.ATL) as f:
    ids = sorted(g for g in f["geometries"] if g.startswith("test_"))
    tr = dict(p=[], p_se=[], mt=[], tsd=[])
    for g in ids:
        fl = f["geometries"][g]["field"]
        tr["p"].append(fl["p_per_emitted"][:].ravel()); tr["p_se"].append(fl["p_se"][:].ravel())
        tr["mt"].append(fl["mean_transport_time_ns"][:].ravel())
        sd, _ = C.tsd_from_h(fl["h_per_ns"][:]); tr["tsd"].append(sd.ravel())
tr = {k: np.array(v) for k, v in tr.items()}
TN = C.truth_time_noise()
data = B.S.load_split_data(B.ATL, ids)
preds = []
for m in B.MODELS:
    mdl, _ = B.S.load_model(m, B.torch.device("cpu")); mdl.eval()
    out = {k: np.zeros((len(ids), 3200), np.float32) for k in ("p0", "mt", "tsd")}
    with B.torch.no_grad():
        for a in range(0, len(ids) * 3200, 8192):
            ii = np.arange(a, min(a + 8192, len(ids) * 3200))
            geom, nodes, uvs, *_ = B.S._batch_from_indices(data, ii, B.torch.device("cpu"))
            o = mdl(geom, nodes, uvs)
            mass = mdl.timing_mass(o["mix0_logits"]); mass = mass / mass.sum(-1, keepdim=True)
            mu = mass @ B.TC
            out["p0"].flat[ii] = o["p0"].numpy(); out["mt"].flat[ii] = mdl.mean_time(o["mix0_logits"]).numpy()
            out["tsd"].flat[ii] = B.torch.sqrt(B.torch.clamp(mass @ B.TC ** 2 - mu ** 2, min=0)).numpy()
    preds.append(out)
ens = {k: np.mean([p[k] for p in preds], 0) for k in ("p0", "mt", "tsd")}
np.savez_compressed(C.ST / "blind_preds.npz", **{f"net{i}_{k}": v for i, p in enumerate(preds) for k, v in p.items()},
                    **{f"truth_{k}": v for k, v in tr.items()}, ids=np.array(ids))
res = dict(old)
for name, pr in [("network_ensemble", ens)] + [(f"network_{i}", p) for i, p in enumerate(preds)]:
    pm = C.pmetrics(pr["p0"], tr)
    res[name] = dict(p=pm, t=C.tmetrics(pr["mt"], "mt", tr, TN["mt"]), sd=C.tmetrics(pr["tsd"], "tsd", tr, TN["tsd"]),
                     geo_t_rms=float(np.sqrt(np.mean(C.geo_time(pr["p0"], pr["mt"], tr) ** 2))))
    print(f"{name:18} p {100*pm['rms']:.2f}% (old {100*old[name]['p']['rms']:.2f})  t {100*res[name]['t']['rms']:.2f}% "
          f"(old {100*old[name]['t']['rms']:.2f})  sd {100*res[name]['sd']['rms']:.2f}% (old {100*old[name]['sd']['rms']:.2f})")
json.dump(res, open(C.ST / "blind.json", "w"), indent=1)
