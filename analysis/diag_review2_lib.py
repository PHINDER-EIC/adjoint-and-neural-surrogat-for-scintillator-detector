"""(a) Test set vs new designs: per-geometry error contributions, bootstrap intervals, permutation test.
(b) Does the network capture how the mean arrival time depends on guide length (the main effect of the guide path on timing)?
Writes study/diag_review2.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import h5py
import numpy as np
import torch

import common as C

sys.path.insert(0, str(paths.REPO / "phinder_bridge/scripts"))
import phinder_stage6B_surrogate as S  # noqa: E402

BL = C.T / "reference_blind40_2e5roots.h5"
MODELS = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
rng = np.random.default_rng(11)


def load(atl, ids):
    d = dict(p=[], se=[], mt=[], L=[], cz=[], w=[])
    with h5py.File(atl) as f:
        for g in ids:
            G = f["geometries"][g]; fl = G["field"]
            d["p"].append(fl["p_per_emitted"][:].ravel()); d["se"].append(fl["p_se"][:].ravel())
            d["mt"].append(fl["mean_transport_time_ns"][:].ravel())
            pts = G["guide0_points_cm"][:]
            d["L"].append(float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
            dj = json.loads(G.attrs["design_json"]); d["cz"].append(dj["cell_cm"][2]); d["w"].append(dj["width_cm"])
    return {k: np.array(v) for k, v in d.items()}


def predict(atl, ids, model):
    data = S.load_split_data(atl, ids)
    out = dict(p0=np.zeros((len(ids), 3200)), mt=np.zeros((len(ids), 3200)))
    with torch.no_grad():
        for a in range(0, len(ids) * 3200, 8192):
            ii = np.arange(a, min(a + 8192, len(ids) * 3200))
            geom, nodes, uvs, *_ = S._batch_from_indices(data, ii, torch.device("cpu"))
            o = model(geom, nodes, uvs)
            out["p0"].flat[ii] = o["p0"].numpy(); out["mt"].flat[ii] = model.mean_time(o["mix0_logits"]).numpy()
    return out


with h5py.File(C.TRUTH_H5) as f:
    test_ids = sorted(f["geometries"].keys())
with h5py.File(BL) as f:
    blind_ids = sorted(g for g in f["geometries"] if g.startswith("test_"))

models = [S.load_model(m, torch.device("cpu"))[0].eval() for m in MODELS]
