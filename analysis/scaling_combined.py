"""Scaling with training size on the test set, the 40 new designs, and both pooled (80 geometries).

Every 240-epoch network of the scaling study (final weights) is evaluated on the new designs; its test-set
predictions are taken from the stored .pred.npz (and checked against a fresh evaluation of the same weights).
Metrics as in the paper: p-weighted rms of ln(p_net/p_ref), reference noise subtracted in quadrature (C.pmetrics),
and the p-weighted relative error of the mean time (C.tmetrics). Writes study/scaling_combined.json.
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

LC = C.T / "surrogate_lc"
SUR = C.ST / "sur"
BLIND = C.T / "reference_blind40_2e5roots.h5"
EXIST = {"n278_e240_s20261003": LC / "sur_278_e240.latest.pt", "n278_e240_s777": LC / "sur2_278_seed777.latest.pt",
         "n450_e240_s20261003": LC / "sur2_450.latest.pt", "n628_e240_s20261003": LC / "sur2_628.latest.pt"}
TC = torch.as_tensor(C.TC, dtype=torch.float32)


def model_path(tag):
    return EXIST.get(tag, SUR / f"{tag}.latest.pt")


def predict(path, data):
    mdl, _ = S.load_model(path, torch.device("cpu")); mdl.eval()
    G, V = data.n_geom, data.n_vox
    out = {k: np.zeros((G, V), np.float32) for k in ("p0", "mt")}
    with torch.no_grad():
        for a in range(0, G * V, 8192):
            ii = np.arange(a, min(a + 8192, G * V))
            geom, nodes, uvs, *_ = S._batch_from_indices(data, ii, torch.device("cpu"))
            o = mdl(geom, nodes, uvs)
            out["p0"].flat[ii] = o["p0"].numpy(); out["mt"].flat[ii] = mdl.mean_time(o["mix0_logits"]).numpy()
    return out


def blind_truth():
    with h5py.File(BLIND) as f:
        ids = sorted(g for g in f["geometries"] if g.startswith("test_"))
        tr = dict(p=[], p_se=[], mt=[])
        for g in ids:
            fl = f["geometries"][g]["field"]
            tr["p"].append(fl["p_per_emitted"][:].ravel()); tr["p_se"].append(fl["p_se"][:].ravel())
            tr["mt"].append(fl["mean_transport_time_ns"][:].ravel())
    return ids, {k: np.array(v) for k, v in tr.items()}


def main():
    TN = C.truth_time_noise()
    tt = C.truth()
    test_tr = {k: tt[k] for k in ("p", "p_se", "mt")}
    bids, btr = blind_truth()
    bdata = S.load_split_data(BLIND, bids)
    tdata = S.load_split_data(C.TRUTH_H5, tt["ids"])
    pool_tr = {k: np.concatenate([test_tr[k], btr[k]]) for k in test_tr}
    tags = sorted(t.name[:-9] for t in SUR.glob("n*_e240_*.pred.npz"))
    res = {}
    for tag in tags:
        d = np.load(SUR / f"{tag}.pred.npz")
        n = int(tag.split("_")[0][1:])
        path = model_path(tag)
        tpred = {"p0": d["final_p0"], "mt": d["final_mt"]}
        chk = predict(path, tdata)
        dev = float(np.max(np.abs(np.log(chk["p0"] / tpred["p0"]))))
        bpred = predict(path, bdata)
        row = dict(n=n, model=str(path), test_check_max_abs_ln=dev)
        for name, pr, tr in (("test", tpred, test_tr), ("new", bpred, btr),
                             ("pooled", {k: np.concatenate([tpred[k], bpred[k]]) for k in tpred}, pool_tr)):
            pm = C.pmetrics(pr["p0"], tr)
            row[name] = dict(p=pm["rms"], p_raw=pm["rms_raw"], t=C.tmetrics(pr["mt"], "mt", tr, TN["mt"])["rms"],
                             geo_max=pm["geo_rel_max"])
        res[tag] = row
        print(f"{tag:22s} check {dev:.1e}  p: test {100*row['test']['p']:.2f} new {100*row['new']['p']:.2f} "
              f"pooled {100*row['pooled']['p']:.2f}   t: test {100*row['test']['t']:.2f} new {100*row['new']['t']:.2f} "
              f"pooled {100*row['pooled']['t']:.2f}", flush=True)
        json.dump(res, open(C.out("scaling_combined.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
