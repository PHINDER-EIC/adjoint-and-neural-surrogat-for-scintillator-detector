"""GP stand-in scaling: PCA-48 + ARD Matern-5/2 GP per component on the 13-D design vector, for log p0,
mean time and time s.d. (0-60 ns) maps. Training sets = the same first-n pools as the network
(make_combined.py label logic, computed in memory so it can run alongside training). Predictions on the 40
test geometries saved to study/gp/gp_n<n>.npz with fit/predict times.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
T = paths.DATA
sys.path.insert(0, str(T / "surrogate_lc"))
sys.argv = sys.argv[:1] + sys.argv[1:]
import run_lc as R  # noqa: E402

R.spacefill.update({"C_" + g["id"]: g["spacefill_vector"] for g in json.load(open(T / "banks/bank.json"))["geometries"]})
OUT = T / "derived/gp"
OUT.mkdir(parents=True, exist_ok=True)
TRUTH = T / "reference_test40_2e5roots.h5"
TC = (np.arange(240) + 0.5) * 0.25


def pools():
    with h5py.File(R.ATLAS, "r") as f:
        allids = sorted(f["geometries"].keys())
    ids = [i for i in allids if i[:2] in ("A_", "B_")]
    extra = [i for i in allids if i[:2] not in ("A_", "B_")]
    perm = np.random.default_rng(20261028).permutation(len(ids))
    return sorted(ids[i] for i in perm[:40]), [ids[i] for i in perm[56:]] + extra


def maps(path, ids):
    P, M, SD = [], [], []
    with h5py.File(path, "r") as f:
        for g in ids:
            fl = f["geometries"][g]["field"]
            P.append(fl["p_per_emitted"][:].ravel()); M.append(fl["mean_transport_time_ns"][:].ravel())
            h = fl["h_per_ns"][:].reshape(-1, 240)
            m0 = h.sum(1); mu = (h * TC).sum(1) / np.maximum(m0, 1e-30)
            SD.append(np.sqrt(np.maximum((h * TC ** 2).sum(1) / np.maximum(m0, 1e-30) - mu ** 2, 0)))
    return np.array(P), np.nan_to_num(np.array(M)), np.array(SD)


def gp_predict(Xtr, Ys, Xte, ncomp=48):
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

    def fit(y):
        k = C(1.0) * Matern(length_scale=np.ones(13), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-3, (1e-8, 1))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(Xtr, y)
        return gp

    outs, tfit, tpred = [], 0.0, 0.0
    for Y in Ys:
        t0 = time.perf_counter()
        pca = PCA(min(ncomp, len(Xtr) - 1)).fit(Y)
        Z = pca.transform(Y)
        gps = Parallel(n_jobs=4)(delayed(fit)(Z[:, j]) for j in range(Z.shape[1]))
        tfit += time.perf_counter() - t0
        t1 = time.perf_counter()
        outs.append(pca.inverse_transform(np.column_stack([g.predict(Xte) for g in gps])))
        tpred += time.perf_counter() - t1
    return outs, tfit, tpred


if __name__ == "__main__":
    test, pool = pools()
    with h5py.File(TRUTH) as f:
        assert test == sorted(f["geometries"].keys())
    Xte = np.array([R.spacefill[i] for i in test])
    for n in [int(x) for x in sys.argv[1:]] or [50, 100, 200, 278, 450, 628]:
        if (OUT / f"gp_n{n}.npz").exists():
            continue
        tr = pool[:n]
        P, M, SD = maps(R.ATLAS, tr)
        Xtr = np.array([R.spacefill[i] for i in tr])
        (lp, mt, sd), tfit, tpred = gp_predict(Xtr, [np.log(np.maximum(P, 1e-6)), M, SD], Xte)
        np.savez_compressed(OUT / f"gp_n{n}.npz", p0=np.exp(lp).astype(np.float32), mt=mt.astype(np.float32),
                            tsd=sd.astype(np.float32), fit_s=tfit, predict_s=tpred, n=n)
        print(f"{time.strftime('%H:%M')} GP n={n} fit {tfit/60:.1f} min predict {tpred:.2f} s", flush=True)
    print("ALLDONE", flush=True)
