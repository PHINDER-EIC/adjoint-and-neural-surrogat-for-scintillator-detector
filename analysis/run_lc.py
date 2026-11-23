"""Learning curve of the repo's Stage-6B neural surrogate vs the Gaussian-process stand-in.

For each n_train: relabel combined.h5 (fixed 40 test / 16 val), train the repo surrogate with its
default settings (80 epochs, CPU), then score the network and a GP (PCA-48 + ARD Matern) on the same
40 test geometries with the same metrics.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = paths.DATA / "surrogate_lc"
PY = sys.executable
REPO = Path(str(paths.REPO))
SUR = REPO / "phinder_bridge/scripts/phinder_stage6B_surrogate.py"
sys.path.insert(0, str(SUR.parent))
import phinder_stage6B_surrogate as S  # noqa: E402

ATLAS = paths.DATA / "atlas_684_geometries_2e4roots.h5"
BANKS = {"A_": REPO / "phinder_bridge/config/stage6B_geometry_bank_default.json",
         "B_": (paths.DATA / "banks/.json")}
spacefill = {}
for pre, p in BANKS.items():
    for g in json.load(open(p))["geometries"]:
        spacefill[pre + g["id"]] = g["spacefill_vector"]


def ids_by_split():
    with h5py.File(ATLAS, "r") as f:
        return S._geometry_ids_by_split(f)


def metrics(pred_p, pred_t, data):
    """pred_p, pred_t: [G, V] predicted p0 and mean time for data's geometries."""
    tp, tt = data.p0, data.mean_t0
    lp, lt = np.log(np.maximum(pred_p, 1e-7)), np.log(np.maximum(tp, 1e-7))
    w = tp
    ok = np.isfinite(tt) & (tt > 0)
    gp_rel = np.abs(pred_p.mean(1) - tp.mean(1)) / tp.mean(1)
    return dict(
        p_weighted_rms_log=float(np.sqrt(np.sum(w * (lp - lt) ** 2) / np.sum(w))),
        abs_log_p0_median=float(np.median(np.abs(lp - lt))), abs_log_p0_p90=float(np.quantile(np.abs(lp - lt), 0.9)),
        mean_t_rel_rms=float(np.sqrt(np.mean(((pred_t[ok] - tt[ok]) / tt[ok]) ** 2))),
        mean_t_abs_median_ns=float(np.median(np.abs(pred_t[ok] - tt[ok]))),
        geometry_p0_relerr_median=float(np.median(gp_rel)), geometry_p0_relerr_max=float(np.max(gp_rel)))


def predict_network(model_path, data):
    model, _ = S.load_model(model_path, torch.device("cpu"))
    model.eval()
    G, V = data.n_geom, data.n_vox
    pp, pt = np.zeros((G, V)), np.zeros((G, V))
    with torch.no_grad():
        for a in range(0, G * V, 4096):
            ii = np.arange(a, min(a + 4096, G * V))
            geom, nodes, uvs, *_ = S._batch_from_indices(data, ii, torch.device("cpu"))
            out = model(geom, nodes, uvs)
            pp.flat[ii] = out["p0"].numpy()
            pt.flat[ii] = model.mean_time(out["mix0_logits"]).numpy()
    return pp, pt


def predict_gp(train, test, tr_ids, te_ids, ncomp=48):
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
    Xtr = np.array([spacefill[i] for i in tr_ids]); Xte = np.array([spacefill[i] for i in te_ids])

    def fit(y):
        k = C(1.0) * Matern(length_scale=np.ones(13), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-3, (1e-8, 1))
        return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(Xtr, y).predict(Xte)

    out = []
    for Y in (np.log(np.maximum(train.p0, 1e-6)), np.nan_to_num(train.mean_t0, nan=0.0)):
        pca = PCA(min(ncomp, len(Xtr) - 1)).fit(Y)
        Z = pca.transform(Y)
        cols = Parallel(n_jobs=-1)(delayed(fit)(Z[:, j]) for j in range(Z.shape[1]))
        out.append(pca.inverse_transform(np.column_stack(cols)))
    return np.exp(out[0]), out[1]


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [100, 200, 278]
    results = json.load(open(HERE / "lc_results.json")) if (HERE / "lc_results.json").exists() else {}
    for n in sizes:
        subprocess.run([PY, str(HERE / "make_combined.py"), "label", str(n)], check=True)
        model = HERE / f"sur_{n}.pt"
        t0 = time.perf_counter()
        r = subprocess.run([PY, "-u", str(SUR), "train", "--atlas", str(ATLAS), "--output", str(model), "--device", "cpu"],
                           capture_output=True, text=True)
        (HERE / f"sur_{n}.train.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:], flush=True); raise SystemExit(f"training failed for n={n}")
        t_train = time.perf_counter() - t0
        sp = ids_by_split()
        test = S.load_split_data(ATLAS, sp["test"])
        pp, pt = predict_network(model, test)
        net = metrics(pp, pt, test)
        t1 = time.perf_counter()
        train = S.load_split_data(ATLAS, sp["train"])
        gp_p, gp_t = predict_gp(train, test, sp["train"], sp["test"])
        gpm = metrics(gp_p, gp_t, test)
        del train
        results[str(n)] = dict(n_train=n, train_minutes=round(t_train / 60, 2), gp_minutes=round((time.perf_counter() - t1) / 60, 2),
                               network=net, gp=gpm)
        json.dump(results, open(HERE / "lc_results.json", "w"), indent=1)
        print(f"n={n} train {t_train/60:.1f} min | network: pw-rms {net['p_weighted_rms_log']:.4f} |dlogp| med {net['abs_log_p0_median']:.4f} "
              f"t-rms {net['mean_t_rel_rms']:.4f} geo-p med {net['geometry_p0_relerr_median']:.4f} | GP: pw-rms {gpm['p_weighted_rms_log']:.4f} "
              f"|dlogp| med {gpm['abs_log_p0_median']:.4f} t-rms {gpm['mean_t_rel_rms']:.4f} geo-p med {gpm['geometry_p0_relerr_median']:.4f}", flush=True)


if __name__ == "__main__":
    main()
