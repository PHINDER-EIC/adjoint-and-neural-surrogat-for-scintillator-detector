"""Surrogate training study: data scaling (with seeds) and epoch scaling, all scored on the 200k-root test maps.

Protocol for every run: repo surrogate defaults (hidden 40, trunk 128, 24 mixtures, batch 512, lr 2e-3 cosine),
samples per epoch = 65536 * n / 278, epochs as listed. Fixed 40 test / 16 val geometries (make_combined.py label).
For each run, predictions of the best-val checkpoint (.pt) and the final weights (.latest.pt) on the 40 test
geometries are saved: p0, p1, mean time, time s.d. (0-60 ns), plus the training time.

Existing models from surrogate_lc/ are scored first. Resumable (skips runs whose .pred.npz exists).
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

T = paths.DATA
LC = T / "surrogate_lc"
OUT = T / "derived/sur"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LC))
sys.argv = sys.argv[:1]
import run_lc as R  # noqa: E402

S = R.S
TRUTH = T / "reference_test40_2e5roots.h5"
TC = torch.as_tensor((np.arange(240) + 0.5) * 0.25, dtype=torch.float32)

import h5py  # noqa: E402
with h5py.File(TRUTH) as f:
    TEST_IDS = sorted(f["geometries"].keys())
truth = S.load_split_data(TRUTH, TEST_IDS)


def predict(model_path):
    model, _ = S.load_model(model_path, torch.device("cpu"))
    model.eval()
    G, V = truth.n_geom, truth.n_vox
    out = {k: np.zeros((G, V), np.float32) for k in ("p0", "p1", "mt", "tsd")}
    t0 = time.perf_counter()
    with torch.no_grad():
        for a in range(0, G * V, 8192):
            ii = np.arange(a, min(a + 8192, G * V))
            geom, nodes, uvs, *_ = S._batch_from_indices(truth, ii, torch.device("cpu"))
            o = model(geom, nodes, uvs)
            m = model.timing_mass(o["mix0_logits"])
            m = m / m.sum(-1, keepdim=True)
            mu = m @ TC
            sd = torch.sqrt(torch.clamp(m @ TC ** 2 - mu ** 2, min=0))
            out["p0"].flat[ii] = o["p0"].numpy(); out["p1"].flat[ii] = o["p1"].numpy()
            out["mt"].flat[ii] = model.mean_time(o["mix0_logits"]).numpy(); out["tsd"].flat[ii] = sd.numpy()
    out["predict_s"] = time.perf_counter() - t0
    return out


def save_preds(tag, best, final, meta):
    d = {f"best_{k}": v for k, v in predict(best).items()}
    if final is not None and Path(final).exists():
        d.update({f"final_{k}": v for k, v in predict(final).items()})
    np.savez_compressed(OUT / f"{tag}.pred.npz", meta=json.dumps(meta), **d)
    print(f"{time.strftime('%H:%M')} scored {tag}", flush=True)


def label(atlas, n_train):
    """make_combined.py label logic, on this worker's own copy of combined.h5."""
    with h5py.File(atlas, "a") as f:
        allids = sorted(f["geometries"].keys())
        ids = [i for i in allids if i[:2] in ("A_", "B_")]
        extra = [i for i in allids if i[:2] not in ("A_", "B_")]
        perm = np.random.default_rng(20261028).permutation(len(ids))
        lab = {ids[i]: "test" for i in perm[:40]}
        lab.update({ids[i]: "val" for i in perm[40:56]})
        pool = [ids[i] for i in perm[56:]] + extra
        for k, gid in enumerate(pool):
            lab[gid] = "train" if k < n_train else "unused"
        for gid, s in lab.items():
            f["geometries"][gid].attrs["split"] = s


def train(tag, n, epochs, seed):
    if (OUT / f"{tag}.pred.npz").exists():
        return
    label(ATLAS_W, n)
    spe = int(round(65536 * n / 278))
    model = OUT / f"{tag}.pt"
    t0 = time.perf_counter()
    r = subprocess.run([R.PY, "-u", str(R.SUR), "train", "--atlas", str(ATLAS_W), "--output", str(model), "--device", "cpu",
                        "--epochs", str(epochs), "--samples-per-epoch", str(spe), "--seed", str(seed), "--print-every", "20"]
                       + (["--resume"] if (OUT / f"{tag}.latest.pt").exists() else []),
                       capture_output=True, text=True)
    (OUT / f"{tag}.train.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        print(f"FAILED {tag}", r.stderr[-1500:], flush=True)
        return
    tmin = (time.perf_counter() - t0) / 60
    save_preds(tag, model, OUT / f"{tag}.latest.pt", dict(n=n, epochs=epochs, seed=seed, spe=spe, train_minutes=tmin,
                                                          threads=torch.get_num_threads()))
    print(f"{time.strftime('%H:%M')} trained {tag} in {tmin:.1f} min", flush=True)


EXISTING = [  # tag, file stem, n, epochs, seed, spe, minutes
    ("old_n100_e80", "sur_100", 100, 80, 20261003, 65536, 2.89), ("old_n200_e80", "sur_200", 200, 80, 20261003, 65536, 3.13),
    ("old_n278_e80", "sur_278", 278, 80, 20261003, 65536, 3.06),
    ("n278_e240_s20261003", "sur_278_e240", 278, 240, 20261003, 65536, None),
    ("n278_e240_s777", "sur2_278_seed777", 278, 240, 777, 65536, 9.2),
    ("n450_e240_s20261003", "sur2_450", 450, 240, 20261003, 106117, 16.1),
    ("n628_e240_s20261003", "sur2_628", 628, 240, 20261003, 148032, 22.62),
]

QUEUE = (
    [(f"n{n}_e240_s1", n, 240, 1) for n in (50, 100, 200, 278, 450, 628)]
    + [(f"n628_e{e}_s20261003", 628, e, 20261003) for e in (10, 20, 40, 80, 160)]
    + [(f"n{n}_e240_s2", n, 240, 2) for n in (50, 100, 200, 450, 628)]
    + [(f"n{n}_e240_s3", n, 240, 3) for n in (50, 100, 200)]
    + [("n628_e480_s20261003", 628, 480, 20261003)]
)


def claim(tag):
    """atomic claim of a queue item across workers (lock file)."""
    try:
        fd = os.open(OUT / f"{tag}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


WORKER, NWORK = (int(os.environ.get("TRAIN_WORKER", 0)), int(os.environ.get("TRAIN_NWORK", 1)))
ATLAS_W = LC / f"combined_w{WORKER}.h5"

if __name__ == "__main__":
    if not ATLAS_W.exists():
        import shutil
        shutil.copy2(R.ATLAS, ATLAS_W)
    for tag, stem, n, e, s, spe, mins in EXISTING:
        if not (OUT / f"{tag}.pred.npz").exists():
            save_preds(tag, LC / f"{stem}.pt", LC / f"{stem}.latest.pt", dict(n=n, epochs=e, seed=s, spe=spe, train_minutes=mins, existing=True))
    for tag, n, e, s in QUEUE:
        if not (OUT / f"{tag}.pred.npz").exists() and claim(tag):
            train(tag, n, e, s)
    print("ALLDONE", flush=True)
