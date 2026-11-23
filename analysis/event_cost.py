"""Per-event cost of turning a response map (or the surrogate) into the detected arrival-time distribution of an event.

Event model: a straight track through the cell with n_dep energy-deposit points (the photon yield is spread over them);
the expected detected intensity is lambda_c(t) = sum_k mu_k [s * h_c(.|r_k)](t), Eq. (superposition), with s the EJ-204
emission profile (0.7 ns rise, 1.8 ns decay). The detected-photon time histogram is then one multinomial draw, whose cost
does not depend on the number of photons. Writes study/event_cost.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common as C

REPO = Path(str(paths.REPO))
sys.path.insert(0, str(REPO / "phinder_bridge/scripts"))
import phinder_stage6B_surrogate as S  # noqa: E402

GID = "B_train_g0180"
rng = np.random.default_rng(1)
h = C.h_truth(GID).reshape(3200, 240) * 0.25               # detected per emitted photon, per 0.25-ns bin
tr = C.truth()
X, Y, Z = tr["cell"][tr["ids"].index(GID)]
t = (np.arange(240) + 0.5) * 0.25
s = (np.exp(-t / 1.8) - np.exp(-t / 0.7)); s /= s.sum()       # EJ-204 emission profile on the same grid
out = dict(geometry=GID, superposition={}, network_points={})


def track(n):
    a = np.array([0.1 * X, 0.2 * Y, Z]); b = np.array([0.9 * X, 0.7 * Y, 0.0])
    f = rng.random(n)
    return a + f[:, None] * (b - a)


def timeit(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


sp = np.array([X / 20, Y / 20, Z / 8])
for n in (10, 100, 1000, 10000):
    pts = track(n)
    vox = np.minimum((pts / sp).astype(int), [19, 19, 7])
    vi = (vox[:, 0] * 20 + vox[:, 1]) * 8 + vox[:, 2]
    mu = np.full(n, 1e9 / n)                                   # photons per deposit (total 1e9; cost independent of it)

    def ev():
        lam = mu @ h[vi]                                        # expected detected counts per bin, channel 0
        lam = np.convolve(lam, s)[:240]
        return rng.multinomial(rng.poisson(lam.sum()), lam / lam.sum())
    out["superposition"][n] = timeit(ev, 200 if n <= 1000 else 30)
    print("superposition", n, f"{1e3*out['superposition'][n]:.3f} ms", flush=True)

# deposits spread over the whole cell (shower-like): per-deposit gather vs binning into voxels first
out["shower"] = dict(n=[], voxels=[], gather_s=[], binned_s=[], network_s=[])
model, _ = S.load_model(C.T / "networks/network_seed20261003.pt", torch.device("cpu")); model.eval()
data = S.load_split_data(C.TRUTH_H5, [GID])
geom, nodes, uvs, *_ = S._batch_from_indices(data, np.arange(1), torch.device("cpu"))
for n in (1, 10, 100, 1000, 10000, 100000):
    pts = rng.random((n, 3)) * np.array([X, Y, Z])
    vox = np.minimum((pts / sp).astype(int), [19, 19, 7])
    vi = (vox[:, 0] * 20 + vox[:, 1]) * 8 + vox[:, 2]
    mu = np.full(n, 1e9 / n)

    def gather():
        lam = np.convolve(mu @ h[vi], s)[:240]
        return rng.multinomial(rng.poisson(lam.sum()), lam / lam.sum())

    def binned():
        lam = np.convolve(np.bincount(vi, weights=mu, minlength=3200) @ h, s)[:240]
        return rng.multinomial(rng.poisson(lam.sum()), lam / lam.sum())
    u = torch.as_tensor(pts / np.array([X, Y, Z]), dtype=torch.float32)
    g = geom.expand(n, -1); nd = nodes.expand(n, -1, -1)

    def net():
        with torch.no_grad():
            o = model(g, nd, u)
            lam = (model.timing_mass(o["mix0_logits"]) * o["p0"][:, None]).sum(0).numpy() * 1e9 / n
        lam = np.convolve(lam, s)[:240]
        return rng.multinomial(rng.poisson(lam.sum()), lam / lam.sum())
    reps = 200 if n <= 1000 else (20 if n <= 10000 else 5)
    out["shower"]["n"].append(n); out["shower"]["voxels"].append(int(len(np.unique(vi))))
    out["shower"]["gather_s"].append(timeit(gather, reps)); out["shower"]["binned_s"].append(timeit(binned, reps))
    out["shower"]["network_s"].append(timeit(net, max(3, reps // 4)))
    print(f"shower n={n:6d} voxels={out['shower']['voxels'][-1]:4d} gather {1e3*out['shower']['gather_s'][-1]:.3f} ms "
          f"binned {1e3*out['shower']['binned_s'][-1]:.3f} ms network {1e3*out['shower']['network_s'][-1]:.2f} ms", flush=True)

model, _ = S.load_model(C.T / "networks/network_seed20261003.pt", torch.device("cpu")); model.eval()
data = S.load_split_data(C.TRUTH_H5, [GID])
geom, nodes, uvs, *_ = S._batch_from_indices(data, np.arange(1), torch.device("cpu"))
for n in (10, 100, 1000, 10000):
    pts = track(n) / np.array([X, Y, Z])
    u = torch.as_tensor(pts, dtype=torch.float32)
    g = geom.expand(n, -1); nd = nodes.expand(n, -1, -1)

    def ev():
        with torch.no_grad():
            o = model(g, nd, u)
            mass = model.timing_mass(o["mix0_logits"]) * o["p0"][:, None]
            lam = (mass.sum(0)).numpy() * 1e9 / n
        lam = np.convolve(lam, s)[:240]
        return rng.multinomial(rng.poisson(lam.sum()), lam / lam.sum())
    out["network_points"][n] = timeit(ev, 50 if n <= 1000 else 10)
    print("network at deposit points", n, f"{1e3*out['network_points'][n]:.3f} ms", flush=True)
out["torch_threads"] = torch.get_num_threads()
json.dump(out, open(C.ST / "event_cost.json", "w"), indent=1)
