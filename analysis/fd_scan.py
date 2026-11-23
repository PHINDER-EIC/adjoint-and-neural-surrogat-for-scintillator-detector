"""Line scans through the finite-difference cases: is the timing-derivative disagreement from the FD step, the reference
noise, or the network?

For a few derivatives of fd/time_derivs.json the adjoint is run at u0 + s e_k for s in S_ADJ (the +-du points already
exist in fd/time_runs.json), with the same 8 seed pairs and 5e4 roots as fd_time.py, so all points share random
numbers. Whole-cell p, t-bar, s_t, t10 as in fd_time.stats. The network is evaluated on a fine grid of s (geometry
features from a 500-root adjoint run, which the network does not use beyond the geometry).
Writes fd/scan_runs.json (restartable), fd/scan_net.json; geometry groups in fd/fd_scan.h5.
usage: python fd_scan.py adj|net
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import h5py
import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "adj"
sys.argv = sys.argv[:1]
import fd_study as F  # noqa: E402
import fd_time as FT  # noqa: E402

CASES = ["B_train_g0180|turn2_x", "B_train_g0035|r_over_w", "A_train_g0042|turn2_x", "B_train_g0035|cell_z",
         "B_train_g0195|cell_z", "B_val_g0025|cell_z"]
S_ADJ = [-0.16, -0.08, -0.04, 0.04, 0.08, 0.16]      # +-du already in fd/time_runs.json is not rerun
import os
if os.environ.get("ALL"):
    CASES = list(json.load(open(F.OUT / "time_derivs.json")))
S_NET = np.round(np.arange(-0.16, 0.1601, 0.01), 3)
RUNS = F.OUT / "scan_runs.json"
H5 = F.OUT / "fd_scan.h5"


def u_at(r, s):
    u = np.array(r["u0"]); k = r["k"]; u[k] += s
    if not (0 <= u[k] <= 1) or (k >= 9 and not np.all(np.diff(u[9:13]) >= 0)):
        return None
    return u


def adj():
    rows = json.load(open(F.OUT / "adjoint.json"))
    runs = json.load(open(RUNS)) if RUNS.exists() else {}
    for key in CASES:
        r = rows[key]
        for s in S_ADJ:
            if abs(abs(s) - r["du"]) < 1e-9:
                continue
            u = u_at(r, s)
            if u is None:
                print(key, s, "out of range", flush=True); continue
            raw = F.raw_from_u(u, f"{key.replace('|', '__')}__s{s:+.2f}")
            raw["spacefill_vector"] = u.tolist()
            for i, seeds in enumerate(FT.SEEDS):
                tag = f"{key}|{s:+.2f}|{i}"
                if tag in runs:
                    continue
                try:
                    spec, p0, p1, prod = F.run(raw, seeds)
                except Exception as e:
                    runs[tag] = dict(error=repr(e)[:200]); print(tag, "failed", e, flush=True); break
                runs[tag] = FT.stats(prod)
                json.dump(runs, open(RUNS, "w"))
            print(key, s, "done", flush=True)
    print("ALLDONE", flush=True)


def net():
    import torch
    import common as C
    import fd_net_time as FN
    S = FN.S
    rows = json.load(open(F.OUT / "adjoint.json"))
    models = [S.load_model(m, torch.device("cpu"))[0].eval() for m in FN.MODELS]
    out = {}
    F.ROOTS = 500
    for key in CASES:
        r = rows[key]
        out[key] = {}
        for s in S_NET:
            u = u_at(r, float(s))
            if u is None:
                continue
            gid = f"{key.replace('|', '__')}__n{s:+.3f}"
            raw = F.raw_from_u(u, gid); raw["spacefill_vector"] = u.tolist()
            try:
                spec, p0, p1, prod = F.run(raw, FT.SEEDS[0])
            except Exception as e:
                print(gid, "invalid", e, flush=True); continue
            with h5py.File(F.OUT / "fd_scan_net.h5", "a") as f:
                with h5py.File(F.OUT / "fd_atlas.h5", "r") as src:
                    for k in ("s_edges", "time_edges_ns", "u_edges", "v_edges"):
                        if k not in f:
                            f.create_dataset(k, data=src[k][:])
                if f"geometries/{gid}" not in f:
                    F.A._write_geometry_group(f, raw, spec, p0, p1, prod, F.ARGS)
            d = S.load_split_data(F.OUT / "fd_scan_net.h5", [gid])
            with torch.no_grad():
                geom, nodes, uvs, *_ = S._batch_from_indices(d, np.arange(d.n_vox), torch.device("cpu"))
                vals = []
                for mdl in models:
                    o = mdl(geom, nodes, uvs)
                    mass = mdl.timing_mass(o["mix0_logits"]).numpy(); mass = mass / mass.sum(1, keepdims=True)
                    p = o["p0"].numpy()
                    vals.append(FN.stats((p[:, None] * mass).mean(0), p.mean()))
            out[key][f"{s:+.3f}"] = vals
        print(key, len(out[key]), "points", flush=True)
        json.dump(out, open(F.OUT / "scan_net.json", "w"), indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    adj() if MODE == "adj" else net()
