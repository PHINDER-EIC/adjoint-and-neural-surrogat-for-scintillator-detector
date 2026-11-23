"""Clean speed benchmarks (run on an otherwise idle machine). Writes study/bench.json.

usage: python bench.py [adjoint] [jax] [g4] [network]   (default: all)
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

T = paths.DATA
ST = T / "derived"
ROOT = Path(str(paths.REPO))
PY = sys.executable
GPY = __import__("os").environ.get("G4_PYTHON", __import__("sys").executable)
OUT = ST / "bench.json"
GEOMS = ["A_train_g0013", "B_train_g0180", "A_train_g0052"]
CPU = "Intel Core Ultra 5 245 (14 cores)"


def load():
    return json.load(open(OUT)) if OUT.exists() else {}


def save(b):
    json.dump(b, open(OUT, "w"), indent=1)


def bench_adjoint(threads=None):
    sys.path[:0] = [str(ROOT / "Scintillator-Simulator-main/extras/numba_adjoint"), str(ROOT / "phinder_bridge/scripts"),
                    str(ROOT / "Scintillator-Simulator-main/src")]
    import numba
    import numba_adjoint as na
    import phinder_stage6_adjoint_geometry_atlas as atlasmod
    a = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005,
                              max_live_branches=64, max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0,
                              start_epsilon_cm=1e-7)
    bank = {g["id"]: g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"]}
    res = dict(threads=numba.get_num_threads(), cpu=CPU, geometries=GEOMS, N=[1000, 2000, 5000, 10000, 20000, 50000],
               wall_s={}, build_s={}, products_s={}, n_volumes={}, compile_s=None)
    for gid in GEOMS:
        t0 = time.perf_counter()
        spec, p0, p1, cfg = atlasmod._geometry_cfg(bank[gid], a, 11, 22)
        tr = na.from_stage6_cfg(cfg)
        res["build_s"][gid] = time.perf_counter() - t0
        res["n_volumes"][gid] = len(tr.scene.volumes)
        if res["compile_s"] is None:
            t0 = time.perf_counter(); tr.run(64, 1, 2); res["compile_s"] = time.perf_counter() - t0
        tr.run(500, 3, 4)
        w = []
        for N in res["N"]:
            t0 = time.perf_counter(); m, _ = tr.run(N, 5, 6); w.append(time.perf_counter() - t0)
        res["wall_s"][gid] = w
        t0 = time.perf_counter()
        atlasmod._compute_products(bank[gid], spec, p0, p1, cfg, m, a, "bench", 5, 6, 0.0)
        res["products_s"][gid] = time.perf_counter() - t0
        print("adjoint", gid, res["n_volumes"][gid], [round(x, 2) for x in w], flush=True)
    return res


def bench_jax():
    code = r'''
import sys, time, json, types, numpy as np
sys.argv = ["x", "B_train_g0180"]
ROOT = r"__REPO__"
sys.path[:0] = [ROOT + "/phinder_bridge/scripts", ROOT + "/Scintillator-Simulator-main/src", ROOT + "/Scintillator-Simulator-main/extras/jax_compaction"]
import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp
import phinder_stage6_adjoint_geometry_atlas as atlasmod, phinder_stage6_adjoint_prototype as proto
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch
from Photon_Simulator.core.photonsEngines.JaxEngine import JaxApproximateEngine, JaxTrackingConfig
from compaction import track_photon_batch_compacted
raw = next(g for g in json.load(open(r"__BANK__"))["geometries"] if g["id"] == "B_train_g0180")
a = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005, max_live_branches=64, max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0, start_epsilon_cm=1e-7)
spec, _, _, cfg = atlasmod._geometry_cfg(raw, a, 1, 2); scene, *_ = proto._build_from_cfg(cfg)
X, Y, Z = map(float, cfg["cell_cm"])
eng = JaxApproximateEngine(scene, JaxTrackingConfig(max_steps=200000), dtype=jnp.float64)
rng = np.random.default_rng(1); out = {}
for B in (1000, 4000, 16000):
    def batch(k):
        pts = rng.random((B, 3)) * [X, Y, Z]; v = rng.normal(size=(B, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        o = track_photon_batch_compacted(eng, PhotonBatch(pts, v, np.zeros(B), np.full(B, -1, np.int32)), key=k); np.asarray(o.statuses)
    t0 = time.perf_counter(); batch(0); first = time.perf_counter() - t0
    reps = max(2, 24000 // B); t0 = time.perf_counter(); c0 = time.process_time()
    for k in range(reps): batch(k + 1)
    w = time.perf_counter() - t0; c = time.process_time() - c0
    out[B] = dict(first_batch_s=first, photons_per_s=B * reps / w, cpu_over_wall=c / w)
    print("jax", B, out[B], flush=True)
print("JSON" + json.dumps(out))
'''
    r = subprocess.run([PY, "-c", code.replace("__REPO__", str(paths.REPO)).replace("__BANK__", str(paths.DATA / "banks/bank_test40.json"))], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        print(r.stdout[-2000:], r.stderr[-3000:]); return None
    return dict(geometry="B_train_g0180", batches=json.loads(line[0][4:]))


def bench_g4():
    env = dict(os.environ, G4ENSDFSTATEDATA=__import__("os").environ.get("G4ENSDFSTATEDATA", ""),
               GEANT4_DATA_DIR=str(ST / "g4" / "nodata"))
    out = dict(geometry="B_train_g0180")
    tmp = ST / "g4" / "bench"; tmp.mkdir(exist_ok=True)
    for mode in ("unpol", "pol"):
        t0 = time.perf_counter()
        subprocess.run([GPY, "-u", str(ST / "g4_sim.py"), "B_train_g0180", mode, "0,0,0", "20,20,8", "50000", "7", str(tmp / f"single_{mode}.npz")],
                       env=env, capture_output=True)
        d = np.load(tmp / f"single_{mode}.npz")
        out[f"single_{mode}"] = dict(photons_per_s=50000 / float(d["wall_s"]), process_s=time.perf_counter() - t0)
        print("g4 single", mode, out[f"single_{mode}"], flush=True)
    procs, t0 = [], time.perf_counter()
    for i in range(14):
        procs.append(subprocess.Popen([GPY, "-u", str(ST / "g4_sim.py"), "B_train_g0180", "pol", "0,0,0", "20,20,8", "30000", str(100 + i),
                                       str(tmp / f"par_{i}.npz")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    wall = time.perf_counter() - t0
    beam = max(float(np.load(tmp / f"par_{i}.npz")["wall_s"]) for i in range(14))
    out["parallel14_pol"] = dict(photons_per_s=14 * 30000 / beam, photons_per_s_incl_startup=14 * 30000 / wall)
    print("g4 parallel", out["parallel14_pol"], flush=True)
    return out


def bench_network():
    import torch
    sys.path.insert(0, str(ROOT / "phinder_bridge/scripts"))
    import phinder_stage6B_surrogate as S
    TRUTH = T / "reference_test40_2e5roots.h5"
    data = S.load_split_data(TRUTH, ["B_train_g0180"])
    model, _ = S.load_model(T / "surrogate_lc/sur2_628.latest.pt", torch.device("cpu")); model.eval()
    out = dict(params=int(sum(p.numel() for p in model.parameters())))
    geom, nodes, uvs, *_ = S._batch_from_indices(data, np.arange(data.n_vox), torch.device("cpu"))
    for th in (14, 1):
        torch.set_num_threads(th)
        with torch.no_grad():
            for _ in range(3):
                o = model(geom, nodes, uvs); model.timing_mass(o["mix0_logits"])
            t0 = time.perf_counter()
            for _ in range(20):
                o = model(geom, nodes, uvs); model.timing_mass(o["mix0_logits"]); model.mean_time(o["mix0_logits"])
            out[f"map_s_threads{th}"] = (time.perf_counter() - t0) / 20
        g = geom.clone().requires_grad_(True); n = nodes.clone().requires_grad_(True)
        t0 = time.perf_counter()
        for _ in range(10):
            o = model(g, n, uvs); (o["p0"].sum() + o["p1"].sum()).backward()
        out[f"map_plus_grad_s_threads{th}"] = (time.perf_counter() - t0) / 10
        print("network", th, out, flush=True)
    torch.set_num_threads(14)
    # 40 test geometries in one call
    data40 = S.load_split_data(TRUTH, sorted(json.load(open(T / "banks/bank_test40.json")) and
                                             [g["id"] for g in json.load(open(T / "banks/bank_test40.json"))["geometries"]]))
    ii = np.arange(data40.n_geom * data40.n_vox)
    with torch.no_grad():
        geom, nodes, uvs, *_ = S._batch_from_indices(data40, ii, torch.device("cpu"))
        t0 = time.perf_counter(); o = model(geom, nodes, uvs); model.timing_mass(o["mix0_logits"])
        out["maps40_s"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["adjoint", "jax", "g4", "network"]
    b = load()
    b["host"] = dict(cpu=CPU, python=platform.python_version(), when=time.strftime("%Y-%m-%d %H:%M"))
    for w in what:
        r = dict(adjoint=bench_adjoint, jax=bench_jax, g4=bench_g4, network=bench_network)[w]()
        if r is not None:
            b[w] = r
            save(b)
    print("DONE", flush=True)
