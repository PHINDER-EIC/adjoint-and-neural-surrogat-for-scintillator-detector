"""Launch phase-space check of the reverse tracer: cosine-weighted directions and uniform port coordinates."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json, sys, types
from pathlib import Path
import numpy as np
ROOT = Path(str(paths.REPO)); T = paths.DATA
sys.path[:0] = [str(ROOT / "Scintillator-Simulator-main/extras/numba_adjoint"), str(ROOT / "phinder_bridge/scripts"), str(ROOT / "Scintillator-Simulator-main/src")]
import numba_adjoint as na, phinder_stage6_adjoint_geometry_atlas as A
raw = next(g for g in json.load(open(T / "banks/bank_test40.json"))["geometries"] if g["id"] == "B_train_g0180")
a = types.SimpleNamespace(facets_per_90=16, nx=20, ny=20, nz=8, dt_ns=0.25, tmax_ns=60.0, split_min_child_weight=0.005, max_live_branches=64,
                          max_branch_nodes=1024, max_steps=5000, max_flight_time_ns=80.0, start_epsilon_cm=1e-7)
spec, p0, p1, cfg = A._geometry_cfg(raw, a, 606271, 808291)
tr = na.from_stage6_cfg(cfg)
N = 200000
u = tr.halton(N, 606271, 0)
L = tr.launch_rays(u)
print(type(L), [getattr(x, "shape", None) for x in (L if isinstance(L, tuple) else [L])])
start, d = L
mu = d @ tr.inward
a = (start - tr.endpoint) @ tr.lateral / tr.core_w + 0.5
b = (start - tr.endpoint) @ tr.vertical / tr.core_h + 0.5
from scipy import stats
res = dict(N=N, mean_cos=float(mu.mean()), mean_cos_expected=2/3, mean_cos2=float((mu**2).mean()), mean_cos2_expected=0.5,
           min_cos=float(mu.min()), ks_cos2_uniform_p=float(stats.kstest(mu**2, "uniform").pvalue),
           port_u_mean=float(a.mean()), port_v_mean=float(b.mean()), ks_port_u_p=float(stats.kstest(a, "uniform").pvalue),
           ks_port_v_p=float(stats.kstest(b, "uniform").pvalue), phi_mean_cos=float(np.mean((d @ tr.lateral) / np.sqrt(np.maximum(1 - mu**2, 1e-12)))))
json.dump(res, open(T / "derived/launch_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
