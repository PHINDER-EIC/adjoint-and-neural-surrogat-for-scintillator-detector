"""Finite-difference derivatives of whole-cell timing quantities with replicated seed pairs.

For every derivative in fd/adjoint.json (same designs, parameter and step du), both signs are run with NREP independent
seed pairs (common random numbers within a pair), ROOTS roots each. From each map: whole-cell detection probability
(voxel mean of p, channel 0) and the whole-cell arrival-time distribution H(t) = voxel mean of h (uniform emission), giving
mean, standard deviation and 10% quantile t10 (linear within 0.25-ns bins). Derivative per pair = central difference;
reported value = mean over pairs, error = s.d. / sqrt(NREP) (NREP-1 degrees of freedom).
Results: fd/time_runs.json (per run, restartable) and fd/time_derivs.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import os
import sys

import numpy as np

sys.argv = sys.argv[:1]
import fd_study as F  # noqa: E402

NREP = int(os.environ.get("NREP", 8))
F.ROOTS = int(os.environ.get("ROOTS", 50000))
SEEDS = [(7_700_101 + i, 9_900_101 + i) for i in range(NREP)]
TC = (np.arange(240) + 0.5) * 0.25
EDGES = np.arange(241) * 0.25
RUNS = F.OUT / "time_runs.json"


def stats(prod):
    p = float(prod["p"].mean())
    H = prod["h"].reshape(-1, 240).mean(0)
    m = H.sum()
    mu = float((H * TC).sum() / m)
    sd = float(np.sqrt((H * TC ** 2).sum() / m - mu ** 2))
    c = np.cumsum(H) / m
    j = int(np.searchsorted(c, 0.1))
    c0 = c[j - 1] if j > 0 else 0.0
    t10 = float(EDGES[j] + (0.1 - c0) / (c[j] - c0) * 0.25)
    return dict(p=p, t=mu, sd=sd, t10=t10)


def quantile_stats(H):
    return H


def main():
    rows = {k: v for k, v in json.load(open(F.OUT / "adjoint.json")).items() if v.get("ok")}
    runs = json.load(open(RUNS)) if RUNS.exists() else {}
    for key, r in rows.items():
        u0 = np.array(r["u0"]); k = r["k"]; du = r["du"]
        for sgn in (1, -1):
            u = u0.copy(); u[k] += sgn * du
            raw = F.raw_from_u(u, f"{r['gid']}__{r['param']}__{'p' if sgn > 0 else 'm'}")
            raw["spacefill_vector"] = u.tolist()
            for i, seeds in enumerate(SEEDS):
                tag = f"{key}|{sgn}|{i}"
                if tag in runs:
                    continue
                _, _, _, prod = F.run(raw, seeds)
                runs[tag] = stats(prod)
                json.dump(runs, open(RUNS, "w"))
        print(key, "done", flush=True)
    out = {}
    for key, r in rows.items():
        d = {}
        for q in ("p", "t", "sd", "t10"):
            vals = []
            for i in range(NREP):
                a, b = runs[f"{key}|1|{i}"][q], runs[f"{key}|-1|{i}"][q]
                vals.append((np.log(a) - np.log(b) if q == "p" else a - b) / (2 * r["du"]))
            vals = np.array(vals)
            d[q] = float(vals.mean()); d[q + "_se"] = float(vals.std(ddof=1) / np.sqrt(NREP))
        d.update(gid=r["gid"], param=r["param"], du=r["du"],
                 base={q: float(np.mean([runs[f"{key}|{s}|{i}"][q] for s in (1, -1) for i in range(NREP)])) for q in ("p", "t", "sd", "t10")})
        out[key] = d
        print(f"{key:28} dlnp {d['p']:+.3f}+-{d['p_se']:.3f}  dt {d['t']:+.3f}+-{d['t_se']:.3f}  dsd {d['sd']:+.3f}+-{d['sd_se']:.3f}  "
              f"dt10 {d['t10']:+.3f}+-{d['t10_se']:.3f}", flush=True)
    json.dump(out, open(F.OUT / "time_derivs.json", "w"), indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
