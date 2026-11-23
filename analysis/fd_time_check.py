"""Check: truncated-histogram mean (fd_time.py) vs untruncated volume mean transport time (fd_study.py) on the same runs.
Rows with the largest old/new dt discrepancy; 8 seed pairs, 5e4 roots. Writes fd/time_check.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import numpy as np
import fd_time as T
F = T.F
ROWS = ["A_train_g0042|turn2_y", "A_train_g0042|cell_z", "B_val_g0025|width"]
rows = json.load(open(F.OUT / "adjoint.json"))
out = {}
for key in ROWS:
    r = rows[key]; u0 = np.array(r["u0"]); k = r["k"]; du = r["du"]
    d = {"H": [], "full": [], "cov": []}
    for i, seeds in enumerate(T.SEEDS):
        v = {}
        for sgn in (1, -1):
            u = u0.copy(); u[k] += sgn * du
            raw = F.raw_from_u(u, f"{r['gid']}__{r['param']}__{'p' if sgn > 0 else 'm'}"); raw["spacefill_vector"] = u.tolist()
            _, _, _, prod = F.run(raw, seeds)
            v[sgn] = (T.stats(prod)["t"], prod["summary"]["volume_mean_transport_time_ns"],
                      float(prod["h"].sum() / prod["p"].sum()))
        d["H"].append((v[1][0] - v[-1][0]) / (2 * du)); d["full"].append((v[1][1] - v[-1][1]) / (2 * du)); d["cov"].append(v[1][2])
        print(key, i, round(d["H"][-1], 3), round(d["full"][-1], 3), flush=True)
    out[key] = {q: [float(np.mean(d[q])), float(np.std(d[q], ddof=1) / np.sqrt(len(d[q])))] for q in d}
    out[key]["diff_full_minus_H"] = [float(np.mean(np.subtract(d["full"], d["H"]))), float(np.std(np.subtract(d["full"], d["H"]), ddof=1) / np.sqrt(8))]
    print(key, out[key], flush=True)
json.dump(out, open(F.OUT / "time_check.json", "w"), indent=1)
print("ALLDONE")
