"""Plot-ready data for adding the numba forward engine to Fig. 11 and Fig. 13 (no figure is changed).
Writes paper_v1/data/fig11_numba.csv, fig11_numba_lines.csv, fig13_this_machine.csv and numba_summary.json."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import csv, json
from pathlib import Path
import numpy as np
import common as C

D = paths.DATA / "derived" / "paper_v1_data"
nb = json.load(open(C.ST / "numba_fwd.json")); g4 = json.load(open(C.ST / "g4_fast.json"))
cm = json.load(open(C.out("cost_matched.json")))
maps = sorted(nb["maps"].values(), key=lambda r: r["n"])
R_NB = sum(r["n"] for r in maps) / sum(r["transport_s"] for r in maps)
R_G4 = cm["rate_g4"]
tr = C.truth()
K_T = float(np.median(3200 * ((tr["tsd"] / tr["mt"]) ** 2).sum(1) / tr["p"].sum(1)))
M_MAP = cm["photons_for_map"]; NG = 2.65e4

# measured points (same task as the Geant4 open circles of Fig. 11)
with open(D / "fig11_numba.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "photons", "transport_s", "total_s", "t_bar_err", "p_err", "events_equiv"])
    for r in maps:
        w.writerow(["numba", r["n"], round(r["transport_s"], 2), round(r["total_s"], 2), round(r["t_err"], 4),
                    round(r["p_err"], 4), round(r["n"] / NG, 1)])
    for r in sorted(g4.values(), key=lambda r: r["n"]):
        w.writerow(["geant4", r["n"], round(r["beam_wall_s"], 2), round(r["total_wall_s"], 2), round(r["t_err"], 4),
                    round(r["map_err"], 4), round(r["n"] / NG, 1)])
# lines: (a) time vs t-bar error, (b) per-geometry map time, (c) per-event time
eps = np.geomspace(0.006, 1.0, 25)
with open(D / "fig11_numba_lines.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["panel", "x", "numba_s", "geant4_s", "x_meaning"])
    for e in eps:
        w.writerow(["a", round(float(e), 5), round(K_T / e ** 2 / R_NB, 3), round(K_T / e ** 2 / R_G4, 3), "t-bar voxel error"])
    for m in np.geomspace(1, 1e5, 11):
        w.writerow(["b", round(float(m), 2), round(m * M_MAP / R_NB, 1), round(m * M_MAP / R_G4, 1), "geometries"])
    for k in np.geomspace(1, 1e7, 15):
        w.writerow(["c", round(float(k), 2), round(k * NG / R_NB, 4), round(k * NG / R_G4, 4), "single-particle events"])

# Fig. 13 task on this machine: N photons from one voxel at the cell centre
rows = []
for k, r in nb["point_scan"].items():
    rows.append([f"numba, {r['threads']} thread(s)", r["N"], round(r["transport_s"], 4), "", round(r["caught"], 4)])
for k, r in nb["point_scan_g4"].items():
    rows.append([f"Geant4, {r['processes']} process(es)", r["N"], round(r["transport_s"], 4), round(r["total_s"], 3),
                 round(r["caught"], 4)])
with open(D / "fig13_this_machine.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "photons", "transport_s", "total_s_incl_startup", "caught_fraction"]); w.writerows(rows)

S = dict(rate_numba_14threads=R_NB, rate_numba_1thread=nb["point_scan"]["1_1000000"]["N"] / nb["point_scan"]["1_1000000"]["transport_s"],
         rate_geant4_14proc=R_G4, rate_geant4_1proc=nb["point_scan_g4"]["1_1000000"]["N"] / nb["point_scan_g4"]["1_1000000"]["transport_s"],
         rate_jax_cpu_all_cores=cm["rate_jax"],
         numba_map_s=M_MAP / R_NB, geant4_map_s=M_MAP / R_G4, numba_event_s=NG / R_NB, geant4_event_s=NG / R_G4,
         numba_1e9_photons_h=1e9 / R_NB / 3600, numba_compile_s=nb["compile_and_first_call_s"],
         other_machine_from_fig13_png_1e6=dict(geant4_s=105, jax_cpu_s=600, jax_gpu_s=27),
         note="numba rate fell from 1.08e5 to 0.87e5 ph/s during the longest runs (sustained load); pooled value used.")
json.dump(S, open(D / "numba_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
