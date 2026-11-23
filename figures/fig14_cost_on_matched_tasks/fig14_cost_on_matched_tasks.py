"""Cost comparison at matched tasks, all on the same 14-core CPU (figM8_cost_matched).

Geant4 rate: g4_fast.py (no Python in the photon loop), 14 processes, g4_fast_bench.py -> g4_fast.json.

(a) One new geometry: wall time to obtain the map of the arrival-time distribution, measured by the voxel error of the
    mean arrival time t-bar (p-weighted rms relative difference to the 2e5-root reference, reference noise subtracted).
    Forward Monte Carlo: uniform emission of M photons, detected photons binned by emission voxel; t-bar of a voxel is the
    mean of its n_v = M p_v / 3200 detected photons, so eps^2 = (3200 / (M sum p)) sum_v (s_t,v / t_v)^2 (median over
    the 40 test geometries); time = M / rate. Open circles: Geant4 maps of B_train_g0180 (g4_fast_time.py).
    Adjoint: measured runs (time: speed_summary.json; t-bar error: adj_convergence.json, 40 test geometries).
    Network: test t-bar error (scaling.json, mean of 3 seeds) at its evaluation time; arrow: one-off build.
(b) Design scan, estimated: number of geometries x measured time per geometry, at the accuracy of a 2e4-root adjoint map
    (2.4% for p and 1.9% for t-bar; forward Monte Carlo needs the photon number of the harder of the two, t-bar).
(c) K single-particle events (26,500 photons) in one geometry: adjoint = one map + K x 0.4 ms; network = K x 2 ms
    (its build is shared by all geometries and not included); Geant4 = K x 26,500 / rate.
v2 (figM8_cost_matched_v2): panel letters only, line names in one legend, no network in (a), a marker where Geant4
reaches the 2e4-root adjoint accuracy, network inference-only line in (b). Writes no json (cost_matched.json unchanged).
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import common as C
import style as Sty

Sty.setup()
SP = json.load(open(C.out("speed_summary.json")))
G4F = json.load(open(C.ST / "g4_fast.json"))
RUNS = sorted(G4F.values(), key=lambda r: r["n"])
R_G4 = sum(r["n"] for r in RUNS) / sum(r["total_wall_s"] for r in RUNS)       # photons / s on 14 cores, incl. initialisation
R_JAX = SP["rates"]["jax"]
TS = {r["N"]: r["s"] for r in SP["adjoint"]}
conv = defaultdict(list)
for r in json.load(open(C.out("adj_convergence.json")))["rows"]:
    if r["steps"] == 5000:
        conv[r["N"]].append(r["t"]["rms"])
ADJ = [dict(N=N, s=TS[N], err=float(np.mean(conv[N]))) for N in sorted(TS) if N in conv]
SC = json.load(open(C.out("scaling.json")))
NET_T = float(np.mean([r["t"]["rms"] for r in SC["rows"] if r["n"] == 628 and r["epochs"] == 240
                       and r["which"] == "final" and not r["old"]]))
NET_S = SP["network"]["s"]
B_DATA, B_TRAIN = SP["surrogate_build_s"]["data"], SP["surrogate_build_s"]["train"]
BUILD = B_DATA + B_TRAIN
T_MAP = TS[20000]
E_T = next(a["err"] for a in ADJ if a["N"] == 20000)
NG_EV = 2.65e4
EV = json.load(open(C.ST / "events/events_timing.json"))
T_EV_ADJ, T_EV_NET = EV["adjoint_s"][0], EV["network_s"][0]

tr = C.truth()
K_T = float(np.median(3200 * ((tr["tsd"] / tr["mt"]) ** 2).sum(1) / tr["p"].sum(1)))   # M * eps_t^2
K_P = float(np.median(3200 * (1 - tr["p"]).sum(1) / tr["p"].sum(1)))                   # M * eps_p^2
M_MAP = max(K_T / E_T ** 2, K_P / SP["adjoint"][4]["err"] ** 2)                         # photons for a 2e4-root-equivalent map
eps = np.geomspace(0.006, 1.0, 60)
tf = {k: K_T / eps ** 2 / r for k, r in (("g4", R_G4), ("jax", R_JAX))}
out = dict(rate_g4=R_G4, rate_jax=R_JAX, build_s=BUILD, map_s=T_MAP, map_err_t=E_T, net_err_t=NET_T,
           photons_for_map=M_MAP, forward_map_s={k: M_MAP / r for k, r in (("g4", R_G4), ("jax", R_JAX))},
           event_s=dict(adjoint=T_EV_ADJ, network=T_EV_NET, g4=NG_EV / R_G4))

COLG4, COLA, COLN = Sty.COL["g4"], Sty.COL["adjoint"], Sty.COL["network"]
LAB = dict(fontsize=5.2, color=Sty.INK3)
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.95), layout="constrained")
G4_MAP_S = M_MAP / R_G4


def time_lines(ax, which, marks):
    from matplotlib.transforms import blended_transform_factory as btf
    for s_, lab in marks:
        if which == "x":
            ax.axvline(s_, color=Sty.INK3, lw=0.6, ls=":", zorder=0)
            ax.text(s_, 1.01, lab, ha="center", va="bottom", transform=btf(ax.transData, ax.transAxes), **LAB)
        else:
            ax.axhline(s_, color=Sty.INK3, lw=0.6, ls=":", zorder=0)
            ax.text(1.01, s_, lab, ha="left", va="center", transform=btf(ax.transAxes, ax.transData), **LAB)


# (a) one new geometry: wall time against voxel error of t-bar
ax = axs[0]
ax.plot(tf["g4"], eps, color=COLG4, lw=1.1)
ax.plot([r["total_wall_s"] for r in RUNS], [r["t_err"] for r in RUNS], "o", color=COLG4, ms=3, mfc="white", mew=0.9)
ax.plot([a["s"] for a in ADJ], [a["err"] for a in ADJ], "o-", color=COLA, ms=3, lw=1.1)
for a in ADJ:
    if a["N"] in (1000, 50000):
        ax.annotate(f"{a['N']//1000}k roots", (a["s"], a["err"]), xytext=(5, -2), textcoords="offset points", fontsize=5.3, color=COLA)
# equal accuracy: the 2e4-root adjoint map and the Geant4 run that reaches it
ax.plot([T_MAP, G4_MAP_S], [E_T, E_T], color=Sty.INK3, lw=0.6, ls=":", zorder=1)
ax.plot([G4_MAP_S], [E_T], "o", color=COLG4, ms=3.4)
ax.annotate(f"{G4_MAP_S/3600:.1f} h", (G4_MAP_S, E_T), xytext=(5, 3), textcoords="offset points", fontsize=5.3, color=COLG4)
ax.annotate(f"{T_MAP:.1f} s", (T_MAP, E_T), xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=5.3, color=COLA)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5e-2, 3e7); ax.set_ylim(0.006, 1.0)
time_lines(ax, "x", ((1, "1 s"), (3600, "1 h"), (86400, "1 d")))
Sty.logpct(ax, ticks=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0))
ax.set_xlabel("wall time for one new geometry (s)"); ax.set_ylabel(r"voxel error of mean arrival time $\bar t$")
Sty.panel(ax, "a")

# (b) scan of M geometries at the accuracy of a 2e4-root adjoint map
ax = axs[1]
M = np.geomspace(1, 1e5, 60)
ax.plot(M, M * G4_MAP_S, color=COLG4, lw=1.1)
ax.plot(M, M * T_MAP, color=COLA, lw=1.1)
ax.plot(M, BUILD + M * NET_S, color=COLN, lw=1.1)
ax.plot(M, M * NET_S, color=COLN, lw=1.0, ls="--")
mstar = BUILD / (T_MAP - NET_S)
ax.axvline(mstar, color=Sty.INK3, lw=0.6, ls="-.")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 1e5); ax.set_ylim(1e-2, 1e10)
time_lines(ax, "y", ((1, "1 s"), (3600, "1 h"), (86400, "1 d"), (3.15e7, "1 yr")))
ax.set_xlabel("geometries"); ax.set_ylabel("total wall time (s)")
Sty.panel(ax, "b")

# (c) K single-particle events (26,500 photons) in one geometry
ax = axs[2]
K = np.geomspace(1, 1e7, 60)
ax.plot(K, K * NG_EV / R_G4, color=COLG4, lw=1.1)
ax.plot([r["n"] / NG_EV for r in RUNS], [r["total_wall_s"] for r in RUNS], "o", color=COLG4, ms=3, mfc="white", mew=0.9)
ax.plot(K, T_MAP + K * T_EV_ADJ, color=COLA, lw=1.1)
ax.plot(K, K * T_EV_ADJ, color=COLA, lw=1.0, ls="--")
ax.plot(K, BUILD + K * T_EV_NET, color=COLN, lw=1.1)
ax.plot(K, K * T_EV_NET, color=COLN, lw=1.0, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 1e7); ax.set_ylim(1e-4, 1e9)
time_lines(ax, "y", ((1, "1 s"), (3600, "1 h"), (86400, "1 d"), (3.15e7, "1 yr")))
ax.set_xlabel("single-particle events in one geometry"); ax.set_ylabel("total wall time (s)")
Sty.panel(ax, "c")

L = plt.Line2D
H = [L([], [], color=COLG4, lw=1.1), L([], [], color=COLG4, lw=0, marker="o", ms=3, mfc="white", mew=0.9),
     L([], [], color=COLA, lw=1.1, marker="o", ms=3), L([], [], color=COLA, lw=1.0, ls="--"),
     L([], [], color=COLN, lw=1.1), L([], [], color=COLN, lw=1.0, ls="--"),
     L([], [], color=Sty.INK3, lw=0.6, ls="-.")]
NAMES = ["Geant4", "Geant4 runs (one geometry)",
         f"adjoint (incl. {T_MAP:.1f}-s map per geometry)", "adjoint, events only (map excluded)",
         f"network (incl. {BUILD/3600:.0f}-h build)", "network, inference only",
         f"break-even, network vs adjoint ({mstar:.0f} geometries)"]
fig.legend(H, NAMES, loc="outside upper center", ncol=4, fontsize=5.6, frameon=False, handlelength=2.6, columnspacing=1.4)
Sty.save(fig, "figM8_cost_matched_v2")
print("Geant4 map at the 2e4-root accuracy: %.2f h; break-even %.0f geometries" % (G4_MAP_S / 3600, mstar))
