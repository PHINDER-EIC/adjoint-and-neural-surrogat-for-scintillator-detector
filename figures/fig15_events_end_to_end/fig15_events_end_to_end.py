"""figM7_events, two panels: (a) arrival times at readout 0 summed over all events, (b) detected photons per event,
Geant4 versus prediction. Uses the stored results of analyze_events.py (events_hist.npz); the cost panel of the
earlier version is replaced by Fig. cost_matched in the paper."""

import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import numpy as np
import matplotlib.pyplot as plt

import types
import common as C
# constants of analyze_events.py (the Geant4 event study that wrote events_hist.npz)
AE = types.SimpleNamespace(H=C.ST / "events", NMU=1000, DT=0.25, EDGES=np.arange(241) * 0.25)
import style as Sty

Sty.setup()
Hh = dict(np.load(AE.H / "events_hist.npz"))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.55), layout="constrained")

# (a) channel 0, summed over all events
k = 0
ax.stairs(np.maximum(Hh[f"g4_{k}"], 0.1) / AE.NMU / AE.DT, AE.EDGES, color=Sty.COL["g4"], lw=1.0, label="Geant4")
ax.stairs(np.maximum(Hh[f"adjoint 2e4_{k}"], 1e-3) / AE.NMU / AE.DT, AE.EDGES, color=Sty.COL["adjoint"], lw=0.9,
          label="adjoint map ($2\\times10^4$ roots)")
ax.stairs(np.maximum(Hh[f"network_{k}"], 1e-3) / AE.NMU / AE.DT, AE.EDGES, color=Sty.COL["network"], lw=0.9,
          label="network")
ax.set_yscale("log"); ax.set_xlim(0, 30); ax.set_ylim(0.05, 300)
ax.set_xlabel("arrival time at readout 0 (ns)"); ax.set_ylabel("detected photons per particle per ns")
ax.legend(loc="upper right", fontsize=5.6)
Sty.panel(ax, "a", f"{AE.NMU} events, summed")

# (b) detected photons per event, both channels
for name, col, mk, lab in (("adjoint 2e4", Sty.COL["adjoint"], "o", "adjoint map ($2\\times10^4$ roots)"),
                           ("network", Sty.COL["network"], "s", "network")):
    for c in (0, 1):
        bx.plot(Hh[f"{name}_lam{c}"], Hh[f"g4_n{c}"], mk, ms=1.4, mew=0, alpha=0.5, color=col, rasterized=True)
    bx.plot([], [], mk, ms=3, color=col, label=lab)
lim = [0, max(Hh["g4_n0"].max(), Hh["g4_n1"].max()) * 1.05]
bx.plot(lim, lim, color=Sty.INK3, lw=0.7, ls=":")
bx.set_xlim(lim); bx.set_ylim(lim)
bx.set_xlabel("predicted detected photons per event"); bx.set_ylabel("Geant4 detected photons per event")
bx.legend(loc="upper left", fontsize=5.6, handletextpad=0.2, borderaxespad=0.6)
Sty.panel(bx, "b", "Each event, both channels")
Sty.save(fig, "figM7_events")
