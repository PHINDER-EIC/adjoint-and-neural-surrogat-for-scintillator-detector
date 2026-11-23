"""Scintillation photons from Geant4 deposits.

Per scintillator step: visible energy with Birks' law, E_vis = E / (1 + kB E/L) (kB = 0.126 mm/MeV, L = step length,
at least 1 um), photon number ~ Poisson(Y E_vis) with Y = 10^4 / MeV, positions uniform along the step, emission time =
step time (interpolated) + rise/decay delay (EJ-204: 0.7 ns rise, 1.8 ns decay, sampled as Exp(0.7) + Exp(1.8)).

usage: python make_photons.py <edep.npz> <n_muons_to_use> <seed> <photons.npz>
Writes pos [n,3] mm, t [n] ns, event [n], plus per-step expected photon numbers (mu) for the adjoint/network.
"""
import sys

import numpy as np

KB, Y, TR, TD = 0.126, 1.0e4, 0.7, 1.8


def visible(D):
    L = np.maximum(D["length"], 1e-3)
    return D["edep"] / (1 + KB * D["edep"] / L)


def main():
    D = dict(np.load(sys.argv[1]))
    nmu, seed, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rng = np.random.default_rng(seed)
    keep = D["event"] < nmu
    S = {k: v[keep] for k, v in D.items() if np.ndim(v) and len(v) == len(keep)}
    ev = visible(S)
    mu = Y * ev
    n = rng.poisson(mu)
    step = np.repeat(np.arange(len(n)), n)
    f = rng.random(len(step))
    pos = S["pre"][step] + f[:, None] * (S["post"][step] - S["pre"][step])
    t = S["t_pre"][step] + f * (S["t_post"][step] - S["t_pre"][step]) + rng.exponential(TR, len(step)) + rng.exponential(TD, len(step))
    order = rng.permutation(len(step))                     # mix events across Geant4 chunks
    np.savez_compressed(out, pos=pos[order], t=t[order], event=S["event"][step][order].astype(np.int32),
                        step=step[order].astype(np.int64), n_step=n, mu_step=mu, e_vis=ev,
                        step_mid=0.5 * (S["pre"] + S["post"]), step_t=0.5 * (S["t_pre"] + S["t_post"]),
                        step_event=S["event"].astype(np.int32), cell_mm=D["cell_mm"], n_muons=nmu)
    E = np.bincount(S["event"], weights=S["edep"], minlength=nmu); V = np.bincount(S["event"], weights=ev, minlength=nmu)
    print(f"{nmu} muons: {len(step)} photons ({len(step)/nmu:.0f}/muon), edep {E.mean():.3f} MeV, visible {V.mean():.3f} MeV, "
          f"steps {len(n)} ({len(n)/nmu:.1f}/muon)")


if __name__ == "__main__":
    main()
