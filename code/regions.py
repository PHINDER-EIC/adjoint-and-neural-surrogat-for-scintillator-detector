import paths  # release: data and repository locations
"""Region-level comparison: JAX, Geant4 (unpol / pol / full), adjoint 200k map, network surrogate.

For every geometry with JAX or Geant4 results and every emission region: captured fraction in channels 0/1
with standard errors, mean transport time and time s.d. of channel 0 (s.d. over 0-60 ns, as the maps), and the
channel-0 arrival-time density on the 0.25-ns grid. Written to study/regions.json + regions_hist.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import common as C

REPO = Path(str(paths.REPO))
sys.path.insert(0, str(REPO / "phinder_bridge/scripts"))
import phinder_stage6B_surrogate as S  # noqa: E402

EDGES = np.arange(241) * 0.25
MODELS = [Path(a) for a in sys.argv[1:]] or [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt",
                                           C.T / "networks/network_seed2.pt"]   # ensemble of final weights


def net_voxels(gid, models):
    """ensemble mean of p0, p1 and of the channel-0 timing mass (p0-weighted)."""
    data = S.load_split_data(C.TRUTH_H5, [gid])
    P0, P1, MM = 0, 0, 0
    with torch.no_grad():
        geom, nodes, uvs, *_ = S._batch_from_indices(data, np.arange(data.n_vox), torch.device("cpu"))
        for model in models:
            o = model(geom, nodes, uvs)
            p0 = o["p0"].numpy()
            P0 = P0 + p0; P1 = P1 + o["p1"].numpy()
            MM = MM + model.timing_mass(o["mix0_logits"]).numpy() * p0[:, None]
    P0, P1 = P0 / len(models), P1 / len(models)
    m0 = MM / len(models) / P0[:, None]
    return (P0.reshape(20, 20, 8), P1.reshape(20, 20, 8), m0.reshape(20, 20, 8, 240))


def tstats(t):
    t60 = t[t < 60]
    return dict(mean=float(t.mean()) if t.size else np.nan, mean_se=float(t.std() / np.sqrt(max(t.size, 1))),
                sd60=float(t60.std()) if t60.size > 1 else np.nan,
                sd60_se=float(t60.std() / np.sqrt(2 * max(t60.size - 1, 1))) if t60.size > 1 else np.nan)


def main():
    tr = C.truth()
    J = C.jax_runs()
    G4 = C.g4_runs()
    models = []
    for mp in MODELS:
        if mp.exists():
            mdl, _ = S.load_model(mp, torch.device("cpu")); mdl.eval(); models.append(mdl)
    gids = sorted(set(J) | {k[0] for k in G4})
    res, hist = {}, {}
    for gid in gids:
        gi = tr["ids"].index(gid)
        p = tr["p"][gi].reshape(20, 20, 8); pse = tr["p_se"][gi].reshape(20, 20, 8)
        mt = tr["mt"][gi].reshape(20, 20, 8)
        h = C.h_truth(gid)
        pn0, pn1, mn = net_voxels(gid, models)
        regs = json.loads(str(np.load(C.ST / "g4/geom" / f"{gid}.npz")["regions_json"]))
        for name, (lo, hi) in regs.items():
            sl = C.region_slices(lo, hi)
            slm = C.region_slices([20 - hi[0], lo[1], 8 - hi[2]], [20 - lo[0], hi[1], 8 - lo[2]])   # mirror for ch1
            row = {}
            # adjoint truth
            hreg = h[sl].reshape(-1, 240).mean(0)
            a_sd, _ = C.tsd_from_h(hreg)
            se0 = tr["pvol_se"][gi] if name == "whole" else float(pse[sl].mean())
            row["adjoint"] = dict(p0=float(p[sl].mean()), p0_se=float(se0), p1=float(p[slm].mean()),
                                  p1_se=float(tr["pvol_se"][gi] if name == "whole" else pse[slm].mean()),
                                  t_mean=float((p[sl] * mt[sl]).sum() / p[sl].sum()), t_sd60=float(a_sd))
            hist[f"{gid}|{name}|adjoint"] = hreg
            # network
            mreg = (mn[sl] * pn0[sl][..., None]).reshape(-1, 240).sum(0) / pn0[sl].sum()
            n_sd, n_mu = C.tsd_from_h(mreg)
            row["network"] = dict(p0=float(pn0[sl].mean()), p1=float(pn1[sl].mean()),
                                  t_mean=float((mreg * C.TC).sum()), t_sd60=float(n_sd))
            hist[f"{gid}|{name}|network"] = mreg * pn0[sl].mean() / 0.25
            # forward engines
            srcs = []
            if gid in J and name in J[gid]["regions"]:
                a = J[gid]["arrays"]
                srcs.append(("jax", J[gid]["regions"][name]["photons"], a[f"{name}_t0"], a[f"{name}_t1"].size,
                             J[gid]["regions"][name]["wall_s"]))
            for mode in ("unpol", "pol", "full"):
                k = (gid, mode, name)
                if k in G4:
                    o = G4[k]
                    srcs.append((f"g4_{mode}", o["n"], o["t"][o["ch"] == 0], int((o["ch"] == 1).sum()), o["wall_s"]))
            for src, n, t0, c1, wall in srcs:
                c0 = t0.size
                q0, q1 = c0 / n, c1 / n
                row[src] = dict(n=int(n), p0=q0, p0_se=float(np.sqrt(q0 * (1 - q0) / n)), p1=q1,
                                p1_se=float(np.sqrt(q1 * (1 - q1) / n)), caught0=int(c0), wall_s=float(wall),
                                **{f"t_{k}": v for k, v in tstats(t0).items()})
                hist[f"{gid}|{name}|{src}"] = np.histogram(t0, EDGES)[0] / (n * 0.25)
            res.setdefault(gid, {})[name] = row
        print(gid, "done", flush=True)
    json.dump(dict(models=[str(m) for m in MODELS], regions=res), open(C.ST / "regions.json", "w"), indent=1, default=float)
    np.savez_compressed(C.ST / "regions_hist.npz", **hist)


if __name__ == "__main__":
    main()
