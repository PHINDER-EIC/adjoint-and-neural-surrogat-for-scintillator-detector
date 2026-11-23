"""Channel-0-only Table III statistics (for the rewritten Sec. VI A). Writes study/ch0_table3.json; changes nothing else.
sigma: Geant4, JAX binomial; adjoint from replicate maps (regions.json, region_noise.py); network: spread between the three
training seeds, pooled over all channel-0 region values. Network statistics: mean over the three seeds."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "code"))
import paths  # noqa: E402,F401
import json, sys
import numpy as np, torch
from scipy import stats
sys.argv = [sys.argv[0]]
import regions as RG, common as C
S = RG.S
R = json.load(open(C.ST / "regions.json"))["regions"]
paths = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
models = [S.load_model(p, torch.device("cpu"))[0].eval() for p in paths]

net = {}   # (gid, region) -> list over seeds of (p0, t_mean, t_sd60)
for gid in R:
    defs = json.loads(str(np.load(C.ST / "g4/geom" / f"{gid}.npz")["regions_json"]))
    for m in models:
        pn0, _, mn = RG.net_voxels(gid, [m])
        for name, (lo, hi) in defs.items():
            sl = C.region_slices(lo, hi)
            mreg = (mn[sl] * pn0[sl][..., None]).reshape(-1, 240).sum(0) / pn0[sl].sum()
            sd, _ = C.tsd_from_h(mreg)
            net.setdefault((gid, name), []).append((float(pn0[sl].mean()), float((mreg * C.TC).sum()), float(sd)))
keys = list(net)
P = np.array([[s[0] for s in net[k]] for k in keys])
dev = (P - P.mean(1, keepdims=True)) / P.mean(1, keepdims=True)
sig_net = float(np.sqrt(np.sum(dev ** 2) / (2 * len(keys))))

def stat(vals):   # vals: list of (m, m_se, g, g_se, t_m, t_g, s_m, s_g)
    v = np.array(vals); m, ms, g, gs = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    r = m / g; s = r * np.hypot(ms / m, gs / g); w = 1 / s ** 2
    chi2 = float(np.sum(((r - 1) / s) ** 2)); n = len(r)
    tm, ts = v[:, 4] / v[:, 5] - 1, v[:, 6] / v[:, 7] - 1
    return dict(n=n, ratio=float(np.sum(w * r) / w.sum()), ratio_se=float(1 / np.sqrt(w.sum())),
                rms=float(np.sqrt(np.mean((r - 1) ** 2))), chi2=chi2, p=float(stats.chi2.sf(chi2, n)),
                t_mean=float(tm.mean()), t_rms=float(np.sqrt(np.mean(tm ** 2))),
                s_mean=float(ts.mean()), s_rms=float(np.sqrt(np.mean(ts ** 2))),
                med_rel_se_method=float(np.median(ms / m)), med_rel_se_g4=float(np.median(gs / g)),
                rms_rel_se_g4=float(np.sqrt(np.mean((gs / g) ** 2))))

out = {}
for meth in ("adjoint", "jax"):
    vals = []
    for (gid, name) in keys:
        row = R[gid][name]
        if meth not in row: continue
        a, G = row[meth], row["g4_unpol"]
        vals.append((a["p0"], a["p0_se"], G["p0"], G["p0_se"], a["t_mean"], G["t_mean"], a["t_sd60"], G["t_sd60"]))
    out[meth] = stat(vals)
seeds = []
for k in range(3):
    vals = []
    for i, (gid, name) in enumerate(keys):
        G = R[gid][name]["g4_unpol"]; p, t, s = net[(gid, name)][k]
        vals.append((p, sig_net * p, G["p0"], G["p0_se"], t, G["t_mean"], s, G["t_sd60"]))
    seeds.append(stat(vals))
    vals0 = [(x[0], 0.0) + x[2:] for x in vals]
    seeds[-1]["chi2_sigma0"] = stat(vals0)["chi2"]
out["network_seeds"] = seeds
out["network"] = {k: float(np.mean([s[k] for s in seeds])) for k in seeds[0]}
out["network"]["chi2_seed_sd"] = float(np.std([s["chi2"] for s in seeds], ddof=1))
out["network"]["rms_seed_range"] = [min(s["rms"] for s in seeds), max(s["rms"] for s in seeds)]
out["network"]["p"] = float(stats.chi2.sf(out["network"]["chi2"], out["network"]["n"]))
out["sigma_net_pooled"] = sig_net
out["n_photons"] = {name: {m: sorted({R[g][name][m]["n"] for g in R if m in R[g][name]}) for m in ("g4_unpol", "jax")}
                    for name in R[keys[0][0]]}
noise = json.load(open(C.ST / "region_noise.json"))
out["adjoint_region_noise"] = noise
json.dump(out, open(C.ST / "ch0_table3.json", "w"), indent=1)
for m in ("adjoint", "jax", "network"):
    v = out[m]
    print(f"{m:8} n={v['n']:.0f} ratio {v['ratio']:.4f}+-{v['ratio_se']:.4f} rms {100*v['rms']:.2f}% chi2 {v['chi2']:.1f} p={v['p']:.3f} "
          f"t {100*v['t_mean']:+.2f}/{100*v['t_rms']:.2f} s {100*v['s_mean']:+.2f}/{100*v['s_rms']:.2f} "
          f"med se method {100*v['med_rel_se_method']:.2f}% g4 {100*v['med_rel_se_g4']:.2f}% (rms {100*v['rms_rel_se_g4']:.2f}%)")
print("network seeds:", [(round(100*s['rms'],2), round(s['chi2'],1), round(s['chi2_sigma0'],1)) for s in seeds],
      "chi2 sd", round(out["network"]["chi2_seed_sd"],1))
print("sigma_net pooled %.2f%%" % (100*sig_net))
print("photons:", out["n_photons"])
print("adjoint region noise keys:", list(noise)[:6])
