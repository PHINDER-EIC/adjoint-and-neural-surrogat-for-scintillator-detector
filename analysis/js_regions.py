"""Jensen-Shannon divergence of the channel-0 arrival-time distribution f_0(t) in the Table III regions, each method vs
Geant4 (unpolarized). Shapes only: each histogram is normalized over 0-60 ns. Bins as in Fig. 6: 0.25 ns to 2 ns,
1 ns to 10 ns, 2 ns to 60 ns (41 bins), so that the Geant4 histograms are not dominated by empty tail bins.
JS in bits (base 2; 0 = identical, 1 = disjoint).

Counting noise makes JS positive even for identical parents. 'floor' = expected JS from the photon counts alone,
by drawing Geant4 (and, for JAX, JAX) multinomial samples of the actual sizes from the method's own shape
(200 draws per region). Network = each of the three seeds (sur2_628_final + two others); per-region values averaged over seeds.
Writes study/js_regions.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import numpy as np
import torch

sys.argv = sys.argv[:1]
import common as C  # noqa: E402
import regions as RG  # noqa: E402

S = RG.S
PATHS = [C.T / "networks/network_seed20261003.pt", C.T / "networks/network_seed1.pt", C.T / "networks/network_seed2.pt"]
IDX = np.concatenate([np.arange(0, 8), np.arange(8, 40, 4), np.arange(40, 241, 8)])
rng = np.random.default_rng(5)


def coarse(h):
    c = np.add.reduceat(np.asarray(h, float), IDX[:-1])
    return c / c.sum()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(m > 0, m, 1)), 0))
    return 0.5 * kl(p) + 0.5 * kl(q)


def floor(shape, n_g4, n_m=None, draws=200):
    v = []
    for _ in range(draws):
        a = rng.multinomial(n_g4, shape) / n_g4
        b = shape if n_m is None else rng.multinomial(n_m, shape) / n_m
        v.append(js(a, b))
    return float(np.mean(v))


def main():
    R = json.load(open(C.ST / "regions.json"))["regions"]
    H = dict(np.load(C.ST / "regions_hist_single.npz"))
    models = [S.load_model(p, torch.device("cpu"))[0].eval() for p in PATHS]
    rows = []
    for gid, regs in R.items():
        defs = json.loads(str(np.load(C.ST / "g4/geom" / f"{gid}.npz")["regions_json"]))
        nets = [RG.net_voxels(gid, [m]) for m in models]
        for name, (lo, hi) in defs.items():
            sl = C.region_slices(lo, hi)
            row = regs[name]
            g = coarse(H[f"{gid}|{name}|g4_unpol"]); ng = row["g4_unpol"]["caught0"]
            out = dict(gid=gid, region=name, n_g4=ng)
            a = coarse(H[f"{gid}|{name}|adjoint"])
            out["adjoint"] = js(a, g); out["adjoint_floor"] = floor(a, ng)
            if "jax" in row:
                j = coarse(H[f"{gid}|{name}|jax"]); nj = row["jax"]["caught0"]
                out["jax"] = js(j, g); out["jax_floor"] = floor(j, ng, nj); out["n_jax"] = nj
            vals, fl = [], []
            for pn0, _, mn in nets:
                mreg = (mn[sl] * pn0[sl][..., None]).reshape(-1, 240).sum(0)
                s = coarse(mreg); vals.append(js(s, g)); fl.append(floor(s, ng, draws=100))
            out["network"] = float(np.mean(vals)); out["network_seeds"] = vals; out["network_floor"] = float(np.mean(fl))
            rows.append(out)
        print(gid, "done", flush=True)
    summ = {}
    for m in ("adjoint", "jax", "network"):
        v = np.array([r[m] for r in rows if m in r]); f = np.array([r[m + "_floor"] for r in rows if m in r])
        summ[m] = dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)), max=float(v.max()),
                       floor_mean=float(f.mean()), excess_mean=float((v - f).mean()), ratio_to_floor=float(v.mean() / f.mean()))
    json.dump(dict(bins="Fig. 6 coarse bins, 41 bins over 0-60 ns", units="bits", summary=summ, rows=rows),
              open(C.ST / "js_regions.json", "w"), indent=1)
    for m, s in summ.items():
        print(f"{m:8} n={s['n']:2d}  JS mean {1e3*s['mean']:.2f}e-3 (median {1e3*s['median']:.2f}, max {1e3*s['max']:.2f})  "
              f"counting floor {1e3*s['floor_mean']:.2f}e-3  ratio {s['ratio_to_floor']:.2f}")
    for r in rows:
        print(f"{r['gid']:15} {r['region']:14} nG4 {r['n_g4']:6d}  adj {1e3*r['adjoint']:.2f}/{1e3*r['adjoint_floor']:.2f}  "
              f"net {1e3*r['network']:.2f}/{1e3*r['network_floor']:.2f}" + (f"  jax {1e3*r['jax']:.2f}/{1e3*r['jax_floor']:.2f}" if 'jax' in r else ""))


if __name__ == "__main__":
    main()
