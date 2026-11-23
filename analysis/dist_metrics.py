"""Kolmogorov-Smirnov distance (max |F-G|) and Wasserstein-1 distance (integral |F-G| dt, in ns) of the channel-0
arrival-time distribution, normalized over 0-60 ns on the 0.25-ns grid (CDF-based, so no rebinning needed).
(1) Table III regions vs Geant4, with the counting-noise floor (Geant4/JAX-size multinomial samples drawn from each
    method's own distribution, 200 per region; network 100);
(2) voxels (p-weighted mean) and whole cell vs the 2e5-root reference on the 80 test geometries, network (3 seeds)
    and 2e4-root adjoint maps. Writes study/dist_metrics.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import h5py
import numpy as np
import torch

sys.argv = sys.argv[:1]
import common as C  # noqa: E402
import js_regions as J  # noqa: E402
import regions as RG  # noqa: E402
import voxel_js as V  # noqa: E402

DT = 0.25
rng = np.random.default_rng(11)


def norm(h):
    h = np.asarray(h, float).reshape(-1, 240)
    s = h.sum(1, keepdims=True)
    return h / np.where(s > 0, s, 1)


def ks_w1(a, b):
    d = np.abs(np.cumsum(a, -1) - np.cumsum(b, -1))
    return d.max(-1), d.sum(-1) * DT


def floor(shape, n_g, n_m=None, draws=200):
    k, w = [], []
    for _ in range(draws):
        a = rng.multinomial(n_g, shape) / n_g
        b = shape if n_m is None else rng.multinomial(n_m, shape) / n_m
        x, y = ks_w1(a, b); k.append(x); w.append(y)
    return float(np.mean(k)), float(np.mean(w))


def regions():
    R = json.load(open(C.ST / "regions.json"))["regions"]
    H = dict(np.load(C.ST / "regions_hist_single.npz"))
    models = [RG.S.load_model(p, torch.device("cpu"))[0].eval() for p in J.PATHS]
    rows = []
    for gid, regs in R.items():
        defs = json.loads(str(np.load(C.ST / "g4/geom" / f"{gid}.npz")["regions_json"]))
        nets = [RG.net_voxels(gid, [m]) for m in models]
        for name, (lo, hi) in defs.items():
            sl = C.region_slices(lo, hi); row = regs[name]
            g = norm(H[f"{gid}|{name}|g4_unpol"])[0]; ng = row["g4_unpol"]["caught0"]
            out = dict(gid=gid, region=name)
            a = norm(H[f"{gid}|{name}|adjoint"])[0]
            out["adjoint"] = [float(v) for v in ks_w1(a, g)]; out["adjoint_floor"] = floor(a, ng)
            if "jax" in row:
                j = norm(H[f"{gid}|{name}|jax"])[0]
                out["jax"] = [float(v) for v in ks_w1(j, g)]; out["jax_floor"] = floor(j, ng, row["jax"]["caught0"])
            vs, fs = [], []
            for pn0, _, mn in nets:
                s = norm((mn[sl] * pn0[sl][..., None]).reshape(-1, 240).sum(0))[0]
                vs.append(ks_w1(s, g)); fs.append(floor(s, ng, draws=100))
            out["network"] = np.mean(vs, 0).tolist(); out["network_floor"] = np.mean(fs, 0).tolist()
            out["g4_tmean"] = row["g4_unpol"]["t_mean"]
            rows.append(out)
    summ = {}
    for m in ("adjoint", "jax", "network"):
        v = np.array([r[m] for r in rows if m in r]); f = np.array([r[m + "_floor"] for r in rows if m in r])
        summ[m] = dict(n=len(v), ks=float(v[:, 0].mean()), ks_floor=float(f[:, 0].mean()),
                       w1_ns=float(v[:, 1].mean()), w1_floor_ns=float(f[:, 1].mean()),
                       ks_max=float(v[:, 0].max()), w1_max_ns=float(v[:, 1].max()))
    return summ, rows


def voxels():
    models = [V.S.load_model(p, torch.device("cpu"))[0].eval() for p in J.PATHS]
    sets = []
    with h5py.File(C.TRUTH_H5) as f:
        for gid in sorted(f["geometries"]):
            pre, g = gid.split("_", 1)
            with h5py.File(V.ATL[pre]) as a:
                h20 = a["geometries"][g]["field/h_per_ns"][:]
            sets.append(("original", C.TRUTH_H5, gid, f["geometries"][gid]["field/p_per_emitted"][:].ravel(),
                         f["geometries"][gid]["field/h_per_ns"][:], h20))
    with h5py.File(V.BLIND) as f:
        for gid in sorted(g for g in f["geometries"] if g.startswith("test_")):
            sets.append(("new", V.BLIND, gid, f["geometries"][gid]["field/p_per_emitted"][:].ravel(),
                         f["geometries"][gid]["field/h_per_ns"][:], np.load(C.ST / "adj_new_h" / f"{gid}.npz")["h"]))
    rows = []
    for name, path, gid, p, href, h20 in sets:
        ref = norm(href); a20 = norm(h20)
        ok = (href.reshape(-1, 240).sum(1) > 0) & (h20.reshape(-1, 240).sum(1) > 0)
        w = np.where(ok, p, 0); w = w / w.sum()
        whole = lambda h: norm(np.asarray(h).reshape(-1, 240).sum(0))[0]
        k, d = ks_w1(a20, ref); kw, dw = ks_w1(whole(h20), whole(href))
        r = dict(set=name, gid=gid, adj20k=[float(np.sum(w * k)), float(np.sum(w * d))], adj20k_whole=[float(kw), float(dw)])
        vv, ww = [], []
        for p0, M in V.network(path, gid, models):
            k, d = ks_w1(norm(M), ref); vv.append([np.sum(w * k), np.sum(w * d)])
            ww.append(ks_w1(whole(M), whole(href)))
        r["net"] = np.mean(vv, 0).tolist(); r["net_whole"] = np.mean(ww, 0).tolist()
        rows.append(r)
    m = lambda k, i: float(np.mean([r[k][i] for r in rows]))
    summ = dict(n=len(rows), net_ks=m("net", 0), net_w1_ns=m("net", 1), adj20k_ks=m("adj20k", 0), adj20k_w1_ns=m("adj20k", 1),
                net_whole_ks=m("net_whole", 0), net_whole_w1_ns=m("net_whole", 1),
                adj20k_whole_ks=m("adj20k_whole", 0), adj20k_whole_w1_ns=m("adj20k_whole", 1))
    for s in ("original", "new"):
        R = [r for r in rows if r["set"] == s]
        summ[s] = dict(net_ks=float(np.mean([r["net"][0] for r in R])), net_w1_ns=float(np.mean([r["net"][1] for r in R])),
                       adj20k_w1_ns=float(np.mean([r["adj20k"][1] for r in R])))
    return summ, rows


if __name__ == "__main__":
    s1, r1 = regions()
    print(json.dumps(s1, indent=1), flush=True)
    s2, r2 = voxels()
    print(json.dumps(s2, indent=1), flush=True)
    json.dump(dict(regions=dict(summary=s1, rows=r1), voxels=dict(summary=s2, rows=r2)), open(C.ST / "dist_metrics.json", "w"), indent=1)
