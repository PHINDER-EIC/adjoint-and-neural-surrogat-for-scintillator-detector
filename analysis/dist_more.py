"""More tests of the channel-0 arrival-time distribution f_0(t) (follow-up of dist_metrics.py / js_regions.py).

(1) Table III regions, each method vs Geant4 (unpolarized), shapes over 0-60 ns:
    a. Pearson chi2 of the Geant4 counts against the method's shape on the 41 coarse bins of Fig. 6, adjacent bins merged
       until the expected count is >= 10; ndf = bins - 1 (shape only). Forward engine: two-sample chi2 (both counted).
    b. quantile times t5, t10, t50, t90 (method - Geant4, ps), transport only and after convolution with the EJ-204
       emission profile (0.7 ns rise, 1.8 ns decay, as in event_cost.py).
    c. time-resolution bound per detected photon, sigma_1 = 1/sqrt(I), I = Fisher information for a time shift,
       only after convolution (transport-only f has a sharp onset, so I there depends on the binning, not the physics).
       d = 0.5 ln(I_g4 / <I_draw>) = ln(sigma_method / sigma_g4) with <I_draw> the mean over Geant4-size samples of the
       method's own shape (removes the bias that counting noise adds to I).
    d. Hellinger distance (coarse bins), Cramer-von Mises and Anderson-Darling distances (0.25-ns CDFs).
    Counting floors: the same statistics for Geant4-size multinomial samples of the method's own shape (and, for the
    forward engine, of its own size), 200 draws per region.
(2) Voxels and whole cell of the 80 test geometries vs the 2e5-root reference: quantile shifts and sigma_1 (convolved)
    for the network (3 seeds) and the 2e4-root adjoint map (= noise level).
Writes study/dist_more.json.
"""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json
import sys

import h5py
import numpy as np
import torch
from scipy.stats import chi2 as CHI2

sys.argv = sys.argv[:1]
import common as C  # noqa: E402
import js_regions as J  # noqa: E402
import regions as RG  # noqa: E402
import voxel_js as V  # noqa: E402

DT = 0.25
QS = (0.05, 0.10, 0.50, 0.90)
NDRAW = 200
rng = np.random.default_rng(23)
t = (np.arange(240) + 0.5) * DT
KER = np.exp(-t / 1.8) - np.exp(-t / 0.7); KER /= KER.sum()
KF = np.fft.rfft(KER, 512)


def norm(h):
    h = np.atleast_2d(np.asarray(h, float))
    s = h.sum(-1, keepdims=True)
    return h / np.where(s > 0, s, 1)


def conv(q):
    return norm(np.fft.irfft(np.fft.rfft(np.atleast_2d(q), 512, axis=-1) * KF, 512, axis=-1)[:, :240].clip(0))


def quant(q):
    """[N,240] shapes -> [N,len(QS)] quantile times (ns), linear within the bin."""
    F = np.cumsum(q, -1); out = []
    for L in QS:
        k = np.argmax(F >= L - 1e-12, -1)
        Fk = np.take_along_axis(F, k[:, None], -1)[:, 0]
        Fp = np.where(k > 0, np.take_along_axis(F, np.maximum(k - 1, 0)[:, None], -1)[:, 0], 0.0)
        out.append((k + (L - Fp) / np.maximum(Fk - Fp, 1e-15)) * DT)
    return np.stack(out, -1)


def fisher(q, ref=None, frac=0.01):
    """Fisher information (1/ns^2) per photon for a time shift of the binned shapes q [N,240]: central differences,
    summed over the bins where the model shape `ref` (default q) exceeds `frac` of its maximum. The onset bins below
    that are dominated by binning and counting noise (f'^2/f of a nearly empty bin) and carry little information."""
    ref = q if ref is None else np.broadcast_to(ref, q.shape)
    mask = ref >= frac * ref.max(-1, keepdims=True)
    qp = np.concatenate([np.zeros_like(q[:, :1]), q, q[:, -1:]], -1)
    dq = (qp[:, 2:] - qp[:, :-2]) / 2
    return np.sum(np.where(mask & (q > 0), dq ** 2 / np.where(q > 0, q, 1), 0), -1) / DT ** 2


def cdf_dists(a, b):
    """Cramer-von Mises and Anderson-Darling distances of shapes a (data) vs b (model), weights = model density."""
    Fa, Fb = np.cumsum(a, -1), np.cumsum(b, -1)
    d2 = (Fa - Fb) ** 2
    cvm = np.sum(d2 * b, -1)
    den = Fb * (1 - Fb)
    ad = np.sum(np.where(den > 1e-9, d2 / np.where(den > 1e-9, den, 1) * b, 0), -1)
    return cvm, ad


def coarse_counts(h):
    return np.add.reduceat(np.atleast_2d(np.asarray(h, float)), J.IDX[:-1], axis=-1)


def hellinger(a, b):
    ca, cb = norm(coarse_counts(a)), norm(coarse_counts(b))
    return np.sqrt(np.clip(1 - np.sum(np.sqrt(ca * cb), -1), 0, None))


def groups(expected, minimum=10.0):
    """merge adjacent coarse bins until each group has >= minimum expected counts; returns group index per bin."""
    g, acc, k = np.zeros(len(expected), int), 0.0, 0
    for i, e in enumerate(expected):
        g[i] = k; acc += e
        if acc >= minimum:
            k += 1; acc = 0.0
    if acc < minimum and k > 0:          # remainder joins the previous group
        g[g == k] = k - 1
    return g


def chi2_one(n_counts, shape):
    """Pearson chi2 of counts [D,41] vs coarse shape [41] (normalized)."""
    N = n_counts.sum(-1, keepdims=True)
    gi = groups(shape * N.mean())
    n = np.stack([n_counts[:, gi == j].sum(-1) for j in range(gi.max() + 1)], -1)
    e = np.array([shape[gi == j].sum() for j in range(gi.max() + 1)]) * N
    return np.sum((n - e) ** 2 / e, -1), gi.max()


def chi2_two(n_counts, m_counts):
    N, M = n_counts.sum(-1, keepdims=True), m_counts.sum(-1, keepdims=True)
    gi = groups((n_counts + m_counts).mean(0))
    n = np.stack([n_counts[:, gi == j].sum(-1) for j in range(gi.max() + 1)], -1)
    m = np.stack([m_counts[:, gi == j].sum(-1) for j in range(gi.max() + 1)], -1)
    return np.sum((np.sqrt(M / N) * n - np.sqrt(N / M) * m) ** 2 / np.maximum(n + m, 1e-9), -1), gi.max()


def stats(data, model, n_model=None):
    """all statistics of data shape(s) [D,240] vs model shape(s) [D,240] (broadcast)."""
    out = dict(q=quant(data) - quant(model), qc=quant(conv(data)) - quant(conv(model)), I=fisher(conv(data), conv(model)),
               hel=hellinger(data, model))
    out["cvm"], out["ad"] = cdf_dists(data, model)
    return out


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
            gcount = np.round(H[f"{gid}|{name}|g4_unpol"] * DT * row["g4_unpol"]["n"])
            ng = int(gcount.sum()); g = norm(gcount)
            meth = {"adjoint": [norm(H[f"{gid}|{name}|adjoint"])[0]]}
            meth["network"] = [norm((mn[sl] * pn0[sl][..., None]).reshape(-1, 240).sum(0))[0] for pn0, _, mn in nets]
            jcount = None
            if "jax" in row:
                jcount = np.round(H[f"{gid}|{name}|jax"] * DT * row["jax"]["n"]); meth["jax"] = [norm(jcount)[0]]
            out = dict(gid=gid, region=name, n_g4=ng, n_jax=None if jcount is None else int(jcount.sum()))
            for mname, shapes in meth.items():
                vals, flo = [], []
                for s in shapes:
                    D = rng.multinomial(ng, s, size=NDRAW).astype(float)
                    if mname == "jax":
                        Mj = rng.multinomial(int(jcount.sum()), s, size=NDRAW).astype(float)
                        c2, ndf = chi2_two(coarse_counts(gcount), coarse_counts(jcount))
                        c2f, _ = chi2_two(coarse_counts(D), coarse_counts(Mj))
                        model_draw = norm(Mj)
                    else:
                        cs = norm(coarse_counts(s))[0]
                        c2, ndf = chi2_one(coarse_counts(gcount), cs)
                        c2f, _ = chi2_one(coarse_counts(D), cs)
                        model_draw = s[None]
                    st = stats(g, s[None] if mname != "jax" else s[None]); sf = stats(norm(D), model_draw)
                    Ibar = sf["I"].mean()
                    v = dict(chi2=float(c2[0]), ndf=int(ndf), p=float(CHI2.sf(c2[0], ndf)), chi2_floor=float(c2f.mean()),
                             dq_ps=(1e3 * st["q"][0]).tolist(), dqc_ps=(1e3 * st["qc"][0]).tolist(),
                             dq_floor_ps=(1e3 * np.sqrt(np.mean(sf["q"] ** 2, 0))).tolist(),
                             dqc_floor_ps=(1e3 * np.sqrt(np.mean(sf["qc"] ** 2, 0))).tolist(),
                             sigma1_ns=float(fisher(conv(s[None]))[0] ** -0.5),
                             dlnsigma=float(0.5 * np.log(st["I"][0] / Ibar)),
                             dlnsigma_floor=float(np.sqrt(np.mean((0.5 * np.log(sf["I"] / Ibar)) ** 2))),
                             hel=float(st["hel"][0]), hel_floor=float(sf["hel"].mean()),
                             cvm=float(st["cvm"][0]), cvm_floor=float(sf["cvm"].mean()),
                             ad=float(st["ad"][0]), ad_floor=float(sf["ad"].mean()))
                    a_c = conv(meth["adjoint"][0][None])
                    v["sigma1_vs_adj"] = float(0.5 * np.log(fisher(a_c)[0] / fisher(conv(s[None]), a_c)[0]))
                    vals.append(v)
                if len(vals) == 1:
                    out[mname] = vals[0]
                else:   # network: average over seeds (lists elementwise)
                    out[mname] = {k: (np.mean([x[k] for x in vals], 0).tolist() if isinstance(vals[0][k], list)
                                      else float(np.mean([x[k] for x in vals]))) for k in vals[0]}
                    out[mname]["seeds"] = vals
            out["sigma1_g4_ns_conv"] = float(fisher(conv(g))[0] ** -0.5)
            rows.append(out)
        print(gid, "regions done", flush=True)
    summ = {}
    for m in ("adjoint", "jax", "network"):
        rr = [r[m] for r in rows if m in r]
        if m == "network":   # chi2 per seed
            c2 = sum(np.mean([s["chi2"] for s in r["seeds"]]) for r in rr)
            npass = float(np.mean([np.mean([s["p"] < 0.05 for s in r["seeds"]]) for r in rr]) * len(rr))
        else:
            c2 = sum(r["chi2"] for r in rr); npass = float(sum(r["p"] < 0.05 for r in rr))
        dq = np.array([r["dq_ps"] for r in rr]); dqc = np.array([r["dqc_ps"] for r in rr])
        fq = np.array([r["dq_floor_ps"] for r in rr]); fqc = np.array([r["dqc_floor_ps"] for r in rr])
        ds = np.array([r["dlnsigma"] for r in rr]); fs = np.array([r["dlnsigma_floor"] for r in rr])
        summ[m] = dict(n=len(rr), chi2_sum=float(c2), ndf_sum=int(sum(r["ndf"] for r in rr)),
                       chi2_floor_sum=float(sum(r["chi2_floor"] for r in rr)), n_p_below_0p05=npass,
                       quantiles=list(QS),
                       dq_mean_ps=dq.mean(0).tolist(), dq_se_ps=(dq.std(0) / np.sqrt(len(rr))).tolist(),
                       dq_rms_ps=np.sqrt((dq ** 2).mean(0)).tolist(), dq_floor_rms_ps=np.sqrt((fq ** 2).mean(0)).tolist(),
                       dqc_mean_ps=dqc.mean(0).tolist(), dqc_se_ps=(dqc.std(0) / np.sqrt(len(rr))).tolist(),
                       dqc_rms_ps=np.sqrt((dqc ** 2).mean(0)).tolist(), dqc_floor_rms_ps=np.sqrt((fqc ** 2).mean(0)).tolist(),
                       dlnsigma_mean=float(ds.mean()), dlnsigma_se=float(ds.std() / np.sqrt(len(rr))),
                       dlnsigma_rms=float(np.sqrt((ds ** 2).mean())), dlnsigma_floor_rms=float(np.sqrt((fs ** 2).mean())),
                       hel=float(np.mean([r["hel"] for r in rr])), hel_floor=float(np.mean([r["hel_floor"] for r in rr])),
                       cvm=float(np.mean([r["cvm"] for r in rr])), cvm_floor=float(np.mean([r["cvm_floor"] for r in rr])),
                       ad=float(np.mean([r["ad"] for r in rr])), ad_floor=float(np.mean([r["ad_floor"] for r in rr])))
    # network vs adjoint directly (both noise-free shapes): sigma_1 ratio
    rat = np.array([r["network"]["sigma1_vs_adj"] for r in rows])
    summ["network_vs_adjoint_sigma1"] = dict(mean_ln_ratio=float(rat.mean()), rms_ln_ratio=float(np.sqrt((rat ** 2).mean())),
                                             adjoint_sigma1_ns_median=float(np.median([r["adjoint"]["sigma1_ns"] for r in rows])))
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
        href = np.asarray(href, float).reshape(-1, 240); h20 = np.asarray(h20, float).reshape(-1, 240)
        ok = (href.sum(1) > 0) & (h20.sum(1) > 0)
        w = np.where(ok, p, 0); w = w / w.sum()
        ref = norm(href); refc = conv(ref)
        qr, qrc, Ir = quant(ref), quant(refc), fisher(refc)
        wq, wqc = quant(norm(href.sum(0))), quant(conv(norm(href.sum(0))))
        Iw = fisher(conv(norm(href.sum(0))))

        def score(M):
            M = np.asarray(M, float).reshape(-1, 240); m = norm(M); mc = conv(m)
            dq = quant(m) - qr; dqc = quant(mc) - qrc; ds = 0.5 * np.log(Ir / np.maximum(fisher(mc, refc), 1e-30))
            wm = norm(M.sum(0))
            return dict(dq_rms_ps=(1e3 * np.sqrt(np.sum(w[:, None] * dq ** 2, 0))).tolist(),
                        dq_mean_ps=(1e3 * np.sum(w[:, None] * dq, 0)).tolist(),
                        dqc_rms_ps=(1e3 * np.sqrt(np.sum(w[:, None] * dqc ** 2, 0))).tolist(),
                        dqc_mean_ps=(1e3 * np.sum(w[:, None] * dqc, 0)).tolist(),
                        dlnsigma_rms=float(np.sqrt(np.sum(w * ds ** 2))), dlnsigma_mean=float(np.sum(w * ds)),
                        whole_dq_ps=(1e3 * (quant(wm) - wq)[0]).tolist(), whole_dqc_ps=(1e3 * (quant(conv(wm)) - wqc)[0]).tolist(),
                        whole_dlnsigma=float(0.5 * np.log(Iw[0] / fisher(conv(wm), conv(norm(href.sum(0))))[0])))
        r = dict(set=name, gid=gid, adj20k=score(h20))
        seeds = [score(M) for _, M in V.network(path, gid, models)]
        r["net"] = {k: (np.mean([s[k] for s in seeds], 0).tolist() if isinstance(seeds[0][k], list)
                        else float(np.mean([s[k] for s in seeds]))) for k in seeds[0]}
        rows.append(r)
    summ = {}
    for m in ("net", "adj20k"):
        summ[m] = {}
        for k in rows[0][m]:
            v = np.array([r[m][k] for r in rows])
            if k.endswith("rms_ps") or k == "dlnsigma_rms":
                summ[m][k] = np.sqrt((v ** 2).mean(0)).tolist()
            elif k.startswith("whole"):
                summ[m][k + "_mean"] = v.mean(0).tolist(); summ[m][k + "_rms"] = np.sqrt((v ** 2).mean(0)).tolist()
            else:
                summ[m][k] = v.mean(0).tolist()
    return summ, rows


if __name__ == "__main__":
    s1, r1 = regions()
    print(json.dumps(s1, indent=1), flush=True)
    s2, r2 = voxels()
    print(json.dumps(s2, indent=1), flush=True)
    json.dump(dict(quantiles=list(QS), regions=dict(summary=s1, rows=r1), voxels=dict(summary=s2, rows=r2)),
              open(C.ST / "dist_more.json", "w"), indent=1)
