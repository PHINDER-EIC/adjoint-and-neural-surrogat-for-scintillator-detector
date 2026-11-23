"""Statistics quoted in Sec. VI.C, recomputed with the current residual of p (common.resid; relative difference by
default): new designs vs test (bootstrap, permutation), offset and scatter, residual correlations, track and block sums.
Writes study/stats_rel.json (or stats_ln.json with P_RESID=ln)."""
import sys as _sys, pathlib as _pl  # release: shared modules in ../../code
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "code"))
import paths  # noqa: E402,F401
import json

import numpy as np

import common as C

rng = np.random.default_rng(5)
ids, gi, M = C.methods()
tr = C.sub_truth(gi)
P = tr["p"]; G = len(ids)
sp = C.sur_preds()
seeds = [t for t in C.NET_ENSEMBLE if t in sp]
test = [sp[t]["final_p0"][gi] for t in seeds]
B = np.load(C.ST / "blind_preds.npz")
btr = dict(p=B["truth_p"], p_se=B["truth_p_se"])
blind = [B[f"net{i}_p0"] for i in range(3)]
k2 = C.ref_noise()["k_p"] ** 2
out = {}


def parts(pred, T):
    tp = T["p"]; r = C.resid(pred, tp)
    return (tp * r ** 2).sum(1), (tp * (T["p_se"] / tp) ** 2).sum(1) * k2, tp.sum(1)


def err(a, n, W):
    return np.sqrt(max((a.sum() - n.sum()) / W.sum(), 0))


PT = [parts(p, tr) for p in test]; PB = [parts(p, btr) for p in blind]
e_t = np.mean([err(*x) for x in PT]); e_b = np.mean([err(*x) for x in PB])
diffs = []
for _ in range(20000):
    it = rng.integers(0, 40, 40); ib = rng.integers(0, 40, 40)
    diffs.append(np.mean([err(a[ib], n[ib], W[ib]) for a, n, W in PB]) - np.mean([err(a[it], n[it], W[it]) for a, n, W in PT]))
perm = []
for _ in range(20000):
    idx = rng.permutation(80); a_, b_ = idx[:40], idx[40:]
    v = []
    for (at, nt, Wt), (ab, nb, Wb) in zip(PT, PB):
        A = np.concatenate([at, ab]); N_ = np.concatenate([nt, nb]); W = np.concatenate([Wt, Wb])
        v.append(err(A[b_], N_[b_], W[b_]) - err(A[a_], N_[a_], W[a_]))
    perm.append(np.mean(v))
out["test"] = float(e_t); out["new"] = float(e_b); out["diff"] = float(e_b - e_t)
out["bootstrap95"] = [float(x) for x in np.percentile(diffs, [2.5, 97.5])]
out["perm_p_one_sided"] = float(np.mean(np.array(perm) >= e_b - e_t))
out["new_seeds"] = [float(err(*x)) for x in PB]
pm = [C.pmetrics(p, tr) for p in test]
out["offset_rms"] = float(np.mean([m["offset_rms"] for m in pm]))
out["shape_rms"] = float(np.mean([m["shape_rms"] for m in pm]))
out["ens_test"] = C.pmetrics(np.mean(test, 0), tr)["rms"]
out["ens_new"] = C.pmetrics(np.mean(blind, 0), btr)["rms"]
pool_tr = {k: np.concatenate([tr[k], btr[k]]) for k in ("p", "p_se")}
out["pooled"] = float(np.mean([C.pmetrics(np.concatenate([a, b]), pool_tr)["rms"] for a, b in zip(test, blind)]))

# residual structure (as in diag_review.py), with the current residual
NET = M["net_628"]["p0"]; ADJ = M["adj_20k"]["p0"]


def shape_resid(pp):
    r = C.resid(pp, P); w = P / P.sum(1, keepdims=True)
    off = (w * r).sum(1, keepdims=True)
    return r - off, w


def lag_corr(r, w, axis, lag):
    R = r.reshape(G, 20, 20, 8); Wt = w.reshape(G, 20, 20, 8); n = R.shape[axis + 1]
    a = np.take(R, range(0, n - lag), axis=axis + 1); b = np.take(R, range(lag, n), axis=axis + 1)
    ww = np.sqrt(np.take(Wt, range(0, n - lag), axis=axis + 1) * np.take(Wt, range(lag, n), axis=axis + 1))
    return float(np.sum(ww * a * b) / np.sqrt(np.sum(ww * a * a) * np.sum(ww * b * b)))


for name, pp in (("network", NET), ("adjoint_2e4", ADJ)):
    r, w = shape_resid(pp)
    out[f"corr_{name}"] = dict(x1=lag_corr(r, w, 0, 1), y1=lag_corr(r, w, 1, 1), z1=lag_corr(r, w, 2, 1),
                               x5=lag_corr(r, w, 0, 5))
rs = [shape_resid(v)[0] for v in test]; w = P / P.sum(1, keepdims=True)
out["seed_patch_corr"] = float(np.mean([np.sum(w * rs[i] * rs[j]) / np.sqrt(np.sum(w * rs[i] ** 2) * np.sum(w * rs[j] ** 2))
                                         for i in range(3) for j in range(i + 1, 3)]))


def track_err(pp, ntr=400):
    rng2 = np.random.default_rng(5)
    errs, errs_shape = [], []
    wv = P / P.sum(1, keepdims=True)
    off = (wv * C.resid(pp, P)).sum(1)
    for g in range(G):
        a = np.c_[rng2.random(ntr) * 20, rng2.random(ntr) * 20, np.full(ntr, 8.0)]
        th = np.arccos(rng2.random(ntr) ** 0.25); ph = rng2.random(ntr) * 2 * np.pi
        d = np.c_[np.sin(th) * np.cos(ph) * 2.5, np.sin(th) * np.sin(ph) * 2.5, -np.cos(th)]
        f = (np.arange(64) + 0.5) / 64 * 8
        pts = a[:, None, :] + f[None, :, None] * d[:, None, :] / np.maximum(-d[:, None, 2:3], 1e-3)
        ok = (pts[..., 0] >= 0) & (pts[..., 0] < 20) & (pts[..., 1] >= 0) & (pts[..., 1] < 20)
        v = (np.clip(pts[..., 0].astype(int), 0, 19) * 20 + np.clip(pts[..., 1].astype(int), 0, 19)) * 8 + \
            np.clip(pts[..., 2].astype(int), 0, 7)
        s_ = np.where(ok, pp[g][v], 0).sum(1); st = np.where(ok, P[g][v], 0).sum(1)
        m = st > 0
        e = C.resid(s_[m], st[m])
        errs += list(e); errs_shape += list(e - off[g])
    return float(np.sqrt(np.mean(np.square(errs)))), float(np.sqrt(np.mean(np.square(errs_shape))))


def block_err(pp, b=10):
    Pp = pp.reshape(G, 20, 20, 8).sum(-1); Pt = P.reshape(G, 20, 20, 8).sum(-1); k = 20 // b
    Bp = Pp.reshape(G, k, b, k, b).sum((2, 4)); Bt = Pt.reshape(G, k, b, k, b).sum((2, 4))
    r = C.resid(Bp, Bt); wv = Bt / Bt.sum((1, 2), keepdims=True)
    return float(np.sqrt(np.sum(wv * r ** 2) / np.sum(wv)))


out["tracks"] = dict(network=track_err(NET), adjoint_2e4=track_err(ADJ))
out["quarter_blocks"] = dict(network=block_err(NET), adjoint_2e4=block_err(ADJ))
json.dump(out, open(C.ST / ("stats_rel.json" if C.P_RESID == "rel" else "stats_ln.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
