import paths  # release: data and repository locations
"""Shared loaders and metrics for the speed/accuracy study.

Truth = 200,000-root adjoint maps of the 40 test geometries (numba tracer, max_steps 5000).
Voxel order everywhere = C-order ravel of the (20, 20, 8) grid; geometries in sorted-id order.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

T = paths.DATA
ST = T / "derived"
TRUTH_H5 = T / "reference_test40_2e5roots.h5"
TC = (np.arange(240) + 0.5) * 0.25
# Residual of p: 'ln' = ln(p_hat/p) (paper), 'rel' = (p_hat - p)/p (same form as the time metric).
P_RESID = os.environ.get("P_RESID", "rel")   # paper metric since the switch; "ln" for the old one


def resid(a, b):
    a = np.maximum(a, 1e-9); b = np.maximum(b, 1e-9)
    return np.log(a / b) if P_RESID == "ln" else (a - b) / b


def out(name):
    """Path of a result file; with P_RESID=ln the file gets an _ln suffix (the relative difference is the paper's metric)."""
    p = ST / name
    return p if P_RESID == "rel" else p.with_name(p.stem + "_ln" + p.suffix)


def tsd_from_h(h):
    m0 = h.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (h * TC).sum(-1) / m0
        return np.sqrt(np.maximum((h * TC ** 2).sum(-1) / m0 - mu ** 2, 0.0)), mu


@lru_cache(None)
def truth():
    d = {k: [] for k in ("p", "p_se", "mt", "tsd", "pvol", "pvol_se", "tvol", "cell", "width", "radius", "h_files")}
    with h5py.File(TRUTH_H5) as f:
        ids = sorted(f["geometries"].keys())
        for g in ids:
            G = f["geometries"][g]
            fl = G["field"]
            d["p"].append(fl["p_per_emitted"][:].ravel()); d["p_se"].append(fl["p_se"][:].ravel())
            d["mt"].append(fl["mean_transport_time_ns"][:].ravel())
            sd, _ = tsd_from_h(fl["h_per_ns"][:])
            d["tsd"].append(sd.ravel())
            d["pvol"].append(float(G.attrs["volume_mean_p"])); d["pvol_se"].append(float(G.attrs["volume_mean_p_se"]))
            d["tvol"].append(float(G.attrs["volume_mean_transport_time_ns"]))
            d["cell"].append(G["cell_cm"][:]); dj = json.loads(G.attrs["design_json"])
            d["width"].append(dj["width_cm"]); d["radius"].append(dj["radius_cm"])
    out = {k: np.array(v) for k, v in d.items() if k != "h_files"}
    out["ids"] = ids
    return out


def h_truth(gid):
    with h5py.File(TRUTH_H5) as f:
        return f["geometries"][gid]["field/h_per_ns"][:]


# ---------------------------------------------------------------- metrics
def pmetrics(pp, tr=None, noise_sub=True):
    """Voxel p error vs truth. pp: [G, V]. Returns dict of fractions (log residual ~ relative)."""
    tr = tr or truth()
    tp = tr["p"]
    r = resid(pp, tp)
    w = tp / tp.sum()
    rel_t = tr["p_se"] / np.maximum(tp, 1e-12)
    # the tracer's self-reported s.e. is too large for quasi-random roots; scale it to the measured noise
    noise2 = float(np.sum(w * rel_t ** 2)) * ref_noise()["k_p"] ** 2 if noise_sub else 0.0
    raw = float(np.sqrt(np.sum(w * r ** 2)))
    mean = float(np.sum(w * r))
    sd = float(np.sqrt(np.sum(w * (r - mean) ** 2)))
    # per-geometry decomposition: level offset (p-weighted mean per geometry) and shape residual
    wg = tp / tp.sum(1, keepdims=True)
    off = (wg * r).sum(1)
    shape = float(np.sqrt(np.sum(w * (r - off[:, None]) ** 2)))
    gv = pp.mean(1) / tp.mean(1) - 1
    return dict(rms=float(np.sqrt(max(raw ** 2 - noise2, 0))), rms_raw=raw, bias=mean, sd=sd,
                med_abs=float(np.median(np.abs(r))), p90_abs=float(np.quantile(np.abs(r), 0.9)),
                offset_rms=float(np.sqrt(np.mean(off ** 2))), shape_rms=float(np.sqrt(max(shape ** 2 - noise2, 0))),
                geo_rel_rms=float(np.sqrt(np.mean(gv ** 2))), geo_rel_med=float(np.median(np.abs(gv))),
                geo_rel_max=float(np.max(np.abs(gv))), geo_bias=float(np.mean(gv)), truth_noise=float(np.sqrt(noise2)))


def tmetrics(pt, key="mt", tr=None, truth_noise=0.0):
    """Relative error of a per-voxel time quantity (mean time 'mt' or time s.d. 'tsd'), p-weighted."""
    tr = tr or truth()
    tt = tr[key]
    ok = np.isfinite(pt) & np.isfinite(tt) & (tt > 0)
    r = np.where(ok, (pt - tt) / np.where(ok, tt, 1), 0.0)
    w = np.where(ok, tr["p"], 0.0); w = w / w.sum()
    raw = float(np.sqrt(np.sum(w * r ** 2)))
    mean = float(np.sum(w * r))
    return dict(rms=float(np.sqrt(max(raw ** 2 - truth_noise ** 2, 0))), rms_raw=raw, bias=mean,
                sd=float(np.sqrt(np.sum(w * (r - mean) ** 2))), med_abs=float(np.median(np.abs(r[ok]))))


def geo_time(pp, pt, tr=None):
    """Per-geometry p-weighted mean transport time, relative error vs truth (uniform emission)."""
    tr = tr or truth()
    a = np.nansum(np.nan_to_num(pt) * pp, 1) / pp.sum(1)
    b = np.nansum(np.nan_to_num(tr["mt"]) * tr["p"], 1) / tr["p"].sum(1)
    return a / b - 1


# ---------------------------------------------------------------- loaders
def adj_runs():
    """Adjoint convergence runs: dict gid -> npz contents, in truth order (missing geometries skipped)."""
    out = {}
    for g in truth()["ids"]:
        f = ST / "adj" / f"{g}.npz"
        if f.exists():
            out[g] = dict(np.load(f))
    return out


def sur_preds():
    out = {}
    for f in sorted((ST / "sur").glob("*.pred.npz")):
        d = dict(np.load(f))
        d["meta"] = json.loads(str(d["meta"]))
        out[f.name[:-9]] = d
    return out


def gp_preds():
    return {int(f.stem.split("_n")[1]): dict(np.load(f)) for f in sorted((ST / "gp").glob("gp_n*.npz"))}


def jax_runs():
    out = {}
    for f in sorted((ST / "jax").glob("*.json")):
        d = json.load(open(f))
        if d.get("skip_marker"):
            continue
        d["arrays"] = dict(np.load(f.with_suffix(".npz")))
        out[d["geometry"]] = d
    return out


def g4_runs():
    """dict (gid, mode, region) -> merged dict(vox, ch, t, emitted_vox, wall_s, n)."""
    out = {}
    for f in sorted((ST / "g4/runs").glob("*.npz")):
        gid, mode, reg, _ = f.stem.split("__")
        d = np.load(f)
        k = (gid, mode, reg)
        o = out.setdefault(k, dict(vox=[], ch=[], t=[], emitted_vox=0, wall_s=0.0, chunks=0))
        o["vox"].append(d["vox"]); o["ch"].append(d["ch"]); o["t"].append(d["t"])
        o["emitted_vox"] = o["emitted_vox"] + d["emitted_vox"]; o["wall_s"] += float(d["wall_s"]); o["chunks"] += 1
    for o in out.values():
        for k in ("vox", "ch", "t"):
            o[k] = np.concatenate(o[k])
        o["n"] = int(o["emitted_vox"].sum())
    return out


def region_slices(lo, hi):
    return tuple(slice(a, b) for a, b in zip(lo, hi))


# ---------------------------------------------------------------- method predictions at standard settings
NET_ENSEMBLE = ["n628_e240_s20261003", "n628_e240_s1", "n628_e240_s2"]


def methods(adj_N=(1000, 5000, 20000)):
    """Predictions of every method on the common set of test geometries.
    Returns (ids, idx_in_truth, dict name -> dict(p0, mt, tsd) [G, 3200])."""
    tr = truth()
    A = adj_runs()
    ids = [g for g in tr["ids"] if g in A]
    gi = np.array([tr["ids"].index(g) for g in ids])
    out = {}
    for N in adj_N:
        sel = [np.where((A[g]["N"] == N) & (A[g]["steps"] == 5000) & (A[g]["rep"] == 0))[0][0] for g in ids]
        out[f"adj_{N//1000}k"] = {k: np.array([A[g][kk][i].ravel() for g, i in zip(ids, sel)])
                                  for k, kk in (("p0", "p"), ("mt", "mt"), ("tsd", "tsd"), ("p_se", "p_se"))}
    gp = gp_preds()
    if 628 in gp:
        out["gp_628"] = {k: gp[628][k][gi] for k in ("p0", "mt", "tsd")}
    sp = sur_preds()
    have = [t for t in NET_ENSEMBLE if t in sp]
    if have:
        out["net_628"] = {k: sp[have[0]][f"final_{k}"][gi] for k in ("p0", "mt", "tsd")}
        if len(have) > 1:
            out["net_628_ens"] = {k: np.mean([sp[t][f"final_{k}"][gi] for t in have], 0) for k in ("p0", "mt", "tsd")}
            out["net_628_ens"]["members"] = len(have)
    return ids, gi, out


def sub_truth(gi):
    tr = truth()
    return {k: (v[gi] if isinstance(v, np.ndarray) and v.shape[:1] == (40,) else v) for k, v in tr.items()}


@lru_cache(None)
def ref_noise():
    """Measured noise of the 2e5-root reference maps (written by ref_noise.py, study/ref_noise.json).

    From pairs of independent replicate maps of the convergence runs (no reference involved): the noise at N roots,
    sigma(N) = a(N)/sqrt(N), with a(N) falling slowly with N because the roots are quasi-random; a(N) is fitted as a
    power law for 1e3 <= N <= 2e4 and extrapolated to 2e5. k_p = measured / self-reported voxel noise of p at 2e5."""
    f = ST / "ref_noise.json"
    if not f.exists():
        return dict(p=None, mt=0.0, tsd=0.0, k_p=1.0)
    return json.load(open(f))


@lru_cache(None)
def truth_time_noise():
    """Noise of the 200k reference for mean time and time s.d. (relative, p-weighted), measured from replicate pairs;
    see ref_noise(). Earlier versions fitted a constant a to the reference-based errors, which gave 0.60% / 0.38%."""
    rn = ref_noise()
    return dict(mt=float(rn["mt"]), tsd=float(rn["tsd"]))
