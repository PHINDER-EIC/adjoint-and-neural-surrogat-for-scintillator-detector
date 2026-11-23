#!/usr/bin/env python3
"""
PHINDER Stage-6B: differentiable geometry-conditioned optical surrogate
trained from the dense weighted-adjoint geometry atlas.

Scientific object
-----------------
For one emitted photon from source r in geometry G, learn

    (G,r) -> {p0, f0(t), p1, f1(t)}

with h_c(t)=p_c f_c(t).  Guide 1 is not learned independently: exact mirror
symmetry is imposed architecturally by evaluating the same canonical channel
network at M(r)=(Dx-x, y, Dz-z).

Geometry representation
-----------------------
The current Stage-6B atlas has a fixed five-segment guide (six ordered
vertices: start, four internal turns, terminal).  An ordered bidirectional GRU
encodes that skeleton, using vertex coordinates and source-relative
coordinates.  A small SiLU MLP combines the path embedding, six global geometry
variables (Dx,Dy,Dz,w,R,R/w), and the source position.

Timing representation
---------------------
Conditional transport-time density is a positive mixture of fixed log-normal
basis densities.  The network predicts only mixture weights.  This keeps the
public h(t) continuous, positive, normalized, compact, and differentiable.

The script is deliberately self-contained and does not alter the old v3
forward-photon operator.  It consumes the Stage-6B adjoint HDF5 format directly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "PHINDER Stage-6B differentiable adjoint-field surrogate"
FORMAT_VERSION = 1
EPS = 1e-12


def _decode(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _device_from_arg(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quantiles(x, qs=(0.5, 0.9)):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return [float("nan") for _ in qs]
    return [float(np.quantile(a, q)) for q in qs]


def _geometry_ids_by_split(f: h5py.File) -> Dict[str, List[str]]:
    out = {"train": [], "val": [], "test": []}
    for gid, g in f["geometries"].items():
        s = _decode(g.attrs["split"])
        out.setdefault(s, []).append(gid)
    for s in out:
        out[s].sort()
    return out


def _check_atlas(f: h5py.File):
    if "geometries" not in f or "time_edges_ns" not in f:
        raise ValueError("not a Stage-6B adjoint geometry atlas: missing geometries/time_edges_ns")
    required = ["u_edges", "v_edges", "s_edges", "time_edges_ns"]
    for k in required:
        if k not in f:
            raise ValueError(f"atlas missing {k}")
    if not len(f["geometries"]):
        raise ValueError("atlas contains no completed geometries")


def _common_uvs(f: h5py.File) -> np.ndarray:
    ue, ve, se = f["u_edges"][:], f["v_edges"][:], f["s_edges"][:]
    uc = 0.5 * (ue[:-1] + ue[1:])
    vc = 0.5 * (ve[:-1] + ve[1:])
    sc = 0.5 * (se[:-1] + se[1:])
    U, V, S = np.meshgrid(uc, vc, sc, indexing="ij")
    return np.column_stack([U.ravel(), V.ravel(), S.ravel()]).astype(np.float32)


def _geom_arrays(g: h5py.Group):
    cell = g["cell_cm"][:].astype(np.float32)
    w = float(g.attrs["width_cm"])
    R = float(g.attrs["radius_cm"])
    geom6 = np.array([cell[0], cell[1], cell[2], w, R, R / w], dtype=np.float32)
    nodes = np.vstack([g["start_xy_cm"][:], g["turns_xy_cm"][:], g["end_xy_cm"][:]]).astype(np.float32)
    nodes_norm = nodes / cell[:2][None, :]
    return geom6, nodes_norm


def _mirror_field(a: np.ndarray) -> np.ndarray:
    # M(x,y,z) = (Dx-x, y, Dz-z).  Voxel centres mirror by reversing x and z.
    return a[::-1, :, ::-1, ...]


@dataclass
class LoadedData:
    ids: List[str]
    geom6: np.ndarray            # [G,6]
    nodes_norm: np.ndarray       # [G,6,2]
    uvs: np.ndarray              # [V,3]
    p0: np.ndarray               # [G,V]
    p1: np.ndarray               # [G,V]
    relse0: np.ndarray           # [G,V]
    mean_t0: np.ndarray          # [G,V]
    hmass0: np.ndarray           # [G,V,T], conditional numerator mass = h*dt
    roots: np.ndarray            # [G]
    time_edges: np.ndarray       # [T+1]

    @property
    def n_geom(self): return len(self.ids)
    @property
    def n_vox(self): return self.uvs.shape[0]
    @property
    def nt(self): return len(self.time_edges) - 1


def load_split_data(atlas: Path, ids: Sequence[str]) -> LoadedData:
    ids = list(ids)
    if not ids:
        raise ValueError("requested geometry list is empty")
    with h5py.File(atlas, "r") as f:
        _check_atlas(f)
        uvs = _common_uvs(f)
        time_edges = f["time_edges_ns"][:].astype(np.float32)
        dt = float(np.median(np.diff(time_edges)))
        geom6, nodes, p0s, p1s, rels, mts, hm, roots = [], [], [], [], [], [], [], []
        for gid in ids:
            if gid not in f["geometries"]:
                raise KeyError(f"geometry {gid} is not present in atlas")
            g = f["geometries"][gid]
            gg, nn = _geom_arrays(g)
            p = g["field/p_per_emitted"][:].astype(np.float32)
            rel = g["field/relative_se_p"][:].astype(np.float32)
            mt = g["field/mean_transport_time_ns"][:].astype(np.float32)
            h = g["field/h_per_ns"][:].astype(np.float32)
            if p.size != len(uvs):
                raise ValueError(f"{gid}: field grid {p.shape} inconsistent with common normalized grid")
            geom6.append(gg); nodes.append(nn)
            p0s.append(p.ravel())
            p1s.append(_mirror_field(p).ravel())
            rels.append(rel.ravel())
            mts.append(mt.ravel())
            hm.append((h * dt).reshape(-1, h.shape[-1]))
            roots.append(int(g.attrs["root_launches_successful"]))
    return LoadedData(
        ids=ids,
        geom6=np.stack(geom6), nodes_norm=np.stack(nodes), uvs=uvs,
        p0=np.stack(p0s), p1=np.stack(p1s), relse0=np.stack(rels),
        mean_t0=np.stack(mts), hmass0=np.stack(hm), roots=np.asarray(roots),
        time_edges=time_edges,
    )


def fit_geom_normalization(data: LoadedData):
    mean = data.geom6.mean(axis=0).astype(np.float32)
    std = data.geom6.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def make_lognormal_basis(time_edges: np.ndarray, mixtures: int, log_sigma: float = 0.34):
    edges = np.asarray(time_edges, float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dt = np.diff(edges)
    positive = centres[centres > 0]
    tlo = max(float(positive.min()), 0.08)
    thi = max(float(centres.max()), tlo * 2)
    medians = np.geomspace(max(0.12, tlo), thi * 0.96, mixtures)
    logt = np.log(np.maximum(centres, 1e-6))[None, :]
    mu = np.log(medians)[:, None]
    sig = float(log_sigma)
    dens = np.exp(-0.5 * ((logt - mu) / sig) ** 2) / (np.maximum(centres, 1e-6)[None, :] * sig * math.sqrt(2 * math.pi))
    mass = dens * dt[None, :]
    mass /= np.maximum(mass.sum(axis=1, keepdims=True), EPS)
    dens = mass / dt[None, :]
    means = (mass * centres[None, :]).sum(axis=1)
    return medians.astype(np.float32), mass.astype(np.float32), dens.astype(np.float32), means.astype(np.float32)


class Stage6BSurrogate(nn.Module):
    def __init__(self, geom_mean, geom_std, basis_mass, basis_density, basis_mean,
                 hidden=40, trunk=128):
        super().__init__()
        gm = torch.as_tensor(geom_mean, dtype=torch.float32)
        gs = torch.as_tensor(geom_std, dtype=torch.float32)
        self.register_buffer("geom_mean", gm)
        self.register_buffer("geom_std", gs)
        self.register_buffer("basis_mass", torch.as_tensor(basis_mass, dtype=torch.float32))
        self.register_buffer("basis_density", torch.as_tensor(basis_density, dtype=torch.float32))
        self.register_buffer("basis_mean", torch.as_tensor(basis_mean, dtype=torch.float32))
        self.mixtures = int(basis_mass.shape[0])
        # [node_x, node_y, node_x-u, node_y-v, ordered_position]
        self.path_gru = nn.GRU(5, hidden, batch_first=True, bidirectional=True)
        in_dim = 2 * hidden + 6 + 3
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk), nn.SiLU(),
            nn.Linear(trunk, trunk), nn.SiLU(),
            nn.Linear(trunk, trunk // 2), nn.SiLU(),
        )
        self.score = nn.Linear(trunk // 2, 1)
        self.mix = nn.Linear(trunk // 2, self.mixtures)

    def canonical(self, geom6, nodes_norm, uvs):
        B, L, _ = nodes_norm.shape
        uv = uvs[:, :2].unsqueeze(1).expand(-1, L, -1)
        rel = nodes_norm - uv
        order = torch.linspace(0.0, 1.0, L, device=nodes_norm.device, dtype=nodes_norm.dtype).view(1, L, 1).expand(B, -1, -1)
        seq = torch.cat([nodes_norm, rel, order], dim=-1)
        out, _ = self.path_gru(seq)
        path = out.mean(dim=1)
        gn = (geom6 - self.geom_mean) / self.geom_std
        z = torch.cat([path, gn, uvs], dim=-1)
        z = self.trunk(z)
        return self.score(z).squeeze(-1), self.mix(z)

    def forward(self, geom6, nodes_norm, uvs):
        s0, mix0 = self.canonical(geom6, nodes_norm, uvs)
        mir = torch.stack([1.0 - uvs[:, 0], uvs[:, 1], 1.0 - uvs[:, 2]], dim=-1)
        s1, mix1 = self.canonical(geom6, nodes_norm, mir)
        fate = torch.softmax(torch.stack([torch.zeros_like(s0), s0, s1], dim=-1), dim=-1)
        return {
            "p_lost": fate[:, 0], "p0": fate[:, 1], "p1": fate[:, 2],
            "mix0_logits": mix0, "mix1_logits": mix1,
        }

    def timing_mass(self, mix_logits):
        a = torch.softmax(mix_logits, dim=-1)
        return a @ self.basis_mass

    def timing_density(self, mix_logits):
        a = torch.softmax(mix_logits, dim=-1)
        return a @ self.basis_density

    def mean_time(self, mix_logits):
        a = torch.softmax(mix_logits, dim=-1)
        return a @ self.basis_mean


def _ids_for_first_fit(atlas: Path, internal_val_geometries: int, seed: int):
    with h5py.File(atlas, "r") as f:
        _check_atlas(f)
        by = _geometry_ids_by_split(f)
    train = list(by.get("train", []))
    if not train:
        raise ValueError("atlas contains no training geometries")
    official_val = list(by.get("val", []))
    if official_val:
        return train, official_val, "official-val"
    n = min(int(internal_val_geometries), max(1, len(train) // 5))
    if len(train) <= n:
        raise ValueError("not enough training geometries for an internal whole-geometry holdout")
    rng = np.random.default_rng(seed)
    perm = np.arange(len(train)); rng.shuffle(perm)
    val_idx = set(int(x) for x in perm[:n])
    fit = [g for i, g in enumerate(train) if i not in val_idx]
    val = [g for i, g in enumerate(train) if i in val_idx]
    return fit, val, "internal-train-holdout"


def _batch_from_indices(data: LoadedData, flat_idx: np.ndarray, device):
    V = data.n_vox
    gi = (flat_idx // V).astype(np.int64)
    vi = (flat_idx % V).astype(np.int64)
    geom = torch.from_numpy(data.geom6[gi]).to(device)
    nodes = torch.from_numpy(data.nodes_norm[gi]).to(device)
    uvs = torch.from_numpy(data.uvs[vi]).to(device)
    p0 = torch.from_numpy(data.p0[gi, vi]).to(device)
    p1 = torch.from_numpy(data.p1[gi, vi]).to(device)
    rel = torch.from_numpy(data.relse0[gi, vi]).to(device)
    mt = torch.from_numpy(data.mean_t0[gi, vi]).to(device)
    hmass = torch.from_numpy(data.hmass0[gi, vi]).to(device)
    return geom, nodes, uvs, p0, p1, rel, mt, hmass


def _loss_components(model, batch, lambda_time=0.55, lambda_mean=0.15, relse_scale=0.35):
    geom, nodes, uvs, p0, p1, rel, mt, hmass = batch
    out = model(geom, nodes, uvs)
    pred0 = torch.clamp(out["p0"], min=1e-7)
    pred1 = torch.clamp(out["p1"], min=1e-7)
    t0 = torch.clamp(p0, min=1e-7)
    t1 = torch.clamp(p1, min=1e-7)

    good_rel = torch.where(torch.isfinite(rel), rel, torch.full_like(rel, 5.0))
    reliability = 1.0 / (1.0 + (good_rel / relse_scale) ** 2)
    reliability = torch.clamp(reliability, 0.05, 1.0)
    rate_each = F.smooth_l1_loss(torch.log(pred0), torch.log(t0), reduction="none") + \
                F.smooth_l1_loss(torch.log(pred1), torch.log(t1), reduction="none")
    rate = (rate_each * reliability).sum() / reliability.sum().clamp_min(1.0)

    mass_sum = hmass.sum(dim=-1)
    valid_t = torch.isfinite(mt) & (mass_sum > 1e-10) & (p0 > 1e-7)
    target_mass = hmass / mass_sum.unsqueeze(-1).clamp_min(1e-12)
    pred_mass = torch.clamp(model.timing_mass(out["mix0_logits"]), min=1e-9)
    ce = -(target_mass * torch.log(pred_mass)).sum(dim=-1)
    # Remove target entropy so timing loss is approximately KL(target||pred).
    ent = -(target_mass * torch.log(target_mass.clamp_min(1e-9))).sum(dim=-1)
    kl = torch.clamp(ce - ent, min=0.0)
    # Timing from very noisy/low-efficiency voxels should not dominate.
    tw = reliability * torch.clamp(p0 / 0.01, 0.05, 1.0) * valid_t.float()
    timing = (kl * tw).sum() / tw.sum().clamp_min(1.0)

    pred_mt = model.mean_time(out["mix0_logits"])
    meanerr = F.smooth_l1_loss(pred_mt / 5.0, torch.nan_to_num(mt, nan=5.0) / 5.0, reduction="none")
    meanloss = (meanerr * tw).sum() / tw.sum().clamp_min(1.0)
    total = rate + lambda_time * timing + lambda_mean * meanloss
    return total, rate.detach(), timing.detach(), meanloss.detach()


@torch.no_grad()
def evaluate_data(model, data: LoadedData, device, batch_size=1024, max_samples=0, seed=1234):
    model.eval()
    N = data.n_geom * data.n_vox
    idx = np.arange(N, dtype=np.int64)
    if max_samples and N > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    rel0, reltot, mt_abs, cdfdiff, logerr = [], [], [], [], []
    geom_pred_sum = np.zeros(data.n_geom, float)
    geom_truth_sum = np.zeros(data.n_geom, float)
    geom_count = np.zeros(data.n_geom, int)
    t_centres = 0.5 * (data.time_edges[:-1] + data.time_edges[1:])
    for a in range(0, len(idx), batch_size):
        ii = idx[a:a+batch_size]
        batch = _batch_from_indices(data, ii, device)
        geom, nodes, uvs, p0, p1, rel, mt, hmass = batch
        out = model(geom, nodes, uvs)
        pp0 = out["p0"].cpu().numpy(); pp1 = out["p1"].cpu().numpy()
        tp0 = p0.cpu().numpy(); tp1 = p1.cpu().numpy()
        rel0.extend((np.abs(pp0-tp0) / np.maximum(tp0,1e-5)).tolist())
        reltot.extend((np.abs((pp0+pp1)-(tp0+tp1)) / np.maximum(tp0+tp1,2e-5)).tolist())
        logerr.extend(np.abs(np.log(np.maximum(pp0,1e-7))-np.log(np.maximum(tp0,1e-7))).tolist())
        pmt = model.mean_time(out["mix0_logits"]).cpu().numpy(); tmt = mt.cpu().numpy()
        mask = np.isfinite(tmt)
        mt_abs.extend(np.abs(pmt[mask]-tmt[mask]).tolist())
        pmass = model.timing_mass(out["mix0_logits"]).cpu().numpy()
        tmass = hmass.cpu().numpy(); ss = tmass.sum(axis=1, keepdims=True)
        ok = ss[:,0] > 1e-10
        if np.any(ok):
            tm = tmass[ok] / ss[ok]
            cd = np.max(np.abs(np.cumsum(pmass[ok],axis=1)-np.cumsum(tm,axis=1)),axis=1)
            cdfdiff.extend(cd.tolist())
        gi = (ii // data.n_vox).astype(int)
        for j, gix in enumerate(gi):
            geom_pred_sum[gix] += pp0[j]
            geom_truth_sum[gix] += tp0[j]
            geom_count[gix] += 1
    qrel = _quantiles(rel0); qtot = _quantiles(reltot); qmt = _quantiles(mt_abs); qcdf = _quantiles(cdfdiff); qlog = _quantiles(logerr)
    vp = geom_pred_sum / np.maximum(geom_count,1)
    vt = geom_truth_sum / np.maximum(geom_count,1)
    geom_rel = np.abs(vp-vt)/np.maximum(vt,1e-8)
    return {
        "n_samples": int(len(idx)),
        "p0_abs_relative_median": qrel[0], "p0_abs_relative_p90": qrel[1],
        "ptotal_abs_relative_median": qtot[0], "ptotal_abs_relative_p90": qtot[1],
        "abs_log_p0_error_median": qlog[0], "abs_log_p0_error_p90": qlog[1],
        "mean_time_abs_error_ns_median": qmt[0], "mean_time_abs_error_ns_p90": qmt[1],
        "timing_cdf_maxdiff_median": qcdf[0], "timing_cdf_maxdiff_p90": qcdf[1],
        "geometry_volume_mean_p_relerr_median": float(np.median(geom_rel)),
        "geometry_volume_mean_p_relerr_max": float(np.max(geom_rel)),
        "geometry_volume_truth": vt.tolist(), "geometry_volume_pred": vp.tolist(),
        "geometry_ids": list(data.ids),
    }


def _print_metrics(title, m):
    print(title)
    print(f"  samples                         = {m['n_samples']}")
    print(f"  |p0-p0truth|/p0 median / p90   = {100*m['p0_abs_relative_median']:.2f}% / {100*m['p0_abs_relative_p90']:.2f}%")
    print(f"  |ptot-ptottruth|/ptot med/p90  = {100*m['ptotal_abs_relative_median']:.2f}% / {100*m['ptotal_abs_relative_p90']:.2f}%")
    print(f"  |Delta log p0| median / p90    = {m['abs_log_p0_error_median']:.4f} / {m['abs_log_p0_error_p90']:.4f}")
    print(f"  |mean t error| median / p90    = {m['mean_time_abs_error_ns_median']:.3f} / {m['mean_time_abs_error_ns_p90']:.3f} ns")
    print(f"  timing CDF maxdiff med / p90   = {m['timing_cdf_maxdiff_median']:.3f} / {m['timing_cdf_maxdiff_p90']:.3f}")
    print(f"  geometry <p0> relerr med / max = {100*m['geometry_volume_mean_p_relerr_median']:.2f}% / {100*m['geometry_volume_mean_p_relerr_max']:.2f}%")


def save_checkpoint(path: Path, model, config, geom_mean, geom_std, basis_medians, train_ids, val_ids, history, best_epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(), "config": config,
        "geom_mean": np.asarray(geom_mean), "geom_std": np.asarray(geom_std),
        "basis_medians_ns": np.asarray(basis_medians),
        "train_ids": list(train_ids), "val_ids": list(val_ids),
        "history": history, "best_epoch": int(best_epoch),
    }, path)


def _latest_path(output: Path) -> Path:
    return output.with_name(output.stem + ".latest.pt")


def _save_latest(path: Path, model, opt, sched, config, geom_mean, geom_std, basis_medians, train_ids, val_ids, history, epoch, best, best_epoch, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT, "format_version": FORMAT_VERSION, "kind": "training-latest",
        "model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
        "config": config, "geom_mean": np.asarray(geom_mean), "geom_std": np.asarray(geom_std),
        "basis_medians_ns": np.asarray(basis_medians), "train_ids": list(train_ids), "val_ids": list(val_ids),
        "history": history, "epoch": int(epoch), "best": float(best), "best_epoch": int(best_epoch),
        "numpy_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
    }, path)


def load_model(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError("not a compatible Stage-6B surrogate checkpoint")
    cfg = ck["config"]
    edges = np.asarray(cfg["time_edges_ns"], np.float32)
    med, mass, dens, means = make_lognormal_basis(edges, int(cfg["mixtures"]), float(cfg["log_sigma"]))
    model = Stage6BSurrogate(ck["geom_mean"], ck["geom_std"], mass, dens, means,
                             hidden=int(cfg["hidden"]), trunk=int(cfg["trunk"])).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def cmd_inspect(args):
    with h5py.File(args.atlas, "r") as f:
        _check_atlas(f)
        by = _geometry_ids_by_split(f)
        print(f"atlas: {args.atlas}")
        print("completed geometries:", {k: len(v) for k,v in by.items()})
        roots=[]; p=[]; pse=[]; rel=[]; t=[]; unresolved=[]; accounting=[]
        bad=0
        for gid,g in f["geometries"].items():
            roots.append(int(g.attrs["root_launches_successful"]))
            p.append(float(g.attrs["volume_mean_p"]))
            pse.append(float(g.attrs["volume_mean_p_se"]))
            t.append(float(g.attrs["volume_mean_transport_time_ns"]))
            unresolved.append(float(g.attrs["unresolved_weight_per_root"]))
            accounting.append(float(g.attrs["weight_accounting_per_root"]))
            rr=g["field/relative_se_p"][:]
            rel.extend(rr[np.isfinite(rr)].ravel().tolist())
            if not np.isfinite(g["field/p_per_emitted"][:]).all(): bad += 1
        print(f"roots/geometry min/median/max      = {min(roots)} / {np.median(roots):.0f} / {max(roots)}")
        print(f"volume mean p min/median/max       = {min(p):.5f} / {np.median(p):.5f} / {max(p):.5f}")
        print(f"volume mean t min/median/max       = {min(t):.3f} / {np.median(t):.3f} / {max(t):.3f} ns")
        print(f"voxel relative SE median / p90     = {100*np.median(rel):.1f}% / {100*np.quantile(rel,.9):.1f}%")
        print(f"unresolved weight/root med / max   = {100*np.median(unresolved):.3f}% / {100*np.max(unresolved):.3f}%")
        print(f"weight accounting min/max          = {min(accounting):.6f} / {max(accounting):.6f}")
        print(f"geometries with nonfinite p field  = {bad}")
        print("grid                            =", tuple(int(x) for x in f.attrs.get("grid_shape", [len(f['u_edges'])-1,len(f['v_edges'])-1,len(f['s_edges'])-1])))
        print("time bins / tmax               =", len(f["time_edges_ns"])-1, float(f["time_edges_ns"][-1]), "ns")


def cmd_self_test(args):
    _set_seed(123)
    edges=np.linspace(0,60,241,dtype=np.float32)
    med,mass,dens,means=make_lognormal_basis(edges,12,0.34)
    assert np.max(np.abs(mass.sum(axis=1)-1)) < 1e-5
    gm=np.array([5,5,1,.6,1.2,2],np.float32); gs=np.ones(6,np.float32)
    model=Stage6BSurrogate(gm,gs,mass,dens,means,hidden=8,trunk=32)
    B=5
    geom=torch.tensor(np.tile(gm,(B,1)),requires_grad=True)
    nodes=np.array([[.2,.1],[.7,.25],[.25,.4],[.75,.55],[.3,.7],[.5,1.05]],np.float32)
    nodes=torch.tensor(np.tile(nodes[None,:,:],(B,1,1)),requires_grad=True)
    uvs=torch.tensor([[.2,.3,.4],[.8,.3,.6],[.5,.5,.5],[.1,.9,.2],[.9,.1,.8]],requires_grad=True)
    out=model(geom,nodes,uvs)
    s=out["p0"]+out["p1"]+out["p_lost"]
    assert torch.max(torch.abs(s-1)).item() < 1e-6
    # Exact mirror fate identity.
    mir=torch.stack([1-uvs[:,0],uvs[:,1],1-uvs[:,2]],dim=-1)
    outm=model(geom,nodes,mir)
    assert torch.max(torch.abs(out["p0"]-outm["p1"])).item() < 2e-6
    assert torch.max(torch.abs(out["p1"]-outm["p0"])).item() < 2e-6
    loss=(out["p0"].sum()+model.mean_time(out["mix0_logits"]).sum())
    loss.backward()
    assert torch.isfinite(geom.grad).all() and torch.isfinite(nodes.grad).all()
    print("Stage-6B surrogate self-test PASS")
    print("  timing basis normalization PASS")
    print("  fate normalization PASS")
    print("  exact mirror symmetry PASS")
    print("  geometry/source autograd PASS")


def cmd_train(args):
    _set_seed(args.seed)
    device=_device_from_arg(args.device)
    print("device:", device)
    fit_ids, val_ids, val_mode=_ids_for_first_fit(args.atlas,args.internal_val_geometries,args.seed)
    print(f"geometry split for this fit: train={len(fit_ids)} val={len(val_ids)} ({val_mode})")
    print("loading dense adjoint fields into memory ...")
    train=load_split_data(args.atlas,fit_ids)
    val=load_split_data(args.atlas,val_ids)
    gm,gs=fit_geom_normalization(train)
    med,bmass,bdens,bmeans=make_lognormal_basis(train.time_edges,args.mixtures,args.log_sigma)
    model=Stage6BSurrogate(gm,gs,bmass,bdens,bmeans,hidden=args.hidden,trunk=args.trunk).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.epochs,1),eta_min=args.lr*0.08)
    N=train.n_geom*train.n_vox
    rng=np.random.default_rng(args.seed+17)
    history=[]; best=float("inf"); best_epoch=0; start_epoch=1
    config={
        "mixtures":args.mixtures,"log_sigma":args.log_sigma,"hidden":args.hidden,"trunk":args.trunk,
        "time_edges_ns":train.time_edges.tolist(),"atlas":str(args.atlas),"seed":args.seed,
        "lambda_time":args.lambda_time,"lambda_mean":args.lambda_mean,
        "architecture":"ordered BiGRU five-segment skeleton + SiLU trunk + exact mirror + fixed lognormal timing basis",
    }
    latest=_latest_path(args.output)
    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"--resume requested but latest training checkpoint does not exist: {latest}")
        st=torch.load(latest,map_location=device,weights_only=False)
        if st.get("format") != FORMAT or st.get("kind") != "training-latest":
            raise ValueError("latest checkpoint is not a compatible Stage-6B training checkpoint")
        if list(st.get("train_ids",[])) != list(fit_ids) or list(st.get("val_ids",[])) != list(val_ids):
            raise ValueError("resume geometry split differs from checkpoint; refusing to mix fits")
        oldcfg=st.get("config",{})
        for k in ("mixtures","log_sigma","hidden","trunk","seed"):
            if oldcfg.get(k) != config.get(k):
                raise ValueError(f"resume configuration mismatch for {k}: {oldcfg.get(k)} vs {config.get(k)}")
        model.load_state_dict(st["model_state"]); opt.load_state_dict(st["optimizer_state"]); sched.load_state_dict(st["scheduler_state"])
        history=list(st.get("history",[])); best=float(st.get("best",float("inf"))); best_epoch=int(st.get("best_epoch",0))
        start_epoch=int(st.get("epoch",0))+1
        if "numpy_rng_state" in st: rng.bit_generator.state=st["numpy_rng_state"]
        if "torch_rng_state" in st: torch.set_rng_state(st["torch_rng_state"].cpu())
        print(f"resuming training from epoch {start_epoch}; best epoch so far={best_epoch}")
    for epoch in range(start_epoch,args.epochs+1):
        model.train()
        nuse=min(N,args.samples_per_epoch) if args.samples_per_epoch>0 else N
        order=rng.choice(N,size=nuse,replace=False) if nuse<N else rng.permutation(N)
        sums=np.zeros(4,float); nb=0
        for a in range(0,nuse,args.batch_size):
            ii=order[a:a+args.batch_size]
            batch=_batch_from_indices(train,ii,device)
            opt.zero_grad(set_to_none=True)
            total,rate,timing,meanloss=_loss_components(model,batch,args.lambda_time,args.lambda_mean,args.relse_scale)
            total.backward()
            if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            opt.step()
            sums += [total.item(),rate.item(),timing.item(),meanloss.item()]; nb+=1
        sched.step()
        # Cheap validation every epoch on a deterministic subset; full metrics periodically.
        vm=evaluate_data(model,val,device,batch_size=args.eval_batch_size,max_samples=args.val_samples,seed=991)
        score=vm["abs_log_p0_error_median"] + 0.6*vm["timing_cdf_maxdiff_median"] + 0.08*vm["mean_time_abs_error_ns_median"]
        row={"epoch":epoch,"loss":float(sums[0]/max(nb,1)),"rate":float(sums[1]/max(nb,1)),
             "timing":float(sums[2]/max(nb,1)),"mean":float(sums[3]/max(nb,1)),"val_score":float(score),
             "val_logp_med":vm["abs_log_p0_error_median"],"val_cdf_med":vm["timing_cdf_maxdiff_median"],
             "val_mt_med":vm["mean_time_abs_error_ns_median"],"lr":float(opt.param_groups[0]["lr"])}
        history.append(row)
        improved=score<best
        if improved:
            best=score; best_epoch=epoch
            save_checkpoint(args.output,model,config,gm,gs,med,fit_ids,val_ids,history,best_epoch)
        _save_latest(latest,model,opt,sched,config,gm,gs,med,fit_ids,val_ids,history,epoch,best,best_epoch,rng)
        if epoch==1 or epoch%args.print_every==0 or epoch==args.epochs or improved and epoch<=5:
            print(f"epoch {epoch:4d} loss={row['loss']:.4f} rate={row['rate']:.4f} timeKL={row['timing']:.4f} "
                  f"val |dlogp|={row['val_logp_med']:.3f} CDF={row['val_cdf_med']:.3f} dt={row['val_mt_med']:.3f}ns "
                  f"score={score:.4f}{' *' if improved else ''}")
    # Load best and make a fuller held-out report.
    model,ck=load_model(args.output,device)
    full=evaluate_data(model,val,device,batch_size=args.eval_batch_size,max_samples=args.final_eval_samples,seed=992)
    _print_metrics(f"BEST HELD-OUT GEOMETRY EVALUATION (epoch {ck['best_epoch']})",full)
    report=args.output.with_suffix(".metrics.json")
    report.write_text(json.dumps({"validation_mode":val_mode,"metrics":full,"train_ids":fit_ids,"val_ids":val_ids,
                                  "best_epoch":int(ck["best_epoch"])},indent=2))
    histp=args.output.with_suffix(".history.csv")
    with histp.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
    print("wrote",args.output)
    print("wrote",report)
    print("wrote",histp)


def _ids_for_eval(atlas: Path, ck, split: str):
    with h5py.File(atlas,"r") as f:
        by=_geometry_ids_by_split(f)
    if split=="holdout": return list(ck.get("val_ids",[]))
    if split=="trainfit": return list(ck.get("train_ids",[]))
    ids=list(by.get(split,[]))
    if not ids: raise ValueError(f"atlas currently contains no completed '{split}' geometries")
    return ids


def cmd_evaluate(args):
    device=_device_from_arg(args.device)
    model,ck=load_model(args.model,device)
    ids=_ids_for_eval(args.atlas,ck,args.split)
    data=load_split_data(args.atlas,ids)
    m=evaluate_data(model,data,device,batch_size=args.batch_size,max_samples=args.max_samples,seed=12345)
    _print_metrics(f"EVALUATION split={args.split} geometries={len(ids)}",m)
    if args.output:
        args.output.write_text(json.dumps(m,indent=2)); print("wrote",args.output)


def cmd_predict_point(args):
    device=_device_from_arg(args.device)
    model,ck=load_model(args.model,device)
    with h5py.File(args.atlas,"r") as f:
        g=f["geometries"][args.geometry_id]
        gg,nn=_geom_arrays(g); cell=g["cell_cm"][:]
        edges=f["time_edges_ns"][:]
    if args.u is not None:
        uvs=np.array([args.u,args.v,args.s],np.float32)
    else:
        uvs=np.array([args.x/cell[0],args.y/cell[1],args.z/cell[2]],np.float32)
    geom=torch.from_numpy(gg[None,:]).to(device); nodes=torch.from_numpy(nn[None,:,:]).to(device); src=torch.from_numpy(uvs[None,:]).to(device)
    with torch.no_grad():
        out=model(geom,nodes,src)
        f0=model.timing_density(out["mix0_logits"])[0].cpu().numpy(); f1=model.timing_density(out["mix1_logits"])[0].cpu().numpy()
        p0=float(out["p0"][0]); p1=float(out["p1"][0])
    tc=0.5*(edges[:-1]+edges[1:])
    outp=args.output
    outp.parent.mkdir(parents=True,exist_ok=True)
    with outp.open("w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["time_ns","f0_per_ns","f1_per_ns","h0_per_ns","h1_per_ns"])
        for t,a,b in zip(tc,f0,f1): w.writerow([float(t),float(a),float(b),float(p0*a),float(p1*b)])
    print(f"p0={p0:.8f} p1={p1:.8f} ptotal={p0+p1:.8f}")
    print("wrote",outp)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sp=p.add_subparsers(dest="cmd",required=True)
    q=sp.add_parser("self-test"); q.set_defaults(func=cmd_self_test)
    q=sp.add_parser("inspect"); q.add_argument("--atlas",type=Path,required=True); q.set_defaults(func=cmd_inspect)
    q=sp.add_parser("train")
    q.add_argument("--atlas",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
    q.add_argument("--device",default="auto"); q.add_argument("--epochs",type=int,default=80); q.add_argument("--batch-size",type=int,default=512)
    q.add_argument("--samples-per-epoch",type=int,default=65536); q.add_argument("--val-samples",type=int,default=24000); q.add_argument("--final-eval-samples",type=int,default=0)
    q.add_argument("--eval-batch-size",type=int,default=1024); q.add_argument("--internal-val-geometries",type=int,default=8)
    q.add_argument("--mixtures",type=int,default=24); q.add_argument("--log-sigma",type=float,default=0.34)
    q.add_argument("--hidden",type=int,default=40); q.add_argument("--trunk",type=int,default=128)
    q.add_argument("--lr",type=float,default=2e-3); q.add_argument("--weight-decay",type=float,default=1e-5); q.add_argument("--grad-clip",type=float,default=5.0)
    q.add_argument("--lambda-time",type=float,default=0.55); q.add_argument("--lambda-mean",type=float,default=0.15); q.add_argument("--relse-scale",type=float,default=0.35)
    q.add_argument("--seed",type=int,default=20261003); q.add_argument("--print-every",type=int,default=5); q.add_argument("--resume",action="store_true",help="resume from OUTPUT stem + .latest.pt") ; q.set_defaults(func=cmd_train)
    q=sp.add_parser("evaluate")
    q.add_argument("--atlas",type=Path,required=True); q.add_argument("--model",type=Path,required=True); q.add_argument("--split",choices=["holdout","trainfit","val","test"],default="holdout")
    q.add_argument("--device",default="auto"); q.add_argument("--batch-size",type=int,default=1024); q.add_argument("--max-samples",type=int,default=0); q.add_argument("--output",type=Path); q.set_defaults(func=cmd_evaluate)
    q=sp.add_parser("predict-point")
    q.add_argument("--atlas",type=Path,required=True); q.add_argument("--model",type=Path,required=True); q.add_argument("--geometry-id",required=True); q.add_argument("--device",default="auto")
    mode=q.add_mutually_exclusive_group(required=True); mode.add_argument("--u",type=float); mode.add_argument("--x",type=float)
    q.add_argument("--v",type=float); q.add_argument("--s",type=float); q.add_argument("--y",type=float); q.add_argument("--z",type=float)
    q.add_argument("--output",type=Path,default=Path("stage6B_point_response.csv")); q.set_defaults(func=cmd_predict_point)
    return p


def main():
    p=build_parser(); args=p.parse_args()
    if args.cmd=="predict-point":
        if args.u is not None and (args.v is None or args.s is None): p.error("with --u, also give --v and --s")
        if args.x is not None and (args.y is None or args.z is None): p.error("with --x, also give --y and --z")
    args.func(args)

if __name__=="__main__": main()
