#!/usr/bin/env python3
"""
PHINDER Stage-6A: weighted branching adjoint 3-D + time optical field.

For one readout channel and one detector geometry, launch reciprocal rays from
that channel's terminal CORE face. Significant Fresnel R/T branches are split
deterministically, small/budget-limited branches fall back to unbiased analog
sampling, and bulk absorption is handled by continuous Beer-Lambert weights.

Every weighted reverse segment inside the scintillator is deposited into a
uniform 3-D voxel grid. The output is the optical Green-function field per
emitted photon:

    p_c(x,y,z)               integrated detection probability
    h_c(x,y,z,t)             dP/dt [1/ns]
    M1_c(x,y,y,z)            integral t h(t) dt [ns per emitted photon]
    mean_t_c(x,y,z)          M1/p where defined

A wall-clock mode uses a scrambled 4-D Halton prefix, so every completed prefix
remains well distributed over terminal position/direction phase space. The run
can checkpoint and resume exactly from the next Halton/root index.

The 18 source positions in the existing high-statistics Stage-6 forward atlas
are scored simultaneously with the historical small-box estimator and written
as validation probes; this costs essentially no extra transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import qmc

from Photon_Simulator import CORE, EJ204
from Photon_Simulator.core.detectorSetup.material import ReflectiveMaterial

import phinder_stage6_adjoint_prototype as base
import phinder_stage6_adjoint_branching as branch


_W = {}


def _worker_init(cfg: dict, source_xyz: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray):
    scene, paths, core_w, core_h = base._build_from_cfg(cfg)
    pts, tans = paths[cfg["channel"]]
    Z = float(cfg["cell_cm"][2])
    thickness = float(cfg["guide_thickness_cm"])
    zc = Z + 0.5 * thickness if cfg["channel"] == 0 else -0.5 * thickness
    endpoint = np.array([pts[-1, 0], pts[-1, 1], zc], dtype=float)
    tf = np.asarray(tans[-1], float)
    tf /= np.linalg.norm(tf)
    outward = np.array([tf[0], tf[1], 0.0], dtype=float)
    inward = -outward
    lateral = np.array([-tf[1], tf[0], 0.0], dtype=float)
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)

    cell = np.asarray(cfg["cell_cm"], float)
    shape = tuple(int(x) for x in cfg["grid_shape"])
    spacing = cell / np.asarray(shape, float)

    _W.clear()
    _W.update(
        cfg=cfg,
        scene=scene,
        engine=base.ApproximateEngine(scene),
        endpoint=endpoint,
        outward=outward,
        inward=inward,
        lateral=lateral,
        vertical=vertical,
        core_w=core_w,
        core_h=core_h,
        cell=cell,
        grid_shape=shape,
        spacing=spacing,
        source_xyz=np.asarray(source_xyz, float),
        box_lo=np.asarray(box_lo, float),
        box_hi=np.asarray(box_hi, float),
    )


def _initial_reverse_ray(u: np.ndarray):
    cfg = _W["cfg"]
    edge_frac = 1.0 - 2.0e-10
    a = (float(u[0]) - 0.5) * _W["core_w"] * edge_frac
    b = (float(u[1]) - 0.5) * _W["core_h"] * edge_frac
    start = _W["endpoint"] + a * _W["lateral"] + b * _W["vertical"]
    start = start - float(cfg["start_epsilon_cm"]) * _W["outward"]

    # q(Omega) = cos(theta)/pi on the inward hemisphere.
    mu = math.sqrt(float(u[2]))
    st = math.sqrt(max(0.0, 1.0 - mu * mu))
    phi = 2.0 * math.pi * float(u[3])
    direction = (
        mu * _W["inward"]
        + st * math.cos(phi) * _W["lateral"]
        + st * math.sin(phi) * _W["vertical"]
    )
    direction /= np.linalg.norm(direction)
    return start, direction


def _grid_intervals(p0: np.ndarray, p1: np.ndarray):
    """Yield (ix,iy,iz,s0,s1) for a segment inside the axis-aligned scintillator.

    s0/s1 are physical distances from p0 along the segment. The algorithm uses
    exact crossings of the uniform-grid planes; no voxel-center stepping is
    involved, so summed sub-lengths reproduce the full segment length.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(d))
    if L <= 0.0:
        return []

    us = [0.0, 1.0]
    cell = _W["cell"]
    shape = _W["grid_shape"]
    spacing = _W["spacing"]

    for a in range(3):
        da = float(d[a])
        if abs(da) < 1e-15:
            continue
        x0 = float(p0[a])
        x1 = float(p1[a])
        lo = min(x0, x1)
        hi = max(x0, x1)
        h = float(spacing[a])
        # Only internal planes can split the segment.
        k0 = max(1, int(math.floor(lo / h)) + 1)
        k1 = min(int(shape[a]) - 1, int(math.floor((hi - 1e-14) / h)))
        for k in range(k0, k1 + 1):
            edge = k * h
            u = (edge - x0) / da
            if 1e-13 < u < 1.0 - 1e-13:
                us.append(float(u))

    us = np.asarray(sorted(us), float)
    # Remove numerical duplicates from a segment hitting a grid edge/corner.
    keep = np.ones(len(us), dtype=bool)
    if len(us) > 1:
        keep[1:] = np.diff(us) > 1e-12
    us = us[keep]

    out = []
    for ua, ub in zip(us[:-1], us[1:]):
        if ub <= ua:
            continue
        um = 0.5 * (ua + ub)
        pm = np.asarray(p0, float) + um * d
        idx = np.floor(pm / spacing).astype(int)
        idx = np.minimum(np.maximum(idx, 0), np.asarray(shape) - 1)
        out.append((int(idx[0]), int(idx[1]), int(idx[2]), ua * L, ub * L))
    return out


def _add_time_interval(hist_1d, t_start, n_over_c, weight0, mu, s0, s1, dt, tmax):
    if s1 <= s0 or weight0 <= 0.0:
        return
    ta = t_start + n_over_c * s0
    tb = t_start + n_over_c * s1
    if tb <= 0.0 or ta >= tmax:
        return
    aa = max(ta, 0.0)
    bb = min(tb, tmax)
    if bb <= aa:
        return
    ss0 = max(s0, (aa - t_start) / n_over_c)
    ss1 = min(s1, (bb - t_start) / n_over_c)
    i0 = max(0, int(math.floor(aa / dt)))
    i1 = min(len(hist_1d) - 1, int(math.floor(np.nextafter(bb, -np.inf) / dt)))
    for ib in range(i0, i1 + 1):
        tl = max(aa, ib * dt)
        tr = min(bb, (ib + 1) * dt)
        if tr <= tl:
            continue
        sl = max(ss0, (tl - t_start) / n_over_c)
        sr = min(ss1, (tr - t_start) / n_over_c)
        I0, _ = branch._exp_integral(mu, sl, sr)
        hist_1d[ib] += weight0 * I0


def _score_scint_segment(
    p0, p1, t0, weight0, mu, n,
    Lgrid, Ltgrid, field_hist,
    Lprobe, Ltprobe, probe_hist,
):
    seg = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(seg))
    if L <= 0.0 or weight0 <= 0.0:
        return
    cfg = _W["cfg"]
    n_over_c = float(n) / float(_W["engine"].config.c_cm_ns)

    # Main field: exact track-length partition over uniform voxels.
    for ix, iy, iz, s0, s1 in _grid_intervals(p0, p1):
        I0, I1 = branch._exp_integral(mu, s0, s1)
        Wlen = weight0 * I0
        if Wlen <= 0.0:
            continue
        Wt = weight0 * (t0 * I0 + n_over_c * I1)
        Lgrid[ix, iy, iz] += Wlen
        Ltgrid[ix, iy, iz] += Wt
        _add_time_interval(
            field_hist[ix, iy, iz], t0, n_over_c, weight0, mu, s0, s1,
            cfg["dt_ns"], cfg["tmax_ns"]
        )

    # Historical 18 source-centered kernels, retained strictly as validation
    # probes against the existing forward atlas.
    for isrc, (lo, hi) in enumerate(zip(_W["box_lo"], _W["box_hi"])):
        ab = base._segment_box_interval(p0, p1, lo, hi)
        if ab is None:
            continue
        a, b = ab
        s0 = a * L
        s1 = b * L
        I0, I1 = branch._exp_integral(mu, s0, s1)
        Wlen = weight0 * I0
        if Wlen <= 0.0:
            continue
        Lprobe[isrc] += Wlen
        Ltprobe[isrc] += weight0 * (t0 * I0 + n_over_c * I1)
        _add_time_interval(
            probe_hist[isrc], t0, n_over_c, weight0, mu, s0, s1,
            cfg["dt_ns"], cfg["tmax_ns"]
        )


_MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """Small deterministic 64-bit mixer (SplitMix64 finalizer)."""
    x = int(x) & _MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK64
    x ^= x >> 31
    return x & _MASK64


def _branch_child_key(parent_key: int, step: int, side: int) -> int:
    """Stable branch identifier, independent of stack traversal order."""
    x = int(parent_key) ^ _mix64(int(step) + 0x9E3779B97F4A7C15)
    x ^= 0xD1B54A32D192ED03 if int(side) == 0 else 0x94D049BB133111EB
    return _mix64(x)


def _keyed_uniform01(seed: int, ray_id: int, branch_key: int, step: int) -> float:
    """Stateless U[0,1) draw for common-random-number geometry pairs.

    Matching roots/branches/interactions in two nearby geometries receive the
    same uniform deviate even if another branch diverges and consumes a
    different number of random choices.
    """
    x = _mix64(int(seed) ^ 0xA0761D6478BD642F)
    x ^= _mix64(int(ray_id) + 0xE7037ED1A0B428DB)
    x ^= _mix64(int(branch_key) + 0x8EBC6AF09C88C6E3)
    x ^= _mix64(int(step) + 0x589965CC75374CC3)
    z = _mix64(x)
    # Use the upper 53 bits so conversion to double is exact and never yields 1.
    return float((z >> 11) * (1.0 / (1 << 53)))


def _rng_for_root(ray_id: int):
    cfg = _W["cfg"]
    ss = np.random.SeedSequence([int(cfg["physics_seed"]), int(ray_id)])
    return np.random.default_rng(ss)


def _track_root(ray_id: int, u: np.ndarray):
    """Track one detector phase-space root and return per-root fields.

    The timing histograms are per-root too. This costs a few MB/root at the
    default grid, but guarantees that a rare transport exception cannot leave
    partial histogram weight in an otherwise rejected root.
    """
    cfg = _W["cfg"]
    shape = _W["grid_shape"]
    nsrc = len(_W["source_xyz"])
    nb = int(math.ceil(cfg["tmax_ns"] / cfg["dt_ns"]))
    Lgrid = np.zeros(shape, dtype=np.float64)
    Ltgrid = np.zeros(shape, dtype=np.float64)
    field_hist = np.zeros(shape + (nb,), dtype=np.float64)
    Lprobe = np.zeros(nsrc, dtype=np.float64)
    Ltprobe = np.zeros(nsrc, dtype=np.float64)
    probe_hist = np.zeros((nsrc, nb), dtype=np.float64)
    use_keyed_crn = bool(cfg.get("keyed_analog_rng", False))
    rng = None if use_keyed_crn else _rng_for_root(ray_id)

    p0, d0 = _initial_reverse_ray(u)
    loc = _W["scene"].locate_volume(p0)
    if loc is None or loc.name != f"guide{cfg['channel']}_terminal":
        raise RuntimeError(f"reverse start not inside terminal core: {None if loc is None else loc.name}")

    # Stack entries carry a stable branch key.  It is ignored by ordinary
    # standalone field runs but makes analog fallback choices pairable across
    # nearby geometries in derivative calculations.
    root_branch_key = _mix64(int(ray_id) ^ 0x243F6A8885A308D3)
    stack = [(np.asarray(p0, float), np.asarray(d0, float), 0.0, 1.0, 0, root_branch_key)]
    nodes_created = 1
    max_live = 1
    splits = 0
    analog_fallbacks = 0
    boundary_events = 0
    reached_scint = False
    terminal_weight = Counter()
    bulk_absorbed_weight = 0.0

    while stack:
        p, d, t, w, steps, branch_key = stack.pop()
        if w <= 0.0:
            continue

        while True:
            if steps >= int(cfg["max_steps"]):
                terminal_weight["UNRESOLVED_MAX_STEPS"] += w
                break
            if t >= float(cfg["max_flight_time_ns"]):
                terminal_weight["UNRESOLVED_MAX_TIME"] += w
                break

            hit = _W["scene"].first_boundary_hit(p, d)
            if hit is None:
                terminal_weight["NO_BOUNDARY_HIT"] += w
                break

            mat = hit.from_material
            L = float(hit.distance)
            if L <= 0.0:
                terminal_weight["ZERO_DISTANCE"] += w
                break
            n = float(mat.refractive_index)
            mu = float(mat.absorption_mu)
            dtseg = L * n / float(_W["engine"].config.c_cm_ns)

            if t + dtseg > float(cfg["max_flight_time_ns"]):
                Lcap = max(
                    0.0,
                    (float(cfg["max_flight_time_ns"]) - t)
                    * float(_W["engine"].config.c_cm_ns) / n,
                )
                pcap = p + Lcap * d
                if hit.from_volume is not None and hit.from_volume.name == "scintillator" and Lcap > 0.0:
                    reached_scint = True
                    _score_scint_segment(
                        p, pcap, t, w, mu, n,
                        Lgrid, Ltgrid, field_hist,
                        Lprobe, Ltprobe, probe_hist,
                    )
                wcap = w * math.exp(-mu * Lcap)
                bulk_absorbed_weight += max(0.0, w - wcap)
                terminal_weight["UNRESOLVED_MAX_TIME"] += wcap
                break

            intercept = np.asarray(hit.point, float)
            t_hit = t + dtseg

            if hit.from_volume is not None and hit.from_volume.name == "scintillator":
                reached_scint = True
                _score_scint_segment(
                    p, intercept, t, w, mu, n,
                    Lgrid, Ltgrid, field_hist,
                    Lprobe, Ltprobe, probe_hist,
                )

            w_boundary = w * math.exp(-mu * L)
            bulk_absorbed_weight += max(0.0, w - w_boundary)
            if w_boundary <= 0.0:
                break

            boundary_events += 1
            steps += 1

            ch_readout = None
            if getattr(_W["scene"], "readout_ports", ()):
                ch_readout = _W["scene"].readout_channel(
                    hit, d, tolerance=_W["engine"].config.catch_tolerance
                )
            if ch_readout is not None:
                terminal_weight[f"READOUT_PORT_{int(ch_readout)}"] += w_boundary
                break

            if hit.from_volume.point_on_edge(intercept):
                if hit.to_volume is not None and hit.from_material == hit.to_material:
                    p, t, w = intercept, t_hit, w_boundary
                    continue
                terminal_weight["AMBIGUOUS_MATERIAL_EDGE"] += w_boundary
                break

            normal = _W["engine"]._oriented_normal(hit.normal, d)

            if isinstance(hit.to_material, ReflectiveMaterial):
                rm = hit.to_material
                if float(rm.specular_fraction) < 1.0 - 1e-12:
                    raise NotImplementedError("weighted adjoint does not yet support diffuse reflector components")
                wr = w_boundary * float(rm.reflectance)
                terminal_weight["REFLECTOR_ABSORBED"] += w_boundary - wr
                if wr <= 0.0:
                    break
                d = _W["engine"]._reflected_direction(d, normal)
                p, t, w = intercept, t_hit, wr
                continue

            n_from = float(hit.from_material.refractive_index)
            n_to = float(hit.to_material.refractive_index)
            theta_in = math.acos(float(np.clip(abs(np.dot(d, normal)), -1.0, 1.0)))
            coeffs = _W["engine"]._fresnel_coefficients(theta_in, n_from, n_to)
            R = max(0.0, float(coeffs.R))
            T = max(0.0, float(coeffs.T))
            sRT = R + T
            if sRT <= 0.0:
                terminal_weight["INTERFACE_ABSORBED"] += w_boundary
                break
            R /= sRT
            T /= sRT

            if T <= 1e-15:
                d = _W["engine"]._reflected_direction(d, normal)
                p, t, w = intercept, t_hit, w_boundary
                continue
            if R <= 1e-15:
                ok, dt = _W["engine"]._transmitted_direction(d, normal, n_from, n_to)
                if not ok:
                    d = _W["engine"]._reflected_direction(d, normal)
                    p, t, w = intercept, t_hit, w_boundary
                    continue
                if hit.to_volume is None:
                    terminal_weight["ESCAPED_OUTSIDE"] += w_boundary
                    break
                p, d, t, w = intercept, dt, t_hit, w_boundary
                continue

            wr = w_boundary * R
            wt = w_boundary * T
            can_split = (
                min(wr, wt) >= float(cfg["split_min_child_weight"])
                and len(stack) + 2 <= int(cfg["max_live_branches"])
                and nodes_created + 1 <= int(cfg["max_branch_nodes"])
            )

            if can_split:
                dr = _W["engine"]._reflected_direction(d, normal)
                ok, dt = _W["engine"]._transmitted_direction(d, normal, n_from, n_to)
                if not ok:
                    p, d, t, w = intercept, dr, t_hit, w_boundary
                    continue

                splits += 1
                nodes_created += 1
                reflected_key = _branch_child_key(branch_key, steps, 0)
                transmitted_key = _branch_child_key(branch_key, steps, 1)
                reflected = (intercept, dr, t_hit, wr, steps, reflected_key)
                if hit.to_volume is None:
                    terminal_weight["ESCAPED_OUTSIDE"] += wt
                    p, d, t, w, steps, branch_key = reflected
                    continue
                transmitted = (intercept, dt, t_hit, wt, steps, transmitted_key)
                if wr >= wt:
                    stack.append(transmitted)
                    p, d, t, w, steps, branch_key = reflected
                else:
                    stack.append(reflected)
                    p, d, t, w, steps, branch_key = transmitted
                max_live = max(max_live, len(stack) + 1)
                continue

            # Unbiased branch-control fallback.
            analog_fallbacks += 1
            if use_keyed_crn:
                uanalog = _keyed_uniform01(cfg["physics_seed"], ray_id, branch_key, steps)
            else:
                uanalog = float(rng.random())
            if uanalog < R:
                d = _W["engine"]._reflected_direction(d, normal)
                p, t, w = intercept, t_hit, w_boundary
                continue
            ok, dt = _W["engine"]._transmitted_direction(d, normal, n_from, n_to)
            if not ok:
                d = _W["engine"]._reflected_direction(d, normal)
                p, t, w = intercept, t_hit, w_boundary
                continue
            if hit.to_volume is None:
                terminal_weight["ESCAPED_OUTSIDE"] += w_boundary
                break
            p, d, t, w = intercept, dt, t_hit, w_boundary

    return dict(
        Lgrid=Lgrid,
        Ltgrid=Ltgrid,
        Lprobe=Lprobe,
        Ltprobe=Ltprobe,
        field_hist=field_hist,
        probe_hist=probe_hist,
        reached_scint=int(reached_scint),
        bulk_absorbed_weight=bulk_absorbed_weight,
        terminal_weight=dict(terminal_weight),
        splits=splits,
        analog_fallbacks=analog_fallbacks,
        nodes_created=nodes_created,
        max_live=max_live,
        boundary_events=boundary_events,
    )


def _run_chunk(task):
    start_id, launch_u = task
    launch_u = np.asarray(launch_u, float)
    cfg = _W["cfg"]
    shape = _W["grid_shape"]
    nsrc = len(_W["source_xyz"])
    nb = int(math.ceil(cfg["tmax_ns"] / cfg["dt_ns"]))

    out = dict(
        n=0,
        sumL=np.zeros(shape),
        sumL2=np.zeros(shape),
        sumLt=np.zeros(shape),
        hist=np.zeros(shape + (nb,)),
        hit_roots=np.zeros(shape, dtype=np.int64),
        sumProbeL=np.zeros(nsrc),
        sumProbeL2=np.zeros(nsrc),
        sumProbeLt=np.zeros(nsrc),
        probeHist=np.zeros((nsrc, nb)),
        probeHitRoots=np.zeros(nsrc, dtype=np.int64),
        sumLvol=0.0, sumLvol2=0.0, sumLtvol=0.0,
        reached_scint=0,
        bulk_absorbed_weight=0.0,
        terminal_weight=Counter(),
        splits=0,
        analog_fallbacks=0,
        nodes_created=0,
        max_live=0,
        boundary_events=0,
        failed=0,
    )

    for j, u in enumerate(launch_u):
        ray_id = int(start_id + j)
        try:
            r = _track_root(ray_id, u)
        except Exception:
            out["failed"] += 1
            continue
        out["n"] += 1
        Lg = r["Lgrid"]
        out["sumL"] += Lg
        out["sumL2"] += Lg * Lg
        out["sumLt"] += r["Ltgrid"]
        out["hist"] += r["field_hist"]
        out["hit_roots"] += (Lg > 0.0)
        Lvol = float(np.sum(Lg))
        Ltvol = float(np.sum(r["Ltgrid"]))
        out["sumLvol"] += Lvol
        out["sumLvol2"] += Lvol * Lvol
        out["sumLtvol"] += Ltvol
        Lp = r["Lprobe"]
        out["sumProbeL"] += Lp
        out["sumProbeL2"] += Lp * Lp
        out["sumProbeLt"] += r["Ltprobe"]
        out["probeHist"] += r["probe_hist"]
        out["probeHitRoots"] += (Lp > 0.0)
        out["reached_scint"] += r["reached_scint"]
        out["bulk_absorbed_weight"] += r["bulk_absorbed_weight"]
        out["terminal_weight"].update(r["terminal_weight"])
        out["splits"] += r["splits"]
        out["analog_fallbacks"] += r["analog_fallbacks"]
        out["nodes_created"] += r["nodes_created"]
        out["max_live"] = max(out["max_live"], r["max_live"])
        out["boundary_events"] += r["boundary_events"]
    return out


def _empty_merge(shape, nsrc, nb):
    return dict(
        n=0,
        attempted=0,
        next_id=0,
        sumL=np.zeros(shape),
        sumL2=np.zeros(shape),
        sumLt=np.zeros(shape),
        hist=np.zeros(shape + (nb,)),
        hit_roots=np.zeros(shape, dtype=np.int64),
        sumProbeL=np.zeros(nsrc),
        sumProbeL2=np.zeros(nsrc),
        sumProbeLt=np.zeros(nsrc),
        probeHist=np.zeros((nsrc, nb)),
        probeHitRoots=np.zeros(nsrc, dtype=np.int64),
        sumLvol=0.0, sumLvol2=0.0, sumLtvol=0.0,
        reached_scint=0,
        bulk_absorbed_weight=0.0,
        terminal_weight=Counter(),
        splits=0,
        analog_fallbacks=0,
        nodes_created=0,
        max_live=0,
        boundary_events=0,
        failed=0,
    )


def _merge_into(out, r, attempted):
    out["attempted"] += int(attempted)
    out["n"] += int(r["n"])
    for k in (
        "sumL", "sumL2", "sumLt", "hist", "hit_roots",
        "sumProbeL", "sumProbeL2", "sumProbeLt", "probeHist", "probeHitRoots",
    ):
        out[k] += r[k]
    for k in ("reached_scint", "splits", "analog_fallbacks", "nodes_created", "boundary_events", "failed"):
        out[k] += int(r[k])
    for k in ("sumLvol", "sumLvol2", "sumLtvol"):
        out[k] += float(r[k])
    out["bulk_absorbed_weight"] += float(r["bulk_absorbed_weight"])
    out["terminal_weight"].update({str(k): float(v) for k, v in r["terminal_weight"].items()})
    out["max_live"] = max(out["max_live"], int(r["max_live"]))


def _config_fingerprint(cfg_for_hash: dict):
    s = json.dumps(cfg_for_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _checkpoint_path(prefix: Path):
    return prefix.with_name(prefix.name + "_checkpoint.npz")


def _save_checkpoint(path: Path, merged: dict, fingerprint: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(
        tmp,
        fingerprint=np.array(fingerprint),
        n=np.array(merged["n"], dtype=np.int64),
        attempted=np.array(merged["attempted"], dtype=np.int64),
        next_id=np.array(merged["next_id"], dtype=np.int64),
        sumL=merged["sumL"], sumL2=merged["sumL2"], sumLt=merged["sumLt"],
        hist=merged["hist"], hit_roots=merged["hit_roots"],
        sumProbeL=merged["sumProbeL"], sumProbeL2=merged["sumProbeL2"], sumProbeLt=merged["sumProbeLt"],
        probeHist=merged["probeHist"], probeHitRoots=merged["probeHitRoots"],
        sumLvol=np.array(merged["sumLvol"], dtype=np.float64),
        sumLvol2=np.array(merged["sumLvol2"], dtype=np.float64),
        sumLtvol=np.array(merged["sumLtvol"], dtype=np.float64),
        reached_scint=np.array(merged["reached_scint"], dtype=np.int64),
        bulk_absorbed_weight=np.array(merged["bulk_absorbed_weight"], dtype=np.float64),
        terminal_weight_json=np.array(json.dumps(dict(merged["terminal_weight"]))),
        splits=np.array(merged["splits"], dtype=np.int64),
        analog_fallbacks=np.array(merged["analog_fallbacks"], dtype=np.int64),
        nodes_created=np.array(merged["nodes_created"], dtype=np.int64),
        max_live=np.array(merged["max_live"], dtype=np.int64),
        boundary_events=np.array(merged["boundary_events"], dtype=np.int64),
        failed=np.array(merged["failed"], dtype=np.int64),
    )
    os.replace(tmp, path)


def _load_checkpoint(path: Path, fingerprint: str, shape, nsrc, nb):
    z = np.load(path, allow_pickle=False)
    got = str(z["fingerprint"].item())
    if got != fingerprint:
        raise RuntimeError(
            "checkpoint configuration does not match this run; use a new output prefix or restore the original options"
        )
    out = _empty_merge(shape, nsrc, nb)
    for k in ("n", "attempted", "next_id", "reached_scint", "splits", "analog_fallbacks", "nodes_created", "max_live", "boundary_events", "failed"):
        out[k] = int(z[k].item())
    out["sumLvol"] = float(z["sumLvol"].item())
    out["sumLvol2"] = float(z["sumLvol2"].item())
    out["sumLtvol"] = float(z["sumLtvol"].item())
    out["bulk_absorbed_weight"] = float(z["bulk_absorbed_weight"].item())
    for k in (
        "sumL", "sumL2", "sumLt", "hist", "hit_roots",
        "sumProbeL", "sumProbeL2", "sumProbeLt", "probeHist", "probeHitRoots",
    ):
        out[k] = np.array(z[k])
    out["terminal_weight"] = Counter(json.loads(str(z["terminal_weight_json"].item())))
    return out


def _run_transport(cfg, xyz, box_lo, box_hi, args, merged, fingerprint):
    method = "fork" if os.name != "nt" else "spawn"
    ctx = mp.get_context(method)
    if method == "fork":
        print("warming geometry/Numba kernels once before fork ...")
        _worker_init(cfg, xyz, box_lo, box_hi)
        try:
            _track_root(10**9, np.array([0.31, 0.47, 0.59, 0.73]))
        except Exception:
            pass

    sampler = qmc.Halton(d=4, scramble=True, seed=int(args.launch_seed))
    if merged["next_id"] > 0:
        sampler.fast_forward(int(merged["next_id"]))

    start_wall = time.monotonic()
    budget_s = max(0.0, 60.0 * float(args.time_budget_min))
    hard_cap = int(args.reverse_rays)
    next_id = int(merged["next_id"])
    base_done = next_id
    nominal_round = max(1, int(args.jobs) * int(args.chunk_size))
    last_print = start_wall
    last_checkpoint = start_wall
    ckpt_path = _checkpoint_path(args.output_prefix)

    with ctx.Pool(
        processes=args.jobs,
        initializer=_worker_init,
        initargs=(cfg, xyz, box_lo, box_hi),
    ) as pool:
        while True:
            now = time.monotonic()
            elapsed = now - start_wall
            if budget_s > 0.0 and elapsed >= budget_s and next_id > base_done:
                break
            if hard_cap > 0 and next_id >= hard_cap:
                break

            nround = nominal_round
            if hard_cap > 0:
                nround = min(nround, hard_cap - next_id)
            if budget_s > 0.0 and next_id > base_done and elapsed > 0.0:
                rate = (next_id - base_done) / elapsed
                remaining = max(0.0, budget_s - elapsed)
                predicted = max(1, int(rate * remaining))
                if remaining < elapsed / max(next_id - base_done, 1) * nominal_round:
                    nround = min(nround, predicted)
            if nround <= 0:
                break

            launch = sampler.random(nround)
            tasks = []
            off = 0
            while off < nround:
                n = min(int(args.chunk_size), nround - off)
                tasks.append((next_id + off, launch[off:off+n]))
                off += n
            results = pool.map(_run_chunk, tasks)
            for task, r in zip(tasks, results):
                _merge_into(merged, r, attempted=len(task[1]))
            next_id += nround
            merged["next_id"] = next_id

            now = time.monotonic()
            if now - last_print >= 5.0 or (hard_cap > 0 and next_id >= hard_cap):
                print(
                    f"  progress prefix={next_id} (+{next_id-base_done} this run) elapsed={now-start_wall:.1f}s "
                    f"successful={merged['n']} reach-scint={merged['reached_scint']}/{max(merged['n'],1)} "
                    f"splits/root={merged['splits']/max(merged['n'],1):.2f}"
                )
                last_print = now
            if args.checkpoint_every_min > 0 and now - last_checkpoint >= 60.0 * args.checkpoint_every_min:
                _save_checkpoint(ckpt_path, merged, fingerprint)
                print(f"  checkpoint {ckpt_path} at prefix={next_id}")
                last_checkpoint = now

    elapsed = time.monotonic() - start_wall
    _save_checkpoint(ckpt_path, merged, fingerprint)
    return merged, elapsed


def _trilinear(field, xyz, cell):
    """Approximate point value from a voxel-center field; diagnostic only."""
    shape = np.asarray(field.shape[:3], int)
    spacing = np.asarray(cell, float) / shape
    q = np.asarray(xyz, float) / spacing - 0.5
    lo = np.floor(q).astype(int)
    f = q - lo
    lo = np.minimum(np.maximum(lo, 0), shape - 1)
    hi = np.minimum(lo + 1, shape - 1)
    # Clamp interpolation fraction when outside center range.
    for a in range(3):
        if q[a] <= 0:
            lo[a] = hi[a] = 0; f[a] = 0.0
        elif q[a] >= shape[a] - 1:
            lo[a] = hi[a] = shape[a] - 1; f[a] = 0.0
    val = 0.0
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                ix = hi[0] if bx else lo[0]
                iy = hi[1] if by else lo[1]
                iz = hi[2] if bz else lo[2]
                w = ((f[0] if bx else 1-f[0]) * (f[1] if by else 1-f[1]) * (f[2] if bz else 1-f[2]))
                val += w * float(field[ix, iy, iz])
    return val


def _grid_self_test():
    # Set a tiny fake worker context containing only grid/time pieces.
    _W.clear()
    _W.update(
        cell=np.array([2.0, 1.0, 1.0]),
        grid_shape=(4, 2, 2),
        spacing=np.array([0.5, 0.5, 0.5]),
    )
    p0 = np.array([0.0, 0.25, 0.25])
    p1 = np.array([2.0, 0.25, 0.25])
    parts = _grid_intervals(p0, p1)
    total = sum(b-a for _,_,_,a,b in parts)
    if len(parts) != 4 or abs(total - 2.0) > 1e-12:
        raise AssertionError((len(parts), total, parts))
    mu = 0.3
    pieces = sum(branch._exp_integral(mu, a, b)[0] for *_, a, b in parts)
    exact = branch._exp_integral(mu, 0.0, 2.0)[0]
    if abs(pieces - exact) > 1e-12:
        raise AssertionError((pieces, exact))
    print("Stage-6A field grid self-test PASS")
    print("  2 cm segment -> 4 exact voxels; length conservation PASS")
    print("  Beer-Lambert weighted track-length partition PASS")


def main():
    ap = argparse.ArgumentParser(description="PHINDER Stage-6A weighted branching adjoint 3-D+time field")
    ap.add_argument("--self-test", action="store_true", help="run only voxel/weight partition unit tests")
    ap.add_argument("--forward-atlas", type=Path, default=Path("phinder_bridge/data/stage6_map_hi_atlas.h5"))
    ap.add_argument("--geometry-id", default="g000")
    ap.add_argument("--channel", type=int, choices=(0, 1), default=0)
    ap.add_argument("--reverse-rays", type=int, default=0, help="total Halton-prefix root cap; 0 means no count cap")
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="wall-clock budget for this invocation")
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--chunk-size", type=int, default=2)
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--dt-ns", type=float, default=0.25)
    ap.add_argument("--tmax-ns", type=float, default=60.0)
    ap.add_argument("--probe-kernel-xy-cm", type=float, default=0.30)
    ap.add_argument("--probe-kernel-z-cm", type=float, default=0.10)
    ap.add_argument("--launch-seed", type=int, default=606061)
    ap.add_argument("--physics-seed", type=int, default=808081)
    ap.add_argument("--keyed-analog-rng", action="store_true",
                    help="use stateless per-root/branch/interaction analog draws; intended for CRN derivative pairs")
    ap.add_argument("--split-min-child-weight", type=float, default=0.005)
    ap.add_argument("--max-live-branches", type=int, default=64)
    ap.add_argument("--max-branch-nodes", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=750)
    ap.add_argument("--max-flight-time-ns", type=float, default=80.0)
    ap.add_argument("--start-epsilon-cm", type=float, default=1e-7)
    ap.add_argument("--checkpoint-every-min", type=float, default=5.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-prefix", type=Path, default=Path("phinder_bridge/data/stage6_adjoint_field_ch0"))
    args = ap.parse_args()

    if args.self_test:
        _grid_self_test()
        return
    if args.reverse_rays <= 0 and args.time_budget_min <= 0:
        raise ValueError("specify --reverse-rays N (>0), --time-budget-min M (>0), or both")
    if min(args.nx, args.ny, args.nz) < 1:
        raise ValueError("grid dimensions must be positive")
    if args.dt_ns <= 0 or args.tmax_ns <= 0:
        raise ValueError("time binning must be positive")
    if args.max_flight_time_ns < args.tmax_ns:
        raise ValueError("--max-flight-time-ns should be >= --tmax-ns so the requested time field can be populated")

    sid, xyz, fw, fw_hist, meta = base._forward_truth(
        args.forward_atlas, args.geometry_id, args.channel, args.dt_ns, args.tmax_ns
    )
    nsrc = len(sid)
    nb = fw_hist.shape[1]
    shape = (args.nx, args.ny, args.nz)
    cell = np.asarray(meta["cell_cm"], float)

    cfg = dict(
        geometry_id=args.geometry_id,
        width_cm=meta["width_cm"], radius_cm=meta["radius_cm"], facets=meta["facets"],
        guide_thickness_cm=meta["guide_thickness_cm"], cladding_thickness_cm=meta["cladding_thickness_cm"],
        cell_cm=cell,
        path0_points=meta["path0_points"], path0_tangents=meta["path0_tangents"],
        path1_points=meta["path1_points"], path1_tangents=meta["path1_tangents"],
        channel=args.channel,
        physics_seed=args.physics_seed,
        keyed_analog_rng=bool(args.keyed_analog_rng),
        split_min_child_weight=args.split_min_child_weight,
        max_live_branches=args.max_live_branches,
        max_branch_nodes=args.max_branch_nodes,
        max_steps=args.max_steps,
        max_flight_time_ns=args.max_flight_time_ns,
        start_epsilon_cm=args.start_epsilon_cm,
        dt_ns=args.dt_ns, tmax_ns=args.tmax_ns,
        grid_shape=shape,
    )

    half = np.array([0.5*args.probe_kernel_xy_cm, 0.5*args.probe_kernel_xy_cm, 0.5*args.probe_kernel_z_cm])
    box_lo = np.maximum(0.0, xyz - half)
    box_hi = np.minimum(cell, xyz + half)
    probe_volumes = np.prod(box_hi - box_lo, axis=1)
    probe_clipped = np.any(np.abs((box_hi-box_lo)-2*half) > 1e-12, axis=1)

    scene, paths, core_w, core_h = base._build_from_cfg(cfg)
    area = core_w * core_h
    n_port = float(CORE.refractive_index)
    n_sc = float(EJ204.refractive_index)
    reciprocity_norm = 0.25 * area * (n_port / n_sc) ** 2
    voxel_spacing = cell / np.asarray(shape, float)
    voxel_volume = float(np.prod(voxel_spacing))
    cell_volume = float(np.prod(cell))

    fingerprint_payload = dict(
        forward_atlas=str(args.forward_atlas.resolve()), geometry_id=args.geometry_id, channel=args.channel,
        cell_cm=cell.tolist(), width_cm=float(meta["width_cm"]), radius_cm=float(meta["radius_cm"]),
        grid_shape=list(shape), dt_ns=args.dt_ns, tmax_ns=args.tmax_ns,
        probe_kernel_xy_cm=args.probe_kernel_xy_cm, probe_kernel_z_cm=args.probe_kernel_z_cm,
        launch_seed=args.launch_seed, physics_seed=args.physics_seed,
        split_min_child_weight=args.split_min_child_weight, max_live_branches=args.max_live_branches,
        max_branch_nodes=args.max_branch_nodes, max_steps=args.max_steps,
        max_flight_time_ns=args.max_flight_time_ns, start_epsilon_cm=args.start_epsilon_cm,
    )
    # Preserve compatibility with checkpoints produced before keyed CRN existed:
    # the false/default case hashes exactly the historical payload.
    if args.keyed_analog_rng:
        fingerprint_payload["keyed_analog_rng"] = True
    fingerprint = _config_fingerprint(fingerprint_payload)

    ckpt_path = _checkpoint_path(args.output_prefix)
    if args.resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint does not exist: {ckpt_path}")
        merged = _load_checkpoint(ckpt_path, fingerprint, shape, nsrc, nb)
        print(f"resuming checkpoint {ckpt_path}: prefix={merged['next_id']} successful roots={merged['n']}")
    else:
        if ckpt_path.exists():
            print(f"NOTE: existing checkpoint ignored (no --resume): {ckpt_path}")
        merged = _empty_merge(shape, nsrc, nb)

    print("PHINDER Stage-6A weighted branching adjoint FIELD")
    print(f"forward atlas: {args.forward_atlas}  geometry={args.geometry_id}  channel={args.channel}")
    print(f"grid: {args.nx} x {args.ny} x {args.nz}  voxel={voxel_spacing[0]:.4f} x {voxel_spacing[1]:.4f} x {voxel_spacing[2]:.4f} cm")
    print(f"time field: dt={args.dt_ns:g} ns  tmax={args.tmax_ns:g} ns  bins={nb}")
    print(f"terminal CORE area={area:.8f} cm^2  reciprocity prefactor={reciprocity_norm:.9g} cm^2")
    print(f"branch control: split min child={args.split_min_child_weight:g}; max live={args.max_live_branches}; max nodes/root={args.max_branch_nodes}")
    print(f"analog RNG: {'stateless keyed CRN' if args.keyed_analog_rng else 'sequential per-root'}")
    cap = "unlimited" if args.reverse_rays <= 0 else str(args.reverse_rays)
    print(f"launch: scrambled Halton prefix; current prefix={merged['next_id']}; cap={cap}; time budget this run={args.time_budget_min:g} min")

    merged, elapsed = _run_transport(cfg, xyz, box_lo, box_hi, args, merged, fingerprint)
    N = int(merged["n"])
    if N <= 0:
        raise RuntimeError("no successful adjoint roots")

    meanL = merged["sumL"] / N
    varL = np.maximum((merged["sumL2"] - merged["sumL"]**2/N) / max(N-1,1), 0.0)
    seL = np.sqrt(varL / N)
    p_field = reciprocity_norm * meanL / voxel_volume
    p_se = reciprocity_norm * seL / voxel_volume
    m1_field = reciprocity_norm * (merged["sumLt"] / N) / voxel_volume
    mean_t_field = np.divide(merged["sumLt"], merged["sumL"], out=np.full(shape, np.nan), where=merged["sumL"]>0)
    h_field = reciprocity_norm * merged["hist"] / (N * voxel_volume * args.dt_ns)
    rel_se = np.divide(p_se, p_field, out=np.full(shape, np.nan), where=p_field>0)
    touch_fraction = merged["hit_roots"] / N

    # Timing-field integral can be smaller than p if some weighted track length
    # lies beyond tmax. Report that explicitly instead of silently renormalizing.
    p_from_time = np.sum(h_field, axis=-1) * args.dt_ns
    time_coverage = np.divide(p_from_time, p_field, out=np.full(shape, np.nan), where=p_field>0)

    # Validation probes.
    meanProbeL = merged["sumProbeL"] / N
    varProbeL = np.maximum((merged["sumProbeL2"] - merged["sumProbeL"]**2/N) / max(N-1,1), 0.0)
    seProbeL = np.sqrt(varProbeL / N)
    p_probe = reciprocity_norm * meanProbeL / probe_volumes
    se_probe = reciprocity_norm * seProbeL / probe_volumes
    mean_t_probe = np.divide(merged["sumProbeLt"], merged["sumProbeL"], out=np.full(nsrc,np.nan), where=merged["sumProbeL"]>0)
    h_probe = reciprocity_norm * merged["probeHist"] / (N * probe_volumes[:,None] * args.dt_ns)

    n_fw, nc_fw, p_fw, se_fw, mean_t_fw = fw.T
    comb_se = np.sqrt(se_fw**2 + se_probe**2)
    z = np.divide(p_probe-p_fw, comb_se, out=np.full(nsrc,np.nan), where=comb_se>0)
    ratio = np.divide(p_probe,p_fw,out=np.full(nsrc,np.nan),where=p_fw>0)
    ks_bin=np.full(nsrc,np.nan)
    for i in range(nsrc):
        a=h_probe[i]; b=fw_hist[i]
        if a.sum()>0 and b.sum()>0:
            ks_bin[i]=float(np.max(np.abs(np.cumsum(a)/a.sum()-np.cumsum(b)/b.sum())))
    fit_scale=base._weighted_scale(p_probe,p_fw,np.sqrt(se_fw**2+se_probe**2))
    corr=float(np.corrcoef(p_probe,p_fw)[0,1]) if nsrc>1 else float("nan")

    validation_rows=[]
    for i in range(nsrc):
        validation_rows.append(dict(
            source_id=int(sid[i]), x_cm=xyz[i,0], y_cm=xyz[i,1], z_cm=xyz[i,2],
            p_forward=p_fw[i], se_forward=se_fw[i],
            p_adjoint_probe=p_probe[i], se_adjoint_probe=se_probe[i], ratio_probe=ratio[i], z_probe=z[i],
            forward_mean_t_ns=mean_t_fw[i], adjoint_probe_mean_t_ns=mean_t_probe[i],
            binned_timing_cdf_maxdiff=ks_bin[i], roots_touching_probe=int(merged["probeHitRoots"][i]),
            probe_volume_cm3=probe_volumes[i], probe_clipped=bool(probe_clipped[i]),
            p_field_trilinear=_trilinear(p_field, xyz[i], cell),
        ))

    # Whole-volume field normalization. Because all voxels have equal volume,
    # the volume mean is simply the mean of p_field.
    pvol_adj=reciprocity_norm * (merged["sumLvol"] / N) / cell_volume
    varLvol=max((merged["sumLvol2"]-merged["sumLvol"]**2/N)/max(N-1,1),0.0)
    pvol_se=reciprocity_norm * math.sqrt(varLvol/N) / cell_volume
    pgrid_fw=float(np.mean(p_fw))
    se_pgrid_fw=float(math.sqrt(np.sum(se_fw**2))/len(se_fw))

    total_track=float(np.sum(merged["sumL"]))
    total_hist_track=float(np.sum(merged["hist"]))
    global_time_track_coverage=total_hist_track/total_track if total_track>0 else float("nan")
    tvol_adj=float(merged["sumLtvol"]/merged["sumLvol"]) if merged["sumLvol"]>0 else float("nan")

    terminal_mean={k:float(v)/N for k,v in merged["terminal_weight"].items()}
    bulk_mean=float(merged["bulk_absorbed_weight"])/N
    accounted=bulk_mean+sum(terminal_mean.values())
    unresolved=terminal_mean.get("UNRESOLVED_MAX_TIME",0.0)+terminal_mean.get("UNRESOLVED_MAX_STEPS",0.0)

    out=args.output_prefix
    out.parent.mkdir(parents=True,exist_ok=True)
    h5_path=out.with_name(out.name+".h5")
    csv_path=out.with_name(out.name+"_validation.csv")
    json_path=out.with_name(out.name+"_summary.json")

    x_edges=np.linspace(0.0,cell[0],args.nx+1)
    y_edges=np.linspace(0.0,cell[1],args.ny+1)
    z_edges=np.linspace(0.0,cell[2],args.nz+1)
    time_edges=np.arange(nb+1)*args.dt_ns

    with h5py.File(h5_path,"w") as f:
        f.attrs["format"]="PHINDER weighted adjoint optical field"
        f.attrs["format_version"]=1
        f.attrs["geometry_id"]=args.geometry_id
        f.attrs["channel"]=args.channel
        f.attrs["root_launches_successful"]=N
        f.attrs["halton_prefix_length"]=int(merged["next_id"])
        f.attrs["analytic_reciprocity_prefactor_cm2"]=reciprocity_norm
        f.attrs["cell_x_cm"]=cell[0]; f.attrs["cell_y_cm"]=cell[1]; f.attrs["cell_z_cm"]=cell[2]
        f.attrs["dt_ns"]=args.dt_ns; f.attrs["tmax_ns"]=args.tmax_ns
        f.attrs["config_fingerprint"]=fingerprint
        f.create_dataset("x_edges_cm",data=x_edges)
        f.create_dataset("y_edges_cm",data=y_edges)
        f.create_dataset("z_edges_cm",data=z_edges)
        f.create_dataset("time_edges_ns",data=time_edges)
        g=f.create_group("field")
        g.create_dataset("p_per_emitted",data=p_field,compression="gzip",shuffle=True)
        g.create_dataset("p_se",data=p_se,compression="gzip",shuffle=True)
        g.create_dataset("relative_se_p",data=rel_se,compression="gzip",shuffle=True)
        g.create_dataset("m1_ns_per_emitted",data=m1_field,compression="gzip",shuffle=True)
        g.create_dataset("mean_transport_time_ns",data=mean_t_field,compression="gzip",shuffle=True)
        g.create_dataset("roots_touching",data=merged["hit_roots"],compression="gzip",shuffle=True)
        g.create_dataset("root_touch_fraction",data=touch_fraction,compression="gzip",shuffle=True)
        g.create_dataset("time_integral_coverage",data=time_coverage,compression="gzip",shuffle=True)
        g.create_dataset(
            "h_per_ns", data=h_field.astype(np.float32), compression="gzip", shuffle=True,
            chunks=(min(args.nx,4), min(args.ny,4), min(args.nz,2), min(nb,64))
        )
        vg=f.create_group("validation")
        vg.create_dataset("source_id",data=sid)
        vg.create_dataset("source_xyz_cm",data=xyz)
        vg.create_dataset("p_forward",data=p_fw)
        vg.create_dataset("se_forward",data=se_fw)
        vg.create_dataset("p_adjoint_probe",data=p_probe)
        vg.create_dataset("se_adjoint_probe",data=se_probe)
        vg.create_dataset("mean_t_forward_ns",data=mean_t_fw)
        vg.create_dataset("mean_t_adjoint_probe_ns",data=mean_t_probe)
        vg.create_dataset("h_forward_per_ns",data=fw_hist.astype(np.float32),compression="gzip")
        vg.create_dataset("h_adjoint_probe_per_ns",data=h_probe.astype(np.float32),compression="gzip")

    pd.DataFrame(validation_rows).to_csv(csv_path,index=False)

    summary=dict(
        forward_atlas=str(args.forward_atlas), geometry_id=args.geometry_id, channel=args.channel,
        grid_shape=list(shape), voxel_spacing_cm=voxel_spacing.tolist(), voxel_volume_cm3=voxel_volume,
        dt_ns=args.dt_ns, tmax_ns=args.tmax_ns, root_launches_successful=N,
        root_launches_attempted=int(merged["attempted"]), halton_prefix_length=int(merged["next_id"]),
        elapsed_this_invocation_s=elapsed, analytic_reciprocity_prefactor_cm2=reciprocity_norm,
        reached_scintillator_fraction=merged["reached_scint"]/N, failed_roots=int(merged["failed"]),
        deterministic_splits_per_root=merged["splits"]/N,
        analog_fallbacks_per_root=merged["analog_fallbacks"]/N,
        branch_nodes_per_root=merged["nodes_created"]/N,
        max_live_branches_observed=int(merged["max_live"]),
        boundary_events_per_root=merged["boundary_events"]/N,
        bulk_absorbed_weight_per_root=bulk_mean, terminal_weight_per_root=terminal_mean,
        weight_accounting_sum_per_root=accounted, unresolved_weight_per_root=unresolved,
        validation_best_fit_scale_forward_over_adjoint=fit_scale,
        validation_source_map_correlation=corr,
        validation_median_abs_z=float(np.nanmedian(np.abs(z))),
        validation_max_abs_z=float(np.nanmax(np.abs(z))),
        validation_median_binned_timing_cdf_maxdiff=float(np.nanmedian(ks_bin)),
        field_volume_mean_p=pvol_adj,
        field_volume_mean_p_se=pvol_se,
        forward_18point_grid_mean_p=pgrid_fw,
        forward_18point_grid_se=se_pgrid_fw,
        forward_grid_over_field_volume_mean=pgrid_fw/pvol_adj if pvol_adj>0 else float("nan"),
        field_volume_mean_transport_t_ns=tvol_adj,
        global_time_track_coverage_to_tmax=global_time_track_coverage,
        checkpoint=str(ckpt_path),
        note="Voxel p SE and whole-volume p SE are root-wise Monte Carlo uncertainties. Resume uses the exact next scrambled-Halton prefix index.",
    )
    json_path.write_text(json.dumps(summary,indent=2))

    finite_rel=rel_se[np.isfinite(rel_se)]
    finite_cov=time_coverage[np.isfinite(time_coverage)]
    print("\nfield summary:")
    print(f"  roots successful/attempted       = {N}/{merged['attempted']}  Halton prefix={merged['next_id']}")
    print(f"  roots reaching scintillator      = {merged['reached_scint']}/{N} ({100*merged['reached_scint']/N:.2f}%)")
    print(f"  deterministic splits/root        = {merged['splits']/N:.2f}")
    print(f"  analog fallback decisions/root   = {merged['analog_fallbacks']/N:.2f}")
    print(f"  unresolved weight/root           = {unresolved:.6g}")
    print(f"  weight-accounting sum/root       = {accounted:.6f}")
    print(f"  volume-mean p(field)             = {pvol_adj:.6f} +/- {pvol_se:.6f}")
    print(f"  forward 18-point mean p          = {pgrid_fw:.6f} +/- {se_pgrid_fw:.6f}")
    print(f"  forward-grid / field-volume      = {pgrid_fw/pvol_adj:.4f}")
    print(f"  volume mean transport t          = {tvol_adj:.3f} ns")
    print(f"  global h(t) coverage to tmax     = {100*global_time_track_coverage:.3f}%")
    if finite_rel.size:
        print(f"  voxel relative-SE median / p90   = {100*np.nanmedian(finite_rel):.1f}% / {100*np.nanpercentile(finite_rel,90):.1f}%")
    if finite_cov.size:
        print(f"  voxel time-coverage median       = {100*np.nanmedian(finite_cov):.2f}%")
    print("validation probes against existing forward atlas:")
    print(f"  best-fit forward/adjoint scale   = {fit_scale:.4f}")
    print(f"  source-map correlation           = {corr:.4f}")
    print(f"  median/max |combined z|          = {np.nanmedian(np.abs(z)):.2f} / {np.nanmax(np.abs(z)):.2f}")
    print(f"  median timing CDF maxdiff        = {np.nanmedian(ks_bin):.3f}")
    print(f"wrote {h5_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"checkpoint {ckpt_path} retained for --resume")
    print("Stage-6A ADJOINT FIELD COMPLETE")


if __name__ == "__main__":
    main()
