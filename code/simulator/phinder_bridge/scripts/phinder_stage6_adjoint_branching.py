#!/usr/bin/env python3
"""
PHINDER Stage-6A: weighted branching adjoint / reciprocal transport.

This extends phinder_stage6_adjoint_prototype.py.  Instead of drawing one
Fresnel outcome at every interface, significant reflection/transmission
branches are propagated simultaneously with their exact Fresnel weights.
Branch growth is controlled without bias by falling back to ordinary analog
sampling when either child is too small or a branch budget would be exceeded.
Bulk absorption is treated continuously with Beer-Lambert weights rather than
by killing rays stochastically.

The estimator is still validated against the 18 source positions in the
existing high-statistics forward Stage-6 atlas.  One launched detector phase-
space sample is one statistical root sample; all branches descended from that
root are accumulated before computing per-root variances.

A wall-clock mode is supported.  Launch points/directions use a scrambled
4-D Halton sequence, so every completed prefix remains well spread over the
terminal phase space.  Work is dispatched in complete contiguous rounds; a
time-limited run therefore always ends on a homogeneous prefix (with at most
one-round budget overshoot).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from Photon_Simulator import PhotonStatus, CORE, EJ204
from Photon_Simulator.core.detectorSetup.material import ReflectiveMaterial

import phinder_stage6_adjoint_prototype as base


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

    # q(Omega)=cos(theta)/pi on inward hemisphere.
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


def _exp_integral(mu: float, s0: float, s1: float):
    """Return I0=integral exp(-mu*s) ds and I1=integral s exp(-mu*s) ds."""
    if s1 <= s0:
        return 0.0, 0.0
    if mu <= 0.0:
        return s1 - s0, 0.5 * (s1 * s1 - s0 * s0)
    e0 = math.exp(-mu * s0)
    e1 = math.exp(-mu * s1)
    I0 = (e0 - e1) / mu
    I1 = (s0 / mu + 1.0 / (mu * mu)) * e0 - (s1 / mu + 1.0 / (mu * mu)) * e1
    return I0, I1


def _add_weighted_time_interval(hist, t_start, n_over_c, w0, mu, s0, s1, dt, tmax):
    """Accumulate integral w(s) ds into time bins for s in [s0,s1]."""
    if s1 <= s0 or w0 <= 0.0:
        return
    ta = t_start + n_over_c * s0
    tb = t_start + n_over_c * s1
    if tb <= 0.0 or ta >= tmax:
        return
    aa = max(ta, 0.0)
    bb = min(tb, tmax)
    if bb <= aa:
        return
    # Convert clipped time interval back to distance from segment start.
    ss0 = max(s0, (aa - t_start) / n_over_c)
    ss1 = min(s1, (bb - t_start) / n_over_c)
    i0 = max(0, int(math.floor(aa / dt)))
    i1 = min(len(hist) - 1, int(math.floor(np.nextafter(bb, -np.inf) / dt)))
    for ib in range(i0, i1 + 1):
        tl = max(aa, ib * dt)
        tr = min(bb, (ib + 1) * dt)
        if tr <= tl:
            continue
        sl = max(ss0, (tl - t_start) / n_over_c)
        sr = min(ss1, (tr - t_start) / n_over_c)
        I0, _ = _exp_integral(mu, sl, sr)
        hist[ib] += w0 * I0


def _score_scint_segment_weighted(p0, p1, t0, weight0, mu, n, Lvec, Ltvec, hist):
    seg = p1 - p0
    L = float(np.linalg.norm(seg))
    if L <= 0.0 or weight0 <= 0.0:
        return 0.0, 0.0
    cfg = _W["cfg"]
    n_over_c = float(n) / float(_W["engine"].config.c_cm_ns)

    # Whole-volume weighted track integrals (for a cheap normalization check).
    I0all, I1all = _exp_integral(mu, 0.0, L)
    Lvol = weight0 * I0all
    Ltvol = weight0 * (t0 * I0all + n_over_c * I1all)

    for isrc, (lo, hi) in enumerate(zip(_W["box_lo"], _W["box_hi"])):
        ab = base._segment_box_interval(p0, p1, lo, hi)
        if ab is None:
            continue
        a, b = ab
        s0 = a * L
        s1 = b * L
        I0, I1 = _exp_integral(mu, s0, s1)
        Wlen = weight0 * I0
        if Wlen <= 0.0:
            continue
        Wt = weight0 * (t0 * I0 + n_over_c * I1)
        Lvec[isrc] += Wlen
        Ltvec[isrc] += Wt
        _add_weighted_time_interval(
            hist[isrc], t0, n_over_c, weight0, mu, s0, s1,
            cfg["dt_ns"], cfg["tmax_ns"]
        )
    return Lvol, Ltvol


def _rng_for_root(ray_id: int):
    cfg = _W["cfg"]
    ss = np.random.SeedSequence([int(cfg["physics_seed"]), int(ray_id)])
    return np.random.default_rng(ss)


def _track_root(ray_id: int, u: np.ndarray):
    """Track one detector phase-space root, with weighted deterministic splitting."""
    cfg = _W["cfg"]
    nsrc = len(_W["source_xyz"])
    nb = int(math.ceil(cfg["tmax_ns"] / cfg["dt_ns"]))
    Lvec = np.zeros(nsrc, float)
    Ltvec = np.zeros(nsrc, float)
    hist = np.zeros((nsrc, nb), float)
    rng = _rng_for_root(ray_id)

    p0, d0 = _initial_reverse_ray(u)
    loc = _W["scene"].locate_volume(p0)
    if loc is None or loc.name != f"guide{cfg['channel']}_terminal":
        raise RuntimeError(f"reverse start not inside terminal core: {None if loc is None else loc.name}")

    # Stack entries: (position, direction, time, weight, interaction_steps)
    stack = [(np.asarray(p0, float), np.asarray(d0, float), 0.0, 1.0, 0)]
    nodes_created = 1
    max_live = 1
    splits = 0
    analog_fallbacks = 0
    boundary_events = 0
    reached_scint = False
    weighted_Lvol = 0.0
    weighted_Ltvol = 0.0
    terminal_weight = Counter()
    bulk_absorbed_weight = 0.0

    while stack:
        p, d, t, w, steps = stack.pop()
        if w <= 0.0:
            continue

        # Continue this branch until it terminates or creates a split.
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

            # Respect flight-time cap inside the current segment, not only at boundaries.
            if t + dtseg > float(cfg["max_flight_time_ns"]):
                Lcap = max(0.0, (float(cfg["max_flight_time_ns"]) - t) * float(_W["engine"].config.c_cm_ns) / n)
                pcap = p + Lcap * d
                if hit.from_volume is not None and hit.from_volume.name == "scintillator" and Lcap > 0.0:
                    reached_scint = True
                    lv, ltv = _score_scint_segment_weighted(p, pcap, t, w, mu, n, Lvec, Ltvec, hist)
                    weighted_Lvol += lv
                    weighted_Ltvol += ltv
                wcap = w * math.exp(-mu * Lcap)
                bulk_absorbed_weight += max(0.0, w - wcap)
                terminal_weight["UNRESOLVED_MAX_TIME"] += wcap
                break

            intercept = np.asarray(hit.point, float)
            t_hit = t + dtseg

            if hit.from_volume is not None and hit.from_volume.name == "scintillator":
                reached_scint = True
                lv, ltv = _score_scint_segment_weighted(p, intercept, t, w, mu, n, Lvec, Ltvec, hist)
                weighted_Lvol += lv
                weighted_Ltvol += ltv

            w_boundary = w * math.exp(-mu * L)
            bulk_absorbed_weight += max(0.0, w - w_boundary)
            if w_boundary <= 0.0:
                terminal_weight["BULK_ABSORBED_TO_ZERO"] += 0.0
                break

            boundary_events += 1
            steps += 1

            # In a reverse calculation, crossing any readout port outward is a terminal escape.
            ch_readout = None
            if getattr(_W["scene"], "readout_ports", ()):
                ch_readout = _W["scene"].readout_channel(
                    hit, d, tolerance=_W["engine"].config.catch_tolerance
                )
            if ch_readout is not None:
                terminal_weight[f"READOUT_PORT_{int(ch_readout)}"] += w_boundary
                break

            # Preserve the Stage-4c edge policy: same-material segmentation edge is transparent.
            if hit.from_volume.point_on_edge(intercept):
                if hit.to_volume is not None and hit.from_material == hit.to_material:
                    p, t, w = intercept, t_hit, w_boundary
                    continue
                terminal_weight["AMBIGUOUS_MATERIAL_EDGE"] += w_boundary
                break

            normal = _W["engine"]._oriented_normal(hit.normal, d)

            # Reflective wrapping: current backend supports specular reflection only.
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
            # Guard harmless floating roundoff in R+T.
            R /= sRT
            T /= sRT

            if T <= 1e-15:
                d = _W["engine"]._reflected_direction(d, normal)
                p, t, w = intercept, t_hit, w_boundary
                continue
            if R <= 1e-15:
                ok, dt = _W["engine"]._transmitted_direction(d, normal, n_from, n_to)
                if not ok:
                    # Numerically this should be TIR, already handled above.
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
                    # Treat as TIR rather than creating an invalid transmitted child.
                    p, d, t, w = intercept, dr, t_hit, w_boundary
                    continue

                splits += 1
                nodes_created += 1
                # Continue the larger-weight child and push the smaller one.
                # A transmitted child outside the scene terminates immediately.
                reflected = (intercept, dr, t_hit, wr, steps)
                if hit.to_volume is None:
                    terminal_weight["ESCAPED_OUTSIDE"] += wt
                    p, d, t, w, steps = reflected
                    continue
                transmitted = (intercept, dt, t_hit, wt, steps)
                if wr >= wt:
                    stack.append(transmitted)
                    p, d, t, w, steps = reflected
                else:
                    stack.append(reflected)
                    p, d, t, w, steps = transmitted
                max_live = max(max_live, len(stack) + 1)
                continue

            # Unbiased branch-control fallback: sample one outcome and retain the
            # full incoming weight.  This prevents combinatorial growth without
            # pruning any expectation value.
            analog_fallbacks += 1
            if float(rng.random()) < R:
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
        Lvec=Lvec, Ltvec=Ltvec, hist=hist,
        hit=(Lvec > 0.0).astype(np.int64),
        reached_scint=int(reached_scint),
        Lvol=weighted_Lvol, Ltvol=weighted_Ltvol,
        bulk_absorbed_weight=bulk_absorbed_weight,
        terminal_weight=dict(terminal_weight),
        splits=splits, analog_fallbacks=analog_fallbacks,
        nodes_created=nodes_created, max_live=max_live,
        boundary_events=boundary_events,
    )


def _run_chunk(task):
    start_id, launch_u = task
    launch_u = np.asarray(launch_u, float)
    cfg = _W["cfg"]
    nsrc = len(_W["source_xyz"])
    nb = int(math.ceil(cfg["tmax_ns"] / cfg["dt_ns"]))

    out = dict(
        n=0,
        sumL=np.zeros(nsrc), sumL2=np.zeros(nsrc), sumLt=np.zeros(nsrc),
        hist=np.zeros((nsrc, nb)), hit_rays=np.zeros(nsrc, dtype=np.int64),
        reached_scint=0,
        sumLvol=0.0, sumLvol2=0.0, sumLtvol=0.0,
        bulk_absorbed_weight=0.0, terminal_weight=Counter(),
        splits=0, analog_fallbacks=0, nodes_created=0, max_live=0,
        boundary_events=0, failed=0,
    )
    for j, u in enumerate(launch_u):
        ray_id = int(start_id + j)
        try:
            r = _track_root(ray_id, u)
        except Exception:
            out["failed"] += 1
            continue
        out["n"] += 1
        out["sumL"] += r["Lvec"]
        out["sumL2"] += r["Lvec"] * r["Lvec"]
        out["sumLt"] += r["Ltvec"]
        out["hist"] += r["hist"]
        out["hit_rays"] += r["hit"]
        out["reached_scint"] += r["reached_scint"]
        out["sumLvol"] += r["Lvol"]
        out["sumLvol2"] += r["Lvol"] * r["Lvol"]
        out["sumLtvol"] += r["Ltvol"]
        out["bulk_absorbed_weight"] += r["bulk_absorbed_weight"]
        out["terminal_weight"].update(r["terminal_weight"])
        out["splits"] += r["splits"]
        out["analog_fallbacks"] += r["analog_fallbacks"]
        out["nodes_created"] += r["nodes_created"]
        out["max_live"] = max(out["max_live"], r["max_live"])
        out["boundary_events"] += r["boundary_events"]
    return out


def _empty_merge(nsrc, nb):
    return dict(
        n=0, attempted=0,
        sumL=np.zeros(nsrc), sumL2=np.zeros(nsrc), sumLt=np.zeros(nsrc),
        hist=np.zeros((nsrc, nb)), hit_rays=np.zeros(nsrc, dtype=np.int64),
        reached_scint=0,
        sumLvol=0.0, sumLvol2=0.0, sumLtvol=0.0,
        bulk_absorbed_weight=0.0, terminal_weight=Counter(),
        splits=0, analog_fallbacks=0, nodes_created=0, max_live=0,
        boundary_events=0, failed=0,
    )


def _merge_into(out, r, attempted):
    out["attempted"] += int(attempted)
    out["n"] += int(r["n"])
    for k in ("sumL", "sumL2", "sumLt", "hist", "hit_rays"):
        out[k] += r[k]
    for k in ("reached_scint", "splits", "analog_fallbacks", "nodes_created", "boundary_events", "failed"):
        out[k] += int(r[k])
    for k in ("sumLvol", "sumLvol2", "sumLtvol", "bulk_absorbed_weight"):
        out[k] += float(r[k])
    out["terminal_weight"].update({str(k): float(v) for k, v in r["terminal_weight"].items()})
    out["max_live"] = max(out["max_live"], int(r["max_live"]))


def _run_transport(cfg, xyz, box_lo, box_hi, args, nsrc, nb):
    method = "fork" if os.name != "nt" else "spawn"
    ctx = mp.get_context(method)
    if method == "fork":
        print("warming geometry/Numba kernels once before fork ...")
        _worker_init(cfg, xyz, box_lo, box_hi)
        # Warm helper kernels only; do not use this result.
        u = np.array([[0.31, 0.47, 0.59, 0.73]])
        try:
            _run_chunk((10**9, u))
        except Exception:
            pass

    sampler = qmc.Halton(d=4, scramble=True, seed=int(args.launch_seed))
    merged = _empty_merge(nsrc, nb)
    next_id = 0
    start_wall = time.monotonic()
    budget_s = max(0.0, 60.0 * float(args.time_budget_min))
    hard_cap = int(args.reverse_rays)
    round_roots = max(1, int(args.jobs) * int(args.chunk_size))
    last_print = start_wall

    with ctx.Pool(
        processes=args.jobs,
        initializer=_worker_init,
        initargs=(cfg, xyz, box_lo, box_hi),
    ) as pool:
        while True:
            elapsed = time.monotonic() - start_wall
            if budget_s > 0.0 and elapsed >= budget_s and next_id > 0:
                break
            if hard_cap > 0 and next_id >= hard_cap:
                break

            nround = round_roots
            if hard_cap > 0:
                nround = min(nround, hard_cap - next_id)
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

            now = time.monotonic()
            if now - last_print >= 5.0 or (hard_cap > 0 and next_id >= hard_cap):
                print(
                    f"  progress roots={next_id} elapsed={now-start_wall:.1f}s "
                    f"reach-scint={merged['reached_scint']}/{max(merged['n'],1)} "
                    f"splits/root={merged['splits']/max(merged['n'],1):.2f} "
                    f"analog/root={merged['analog_fallbacks']/max(merged['n'],1):.2f}"
                )
                last_print = now

    elapsed = time.monotonic() - start_wall
    return merged, elapsed


def main():
    ap = argparse.ArgumentParser(description="PHINDER Stage-6A weighted branching adjoint transport")
    ap.add_argument("--forward-atlas", type=Path, default=Path("phinder_bridge/data/stage6_map_hi_atlas.h5"))
    ap.add_argument("--geometry-id", default="g000")
    ap.add_argument("--channel", type=int, choices=(0, 1), default=0)
    ap.add_argument("--reverse-rays", type=int, default=0,
                    help="root launch cap; 0 means no count cap (use with --time-budget-min)")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="approximate wall-clock transport budget; completes the current homogeneous round before stopping")
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--chunk-size", type=int, default=8,
                    help="root launches per worker task; a time-budget run overshoots by at most about jobs*chunk-size roots")
    ap.add_argument("--kernel-xy-cm", type=float, default=0.30)
    ap.add_argument("--kernel-z-cm", type=float, default=0.10)
    ap.add_argument("--dt-ns", type=float, default=0.25)
    ap.add_argument("--tmax-ns", type=float, default=40.0)
    ap.add_argument("--launch-seed", type=int, default=606061)
    ap.add_argument("--physics-seed", type=int, default=808081,
                    help="seed for analog fallback decisions when deterministic splitting is suppressed")
    ap.add_argument("--split-min-child-weight", type=float, default=0.005,
                    help="split Fresnel R/T deterministically only when both child weights exceed this value")
    ap.add_argument("--max-live-branches", type=int, default=64,
                    help="soft branching cap; exceeding it triggers unbiased analog fallback rather than pruning")
    ap.add_argument("--max-branch-nodes", type=int, default=1024,
                    help="soft per-root total branch-node cap; exceeding it triggers analog fallback")
    ap.add_argument("--max-steps", type=int, default=750,
                    help="hard interaction cap per branch; unresolved remaining weight is reported")
    ap.add_argument("--max-flight-time-ns", type=float, default=80.0,
                    help="hard reverse-flight cap per branch; unresolved remaining weight is reported")
    ap.add_argument("--start-epsilon-cm", type=float, default=1e-7)
    ap.add_argument("--output-prefix", type=Path, default=Path("phinder_bridge/data/stage6_adjoint_branching_ch0"))
    args = ap.parse_args()

    if args.reverse_rays <= 0 and args.time_budget_min <= 0:
        raise ValueError("specify --reverse-rays N (>0), --time-budget-min M (>0), or both")
    if args.jobs < 1 or args.chunk_size < 1:
        raise ValueError("jobs/chunk-size must be positive")
    if not (0.0 <= args.split_min_child_weight < 0.5):
        raise ValueError("split-min-child-weight must be in [0,0.5)")
    if args.max_live_branches < 1 or args.max_branch_nodes < 1:
        raise ValueError("branch caps must be positive")

    sid, xyz, fw, fw_hist, meta = base._forward_truth(
        args.forward_atlas, args.geometry_id, args.channel, args.dt_ns, args.tmax_ns
    )
    nsrc = len(sid)
    nb = fw_hist.shape[1]
    cfg = dict(
        geometry_id=args.geometry_id,
        width_cm=meta["width_cm"], radius_cm=meta["radius_cm"], facets=meta["facets"],
        guide_thickness_cm=meta["guide_thickness_cm"],
        cladding_thickness_cm=meta["cladding_thickness_cm"],
        cell_cm=np.asarray(meta["cell_cm"], float),
        path0_points=meta["path0_points"], path0_tangents=meta["path0_tangents"],
        path1_points=meta["path1_points"], path1_tangents=meta["path1_tangents"],
        channel=args.channel,
        physics_seed=args.physics_seed,
        split_min_child_weight=args.split_min_child_weight,
        max_live_branches=args.max_live_branches,
        max_branch_nodes=args.max_branch_nodes,
        max_steps=args.max_steps,
        max_flight_time_ns=args.max_flight_time_ns,
        start_epsilon_cm=args.start_epsilon_cm,
        dt_ns=args.dt_ns, tmax_ns=args.tmax_ns,
    )
    scene, paths, core_w, core_h = base._build_from_cfg(cfg)

    half = np.array([0.5*args.kernel_xy_cm, 0.5*args.kernel_xy_cm, 0.5*args.kernel_z_cm])
    cell = np.asarray(meta["cell_cm"], float)
    box_lo = np.maximum(0.0, xyz - half)
    box_hi = np.minimum(cell, xyz + half)
    volumes = np.prod(box_hi - box_lo, axis=1)
    clipped = np.any(np.abs((box_hi-box_lo)-2*half) > 1e-12, axis=1)

    area = core_w * core_h
    n_port = float(CORE.refractive_index)
    n_sc = float(EJ204.refractive_index)
    reciprocity_norm = 0.25 * area * (n_port / n_sc) ** 2
    cell_volume = float(np.prod(cell))

    print("PHINDER Stage-6A weighted branching adjoint")
    print(f"forward atlas: {args.forward_atlas}  geometry={args.geometry_id}  channel={args.channel}")
    print("geometry rebuild: PASS")
    print(f"terminal CORE area={area:.8f} cm^2  reciprocity prefactor={reciprocity_norm:.9g} cm^2")
    print(f"branch control: split if min child >= {args.split_min_child_weight:g}; max live={args.max_live_branches}; max nodes/root={args.max_branch_nodes}")
    print(f"bulk absorption: continuous Beer-Lambert weighting")
    if args.time_budget_min > 0:
        cap = "unlimited" if args.reverse_rays <= 0 else str(args.reverse_rays)
        print(f"launch mode: scrambled Halton prefix; time budget={args.time_budget_min:g} min; root cap={cap}")
    else:
        print(f"launch mode: scrambled Halton prefix; roots={args.reverse_rays}")

    rr, elapsed = _run_transport(cfg, xyz, box_lo, box_hi, args, nsrc, nb)
    N = int(rr["n"])
    if N <= 0:
        raise RuntimeError("no successful adjoint roots")

    meanL = rr["sumL"] / N
    varL = np.maximum((rr["sumL2"] - rr["sumL"]**2/N) / max(N-1,1), 0.0)
    se_meanL = np.sqrt(varL / N)
    p_adj = reciprocity_norm * meanL / volumes
    se_adj = reciprocity_norm * se_meanL / volumes
    mean_t_adj = np.divide(rr["sumLt"], rr["sumL"], out=np.full(nsrc,np.nan), where=rr["sumL"]>0)
    h_adj = reciprocity_norm * rr["hist"] / (N * volumes[:,None] * args.dt_ns)

    n_fw, nc_fw, p_fw, se_fw, mean_t_fw = fw.T
    comb_se = np.sqrt(se_fw**2 + se_adj**2)
    z = np.divide(p_adj-p_fw, comb_se, out=np.full(nsrc,np.nan), where=comb_se>0)
    ratio = np.divide(p_adj,p_fw,out=np.full(nsrc,np.nan),where=p_fw>0)
    ks_bin=np.full(nsrc,np.nan)
    for i in range(nsrc):
        a=h_adj[i]; b=fw_hist[i]
        if a.sum()>0 and b.sum()>0:
            ks_bin[i]=float(np.max(np.abs(np.cumsum(a)/a.sum()-np.cumsum(b)/b.sum())))

    scale_unc=np.sqrt(se_fw**2+se_adj**2)
    fit_scale=base._weighted_scale(p_adj,p_fw,scale_unc)
    corr=float(np.corrcoef(p_adj,p_fw)[0,1]) if nsrc>1 else float("nan")
    rel_rms=float(np.sqrt(np.nanmean(((p_adj-p_fw)/np.maximum(p_fw,1e-15))**2)))

    # Whole-volume track-length estimator.  The 18-point forward average is only
    # an approximate regular-grid quadrature, but this comparison has far lower
    # reverse-MC variance than the tiny local boxes.
    meanLvol=rr["sumLvol"]/N
    varLvol=max((rr["sumLvol2"]-rr["sumLvol"]**2/N)/max(N-1,1),0.0)
    pvol_adj=reciprocity_norm*meanLvol/cell_volume
    se_pvol_adj=reciprocity_norm*math.sqrt(varLvol/N)/cell_volume
    pgrid_fw=float(np.mean(p_fw))
    se_pgrid_fw=float(math.sqrt(np.sum(se_fw**2))/len(se_fw))
    vol_scale=pgrid_fw/pvol_adj if pvol_adj>0 else float("nan")
    tvol_adj=rr["sumLtvol"]/rr["sumLvol"] if rr["sumLvol"]>0 else float("nan")

    rows=[]
    print("\nsource-by-source comparison:")
    for i in range(nsrc):
        rows.append(dict(
            source_id=int(sid[i]),x_cm=xyz[i,0],y_cm=xyz[i,1],z_cm=xyz[i,2],
            p_forward=p_fw[i],se_forward=se_fw[i],p_adjoint=p_adj[i],se_adjoint=se_adj[i],
            ratio=ratio[i],z_difference=z[i],forward_mean_t_ns=mean_t_fw[i],adjoint_mean_t_ns=mean_t_adj[i],
            binned_timing_cdf_maxdiff=ks_bin[i],roots_touching_kernel=int(rr["hit_rays"][i]),
            kernel_volume_cm3=volumes[i],kernel_clipped=bool(clipped[i]),
        ))
        print(
            f"  src {int(sid[i]):2d} p_fwd={p_fw[i]:.5f}+/-{se_fw[i]:.5f} "
            f"p_adj={p_adj[i]:.5f}+/-{se_adj[i]:.5f} ratio={ratio[i]:.3f} z={z[i]:+.2f} "
            f"t={mean_t_fw[i]:.2f}/{mean_t_adj[i]:.2f} ns touch={rr['hit_rays'][i]}/{N}"
        )

    terminal_mean={k:float(v)/N for k,v in rr["terminal_weight"].items()}
    bulk_mean=float(rr["bulk_absorbed_weight"])/N
    accounted=bulk_mean+sum(terminal_mean.values())

    out=args.output_prefix
    out.parent.mkdir(parents=True,exist_ok=True)
    csv_path=out.with_name(out.name+"_comparison.csv")
    npz_path=out.with_name(out.name+"_timing.npz")
    json_path=out.with_name(out.name+"_summary.json")
    pd.DataFrame(rows).to_csv(csv_path,index=False)
    np.savez_compressed(npz_path,source_id=sid,source_xyz_cm=xyz,time_edges_ns=np.arange(nb+1)*args.dt_ns,forward_h_per_ns=fw_hist,adjoint_h_per_ns=h_adj)

    summary=dict(
        forward_atlas=str(args.forward_atlas),geometry_id=args.geometry_id,channel=args.channel,
        root_launches_successful=N,root_launches_attempted=int(rr["attempted"]),elapsed_wall_s=elapsed,
        time_budget_min=args.time_budget_min,root_cap=args.reverse_rays,launch_sampler="scrambled Halton 4-D prefix",
        analytic_reciprocity_prefactor_cm2=reciprocity_norm,
        reached_scintillator_fraction=rr["reached_scint"]/N,failed_roots=int(rr["failed"]),
        deterministic_splits_per_root=rr["splits"]/N,analog_fallbacks_per_root=rr["analog_fallbacks"]/N,
        branch_nodes_per_root=rr["nodes_created"]/N,max_live_branches_observed=int(rr["max_live"]),
        boundary_events_per_root=rr["boundary_events"]/N,
        bulk_absorbed_weight_per_root=bulk_mean,terminal_weight_per_root=terminal_mean,
        weight_accounting_sum_per_root=accounted,
        unresolved_weight_per_root=terminal_mean.get("UNRESOLVED_MAX_TIME",0.0)+terminal_mean.get("UNRESOLVED_MAX_STEPS",0.0),
        best_fit_global_scale_forward_over_adjoint=fit_scale,source_map_correlation=corr,source_fractional_rms=rel_rms,
        median_abs_z=float(np.nanmedian(np.abs(z))),max_abs_z=float(np.nanmax(np.abs(z))),
        median_binned_timing_cdf_maxdiff=float(np.nanmedian(ks_bin)),
        whole_volume_adjoint_mean_p=pvol_adj,whole_volume_adjoint_se=se_pvol_adj,
        forward_18point_grid_mean_p=pgrid_fw,forward_18point_grid_se=se_pgrid_fw,
        forward_grid_over_whole_volume_adjoint_scale=vol_scale,whole_volume_adjoint_mean_t_ns=tvol_adj,
        note="Significant Fresnel branches split deterministically; suppressed splits use unbiased analog sampling. Bulk absorption is continuous.",
    )
    json_path.write_text(json.dumps(summary,indent=2))

    print("\naggregate branching-adjoint diagnostic:")
    print(f"  roots successful/attempted         = {N}/{rr['attempted']}  elapsed={elapsed:.1f} s")
    print(f"  roots reaching scintillator        = {rr['reached_scint']}/{N} ({100*rr['reached_scint']/N:.2f}%)")
    print(f"  deterministic splits/root          = {rr['splits']/N:.2f}")
    print(f"  analog fallback decisions/root     = {rr['analog_fallbacks']/N:.2f}")
    print(f"  branch nodes/root                  = {rr['nodes_created']/N:.2f}; max live observed={rr['max_live']}")
    print(f"  boundary events/root               = {rr['boundary_events']/N:.2f}")
    print(f"  unresolved weight/root             = {summary['unresolved_weight_per_root']:.6g}")
    print(f"  weight-accounting sum/root         = {accounted:.6f} (should be ~1)")
    print(f"  best-fit scale forward/adjoint     = {fit_scale:.4f} (target ~1)")
    print(f"  source-map correlation             = {corr:.4f}")
    print(f"  fractional RMS source residual     = {100*rel_rms:.2f}%")
    print(f"  median/max |combined z|            = {np.nanmedian(np.abs(z)):.2f} / {np.nanmax(np.abs(z)):.2f}")
    print(f"  median binned timing CDF maxdiff   = {np.nanmedian(ks_bin):.3f}")
    print("  whole-volume normalization check:")
    print(f"    adjoint volume-mean p            = {pvol_adj:.6f} +/- {se_pvol_adj:.6f}")
    print(f"    forward 18-point grid mean p     = {pgrid_fw:.6f} +/- {se_pgrid_fw:.6f}")
    print(f"    forward-grid / adjoint-volume    = {vol_scale:.4f} (not exact quadrature, but useful)")
    print(f"    adjoint volume mean transport t  = {tvol_adj:.3f} ns")
    print(f"wrote {csv_path}\nwrote {npz_path}\nwrote {json_path}")
    print("Stage-6A WEIGHTED BRANCHING ADJOINT COMPLETE")


if __name__ == "__main__":
    main()
