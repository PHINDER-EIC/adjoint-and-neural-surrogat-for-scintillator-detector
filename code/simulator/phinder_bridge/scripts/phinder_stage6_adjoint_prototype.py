#!/usr/bin/env python3
"""
PHINDER Stage-6A: reciprocal/adjoint optical-transport prototype.

This is a validation prototype, not yet the production truth generator.
It reverses the usual simulation:

  readout terminal -> passive optical transport -> scintillator volume

and uses a track-length estimator in small boxes around the source points of an
existing forward Stage-6 atlas.  For a reciprocal passive optical system, a
reverse ray population launched uniformly over the terminal CORE face with a
cosine-weighted inward angular distribution estimates the adjoint scalar flux.
For an isotropic point source in scintillator, the corresponding forward
readout probability is

    p_c(r) = (A_port/4) * (n_port/n_scint)^2 * phi_adj(r),

where phi_adj is the reverse track-length density per launched ray.  We estimate
phi_adj by the total reverse track length inside a small source-centred kernel
box divided by kernel volume and number of reverse rays.

The same track-length contributions are accumulated versus reverse flight time,
which is the forward optical transport time of the reciprocal path.

The decisive intended test is against the existing high-statistics forward
atlas stage6_map_hi_atlas.h5 at w=0.6 cm, R=1.2 cm.

Caveats
-------
* A finite spatial kernel estimates a local spatial average, not an exact point.
  The kernel can be reduced after the method is shown to work.
* The analytic n^2 normalization follows optical etendue/radiance reciprocity.
  The script ALSO reports a best-fit global scale.  A scale far from one is a
  red flag that the adjoint measure/normalization is wrong.
* The prototype assumes reciprocal, passive transport.  Wavelength shifting or
  other non-reciprocal processes would need an extended adjoint treatment.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from Photon_Simulator import (
    ApproximateEngine, PhotonStatus, CORE, EJ204, AIR, CLAD,
    BoxVolume, FrustumVolume, Scene, ReadoutPort,
)
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonEvent


# -----------------------------------------------------------------------------
# Worker globals
# -----------------------------------------------------------------------------
_W = {}


def _is_running(status) -> bool:
    return int(status) in (int(PhotonStatus.RUNNING_INSIDE), int(PhotonStatus.RUNNING_TRANSMITTED))


def _segment_box_interval(p0: np.ndarray, p1: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Return parametric [a,b] subset of segment p(a)=p0+a*(p1-p0) inside AABB."""
    d = p1 - p0
    a, b = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-15:
            if p0[k] < lo[k] or p0[k] > hi[k]:
                return None
            continue
        u = (lo[k] - p0[k]) / d[k]
        v = (hi[k] - p0[k]) / d[k]
        mn, mx = (u, v) if u <= v else (v, u)
        a = max(a, mn)
        b = min(b, mx)
        if b <= a:
            return None
    a = max(0.0, a)
    b = min(1.0, b)
    if b <= a:
        return None
    return a, b


def _add_time_interval(hist: np.ndarray, t0: float, t1: float, length: float, dt: float, tmax: float):
    """Distribute a track-length contribution exactly over uniform time bins."""
    if length <= 0.0 or t1 <= t0 or t1 <= 0.0 or t0 >= tmax:
        return
    aa = max(0.0, t0)
    bb = min(tmax, t1)
    if bb <= aa:
        return
    rate = length / (t1 - t0)  # ds/dt along this straight segment
    i0 = max(0, int(math.floor(aa / dt)))
    i1 = min(len(hist) - 1, int(math.floor(np.nextafter(bb, -np.inf) / dt)))
    for ib in range(i0, i1 + 1):
        left = max(aa, ib * dt)
        right = min(bb, (ib + 1) * dt)
        if right > left:
            hist[ib] += rate * (right - left)


def _cross_section_z(center_xy, tangent_xy, width: float, thickness: float, z_center: float):
    t = np.asarray(tangent_xy, float)
    t /= np.linalg.norm(t)
    n = np.array([-t[1], t[0], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    c = np.array([float(center_xy[0]), float(center_xy[1]), float(z_center)])
    hw = 0.5 * float(width); hz = 0.5 * float(thickness)
    return np.asarray([c-hw*n-hz*z, c+hw*n-hz*z, c+hw*n+hz*z, c-hw*n+hz*z])


def _build_from_cfg(cfg: dict):
    """Rebuild the exact forward-atlas scene from the stored path nodes/tangents."""
    X, Y, Z = map(float, cfg["cell_cm"])
    width = float(cfg["width_cm"])
    thickness = float(cfg["guide_thickness_cm"])
    clad = float(cfg["cladding_thickness_cm"])
    core_w = width - 2.0 * clad
    core_h = thickness - 2.0 * clad
    volumes = [BoxVolume("scintillator", EJ204, [0,0,0], [X,Y,Z], priority=0)]
    ports = []
    paths = []
    for ch in (0,1):
        pts = np.asarray(cfg[f"path{ch}_points"], float)
        tans = np.asarray(cfg[f"path{ch}_tangents"], float)
        zc = Z + 0.5*thickness if ch == 0 else -0.5*thickness
        nseg = len(pts)-1
        for i in range(nseg):
            outer0=_cross_section_z(pts[i],tans[i],width,thickness,zc)
            outer1=_cross_section_z(pts[i+1],tans[i+1],width,thickness,zc)
            core0=_cross_section_z(pts[i],tans[i],core_w,core_h,zc)
            core1=_cross_section_z(pts[i+1],tans[i+1],core_w,core_h,zc)
            volumes.append(FrustumVolume(f"guide{ch}_clad_{i:03d}",CLAD,bottom_vertices=outer0,top_vertices=outer1,priority=10))
            name=f"guide{ch}_terminal" if i==nseg-1 else f"guide{ch}_core_{i:03d}"
            volumes.append(FrustumVolume(name,CORE,bottom_vertices=core0,top_vertices=core1,priority=20))
        tf=tans[-1]/np.linalg.norm(tans[-1])
        ports.append(ReadoutPort(channel=ch,volume_name=f"guide{ch}_terminal",point=(float(pts[-1,0]),float(pts[-1,1]),float(zc)),outward_normal=(float(tf[0]),float(tf[1]),0.0),label=f"stage6A adjoint ch{ch}"))
        paths.append((pts,tans))
    scene=Scene(volumes=volumes,outside_material=AIR,scintillator_material=EJ204,readout_ports=ports)
    return scene, paths, core_w, core_h


def _worker_init(cfg: dict, source_xyz: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray):
    scene, paths, core_w, core_h = _build_from_cfg(cfg)
    pts, tans = paths[cfg["channel"]]
    Z = float(cfg["cell_cm"][2]); thickness=float(cfg["guide_thickness_cm"])
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
        engine=ApproximateEngine(scene),
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


def _initial_reverse_ray(ray_id: int):
    cfg = _W["cfg"]
    # Per-ray generator makes launch phase space independent of process/chunk scheduling.
    ss = np.random.SeedSequence([int(cfg["launch_seed"]), int(ray_id)])
    rng = np.random.default_rng(ss)

    # Uniform terminal CORE area.  Stay microscopically away from the rectangle edge.
    edge_frac = 1.0 - 2.0e-10
    a = rng.uniform(-0.5, 0.5) * _W["core_w"] * edge_frac
    b = rng.uniform(-0.5, 0.5) * _W["core_h"] * edge_frac
    start = _W["endpoint"] + a * _W["lateral"] + b * _W["vertical"]
    start = start - float(cfg["start_epsilon_cm"]) * _W["outward"]

    # Cosine/Lambertian inward hemisphere: q(Omega)=cos(theta)/pi.
    mu = math.sqrt(float(rng.random()))
    st = math.sqrt(max(0.0, 1.0 - mu * mu))
    phi = 2.0 * math.pi * float(rng.random())
    direction = (
        mu * _W["inward"]
        + st * math.cos(phi) * _W["lateral"]
        + st * math.sin(phi) * _W["vertical"]
    )
    direction /= np.linalg.norm(direction)
    return start, direction


def _score_scint_segment(p0, p1, t0, t1, Lvec, Ltvec, hist):
    seg = p1 - p0
    full_L = float(np.linalg.norm(seg))
    if full_L <= 0.0:
        return
    cfg = _W["cfg"]
    for isrc, (lo, hi) in enumerate(zip(_W["box_lo"], _W["box_hi"])):
        ab = _segment_box_interval(p0, p1, lo, hi)
        if ab is None:
            continue
        a, b = ab
        ell = full_L * (b - a)
        if ell <= 0.0:
            continue
        ta = t0 + a * (t1 - t0)
        tb = t0 + b * (t1 - t0)
        tm = 0.5 * (ta + tb)
        Lvec[isrc] += ell
        Ltvec[isrc] += ell * tm
        _add_time_interval(hist[isrc], ta, tb, ell, cfg["dt_ns"], cfg["tmax_ns"])


def _run_chunk(bounds):
    start_id, stop_id = bounds
    cfg = _W["cfg"]
    nsrc = len(_W["source_xyz"])
    nb = int(math.ceil(cfg["tmax_ns"] / cfg["dt_ns"]))

    sumL = np.zeros(nsrc, float)
    sumL2 = np.zeros(nsrc, float)
    sumLt = np.zeros(nsrc, float)
    hist = np.zeros((nsrc, nb), float)
    hit_rays = np.zeros(nsrc, np.int64)
    status_counts = Counter()
    n_reached_scint = 0
    n_failed = 0
    n_steps_total = 0
    n_truncated = 0

    for ray_id in range(start_id, stop_id):
        Lvec = np.zeros(nsrc, float)
        Ltvec = np.zeros(nsrc, float)
        hray = np.zeros((nsrc, nb), float)
        reached = False
        try:
            p, d = _initial_reverse_ray(ray_id)
            loc = _W["scene"].locate_volume(p)
            if loc is None or loc.name != f"guide{cfg['channel']}_terminal":
                raise RuntimeError(f"reverse start not inside terminal core: {None if loc is None else loc.name}")

            # Engine uses numpy's global RNG for absorption/Fresnel choices.
            phys_seed = (int(cfg["physics_seed"]) + 1000003 * int(ray_id)) & 0xFFFFFFFF
            np.random.seed(phys_seed)
            state = PhotonEvent(p, d, PhotonStatus.RUNNING_INSIDE, 0.0)
            steps = 0
            truncated = False
            while _is_running(state.status):
                prev = state
                state = _W["engine"]._next_event(state)
                steps += 1

                p0 = np.asarray(prev.interaction_point, float)
                p1 = np.asarray(state.interaction_point, float)
                segL = float(np.linalg.norm(p1 - p0))
                if segL > 0.0:
                    # Do not call scene.locate_volume here: _next_event already did the
                    # expensive geometry work.  The material traversed by this segment
                    # is encoded exactly in its flight time dt = L*n/c.
                    dtseg = float(state.time) - float(prev.time)
                    n_eff = dtseg * float(_W["engine"].config.c_cm_ns) / segL
                    if abs(n_eff - float(EJ204.refractive_index)) < 1e-7:
                        reached = True
                        _score_scint_segment(p0, p1, float(prev.time), float(state.time), Lvec, Ltvec, hray)
                if float(state.time) >= float(cfg["max_flight_time_ns"]) or steps >= int(cfg["max_steps"]):
                    truncated = True
                    break
            n_steps_total += steps
            if truncated:
                n_truncated += 1
                status_counts[998] += 1
            else:
                status_counts[int(state.status)] += 1
            if reached:
                n_reached_scint += 1
        except Exception:
            n_failed += 1
            status_counts[999] += 1

        sumL += Lvec
        sumL2 += Lvec * Lvec
        sumLt += Ltvec
        hist += hray
        hit_rays += (Lvec > 0.0)

    return dict(
        n=stop_id - start_id,
        sumL=sumL,
        sumL2=sumL2,
        sumLt=sumLt,
        hist=hist,
        hit_rays=hit_rays,
        reached_scint=n_reached_scint,
        failed=n_failed,
        truncated=n_truncated,
        steps=n_steps_total,
        statuses=dict(status_counts),
    )


def _forward_truth(atlas_path: Path, geometry_id: str, channel: int, dt: float, tmax: float):
    with h5py.File(atlas_path, "r") as f:
        if geometry_id not in f["geometries"]:
            raise KeyError(f"{geometry_id!r} not in {atlas_path}")
        g = f["geometries"][geometry_id]
        srcgrp = f["sources"]
        sid = np.asarray(srcgrp["source_id"][:], int)
        order = np.argsort(sid)
        sid = sid[order]
        xyz = np.column_stack([srcgrp["x_cm"][:], srcgrp["y_cm"][:], srcgrp["z_cm"][:]])[order]

        psid = np.asarray(g["photons/source_id"][:], int)
        ch = np.asarray(g["photons/channel"][:], int)
        times = np.asarray(g["photons/transport_time_ns"][:], float)
        nph_default = int(f.attrs.get("photons_per_source", 0))

        rows = []
        nb = int(math.ceil(tmax / dt))
        h = np.zeros((len(sid), nb), float)
        for i, s in enumerate(sid):
            m = psid == s
            n = int(np.count_nonzero(m))
            if n == 0 and nph_default:
                n = nph_default
            mc = m & (ch == channel)
            nc = int(np.count_nonzero(mc))
            p = nc / n
            se = math.sqrt(max(p * (1.0 - p) / n, 0.0))
            tt = times[mc]
            mean_t = float(tt.mean()) if len(tt) else float("nan")
            hh, _ = np.histogram(tt, bins=np.arange(nb + 1) * dt)
            h[i] = hh / (n * dt)
            rows.append((n, nc, p, se, mean_t))

        meta = dict(
            width_cm=float(g.attrs["width_cm"]),
            radius_cm=float(g.attrs["radius_cm"]),
            facets=int(g.attrs["facets"]),
            guide_thickness_cm=float(g.attrs.get("guide_thickness_cm", 0.10)),
            cladding_thickness_cm=float(g.attrs.get("cladding_thickness_cm", 0.001)),
            cell_cm=np.array([float(f.attrs.get("cell_x_cm",5.0)), float(f.attrs.get("cell_y_cm",5.0)), float(f.attrs.get("cell_z_cm",1.0))]),
            path0_points=np.asarray(g["guide0_points_cm"][:], float),
            path0_tangents=np.asarray(g["guide0_tangents"][:], float),
            path1_points=np.asarray(g["guide1_points_cm"][:], float),
            path1_tangents=np.asarray(g["guide1_tangents"][:], float),
        )
    return sid, xyz, np.asarray(rows, float), h, meta


def _merge_results(results, nsrc, nb):
    out = dict(
        n=0,
        sumL=np.zeros(nsrc),
        sumL2=np.zeros(nsrc),
        sumLt=np.zeros(nsrc),
        hist=np.zeros((nsrc, nb)),
        hit_rays=np.zeros(nsrc, dtype=np.int64),
        reached_scint=0,
        failed=0,
        truncated=0,
        steps=0,
        statuses=Counter(),
    )
    for r in results:
        out["n"] += int(r["n"])
        for k in ("sumL", "sumL2", "sumLt", "hist", "hit_rays"):
            out[k] += r[k]
        out["reached_scint"] += int(r["reached_scint"])
        out["failed"] += int(r["failed"])
        out["truncated"] += int(r.get("truncated",0))
        out["steps"] += int(r["steps"])
        out["statuses"].update({int(k): int(v) for k, v in r["statuses"].items()})
    return out


def _weighted_scale(x, y, sy):
    """Fit y ~= a*x with uncertainty weights from forward+adjoint errors."""
    var = np.maximum(sy * sy, 1e-16)
    w = 1.0 / var
    den = float(np.sum(w * x * x))
    return float(np.sum(w * x * y) / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="PHINDER Stage-6A adjoint/reciprocity prototype")
    ap.add_argument("--forward-atlas", type=Path, default=Path("phinder_bridge/data/stage6_map_hi_atlas.h5"))
    ap.add_argument("--geometry-id", default="g000")
    ap.add_argument("--channel", type=int, choices=(0, 1), default=0)
    ap.add_argument("--reverse-rays", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--kernel-xy-cm", type=float, default=0.30,
                    help="full x/y width of local track-length scoring box")
    ap.add_argument("--kernel-z-cm", type=float, default=0.10,
                    help="full z width of local track-length scoring box")
    ap.add_argument("--dt-ns", type=float, default=0.25)
    ap.add_argument("--tmax-ns", type=float, default=40.0)
    ap.add_argument("--launch-seed", type=int, default=606061)
    ap.add_argument("--physics-seed", type=int, default=707071)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--max-flight-time-ns", type=float, default=60.0,
                    help="terminate reverse rays after this optical flight time; 60 ns exceeds the existing forward atlas tail")
    ap.add_argument("--start-epsilon-cm", type=float, default=1e-7)
    ap.add_argument("--output-prefix", type=Path, default=Path("phinder_bridge/data/stage6_adjoint_proto_ch0"))
    args = ap.parse_args()

    if args.reverse_rays < 1 or args.jobs < 1 or args.chunk_size < 1:
        raise ValueError("reverse-rays/jobs/chunk-size must be positive")
    if args.kernel_xy_cm <= 0 or args.kernel_z_cm <= 0 or args.dt_ns <= 0 or args.tmax_ns <= 0:
        raise ValueError("kernel/time parameters must be positive")

    sid, xyz, fw, fw_hist, meta = _forward_truth(args.forward_atlas, args.geometry_id, args.channel, args.dt_ns, args.tmax_ns)
    nsrc = len(sid)
    nb = fw_hist.shape[1]

    # Rebuild directly from the exact guide nodes/tangents stored in the forward atlas.
    cfg = dict(
        geometry_id=args.geometry_id,
        width_cm=meta["width_cm"],
        radius_cm=meta["radius_cm"],
        facets=meta["facets"],
        guide_thickness_cm=meta["guide_thickness_cm"],
        cladding_thickness_cm=meta["cladding_thickness_cm"],
        cell_cm=np.asarray(meta["cell_cm"],float),
        path0_points=meta["path0_points"], path0_tangents=meta["path0_tangents"],
        path1_points=meta["path1_points"], path1_tangents=meta["path1_tangents"],
        channel=args.channel,
        launch_seed=args.launch_seed,
        physics_seed=args.physics_seed,
        max_steps=args.max_steps,
        max_flight_time_ns=args.max_flight_time_ns,
        start_epsilon_cm=args.start_epsilon_cm,
        dt_ns=args.dt_ns,
        tmax_ns=args.tmax_ns,
    )
    scene, paths, core_w, core_h = _build_from_cfg(cfg)

    half = np.array([0.5 * args.kernel_xy_cm, 0.5 * args.kernel_xy_cm, 0.5 * args.kernel_z_cm])
    cell = np.asarray(meta["cell_cm"], float)
    box_lo = np.maximum(0.0, xyz - half)
    box_hi = np.minimum(cell, xyz + half)
    volumes = np.prod(box_hi - box_lo, axis=1)
    if np.any(volumes <= 0):
        raise RuntimeError("zero source-kernel volume")
    clipped = np.any(np.abs((box_hi - box_lo) - 2 * half) > 1e-12, axis=1)

    area = core_w * core_h
    n_port = float(CORE.refractive_index)
    n_sc = float(EJ204.refractive_index)
    reciprocity_norm = 0.25 * area * (n_port / n_sc) ** 2

    print("PHINDER Stage-6A reciprocal/adjoint prototype")
    print(f"forward atlas: {args.forward_atlas}  geometry={args.geometry_id}  channel={args.channel}")
    print("geometry rebuild: PASS (using exact path nodes/tangents stored in forward atlas)")
    print(f"terminal CORE area={area:.8f} cm^2  n_port/n_sc={n_port/n_sc:.8f}")
    print(f"analytic reciprocity prefactor A/4*(n_port/n_sc)^2 = {reciprocity_norm:.8g} cm^2")
    print(f"source kernels: {args.kernel_xy_cm:g} x {args.kernel_xy_cm:g} x {args.kernel_z_cm:g} cm; clipped={int(clipped.sum())}/{nsrc}")
    print(f"launching {args.reverse_rays} reverse rays with {args.jobs} workers")

    chunks = [(i, min(i + args.chunk_size, args.reverse_rays)) for i in range(0, args.reverse_rays, args.chunk_size)]
    method = "fork" if os.name != "nt" else "spawn"
    ctx = mp.get_context(method)
    if method == "fork":
        # Warm Numba/geometry kernels once in the parent before forking.  The
        # compiled code pages are then inherited by all workers, avoiding an
        # expensive first-ray compilation in every process.  Discard this ray.
        print("warming transport kernels once before fork ...")
        _worker_init(cfg, xyz, box_lo, box_hi)
        _run_chunk((args.reverse_rays + 1000000, args.reverse_rays + 1000001))
    with ctx.Pool(
        processes=args.jobs,
        initializer=_worker_init,
        initargs=(cfg, xyz, box_lo, box_hi),
    ) as pool:
        results = list(pool.imap_unordered(_run_chunk, chunks))
    rr = _merge_results(results, nsrc, nb)

    N = rr["n"]
    meanL = rr["sumL"] / N
    # Sample SE of the mean per-ray track length.
    varL = np.maximum((rr["sumL2"] - rr["sumL"] ** 2 / N) / max(N - 1, 1), 0.0)
    se_meanL = np.sqrt(varL / N)
    p_adj = reciprocity_norm * meanL / volumes
    se_adj = reciprocity_norm * se_meanL / volumes
    mean_t_adj = np.divide(rr["sumLt"], rr["sumL"], out=np.full(nsrc, np.nan), where=rr["sumL"] > 0)
    h_adj = reciprocity_norm * rr["hist"] / (N * volumes[:, None] * args.dt_ns)
    p_adj_tmax = h_adj.sum(axis=1) * args.dt_ns

    n_fw, nc_fw, p_fw, se_fw, mean_t_fw = fw.T
    comb_se = np.sqrt(se_fw ** 2 + se_adj ** 2)
    z = np.divide(p_adj - p_fw, comb_se, out=np.full(nsrc, np.nan), where=comb_se > 0)
    ratio = np.divide(p_adj, p_fw, out=np.full(nsrc, np.nan), where=p_fw > 0)

    # Binned timing-CDF comparison (shape only).
    ks_bin = np.full(nsrc, np.nan)
    for i in range(nsrc):
        a = h_adj[i].copy(); b = fw_hist[i].copy()
        if a.sum() > 0 and b.sum() > 0:
            ca = np.cumsum(a) / np.sum(a)
            cb = np.cumsum(b) / np.sum(b)
            ks_bin[i] = float(np.max(np.abs(ca - cb)))

    # A global scale near 1 is a stringent test of the analytic adjoint normalization.
    scale_unc = np.sqrt(se_fw ** 2 + se_adj ** 2)
    fit_scale = _weighted_scale(p_adj, p_fw, scale_unc)
    corr = float(np.corrcoef(p_adj, p_fw)[0, 1]) if nsrc > 1 else float("nan")
    rel_rms = float(np.sqrt(np.nanmean(((p_adj - p_fw) / np.maximum(p_fw, 1e-15)) ** 2)))

    rows = []
    print("\nsource-by-source comparison:")
    for i in range(nsrc):
        rows.append(dict(
            source_id=int(sid[i]), x_cm=xyz[i,0], y_cm=xyz[i,1], z_cm=xyz[i,2],
            forward_generated=int(n_fw[i]), forward_caught=int(nc_fw[i]),
            p_forward=p_fw[i], se_forward=se_fw[i],
            p_adjoint=p_adj[i], se_adjoint=se_adj[i], p_adjoint_tmax=p_adj_tmax[i],
            adjoint_over_forward=ratio[i], z_difference=z[i],
            forward_mean_t_ns=mean_t_fw[i], adjoint_mean_t_ns=mean_t_adj[i],
            binned_timing_cdf_maxdiff=ks_bin[i],
            reverse_rays_touching_kernel=int(rr["hit_rays"][i]),
            reverse_touch_fraction=rr["hit_rays"][i]/N,
            kernel_volume_cm3=volumes[i], kernel_clipped=bool(clipped[i]),
        ))
        print(
            f"  src {int(sid[i]):2d} r=({xyz[i,0]:.2f},{xyz[i,1]:.2f},{xyz[i,2]:.2f}) "
            f"p_fwd={p_fw[i]:.5f}+/-{se_fw[i]:.5f}  "
            f"p_adj={p_adj[i]:.5f}+/-{se_adj[i]:.5f}  ratio={ratio[i]:.3f}  z={z[i]:+.2f}  "
            f"t={mean_t_fw[i]:.2f}/{mean_t_adj[i]:.2f} ns  touch={rr['hit_rays'][i]}/{N}"
        )

    out = args.output_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_name(out.name + "_comparison.csv")
    npz_path = out.with_name(out.name + "_timing.npz")
    json_path = out.with_name(out.name + "_summary.json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        source_id=sid,
        source_xyz_cm=xyz,
        time_edges_ns=np.arange(nb + 1) * args.dt_ns,
        forward_h_per_ns=fw_hist,
        adjoint_h_per_ns=h_adj,
    )

    status_named = {}
    for k, v in rr["statuses"].items():
        if k == 999:
            name = "EXCEPTION"
        elif k == 998:
            name = "TRUNCATED_AT_MAX_TIME"
        else:
            try:
                name = PhotonStatus(k).name
            except Exception:
                name = str(k)
        status_named[name] = int(v)

    summary = dict(
        forward_atlas=str(args.forward_atlas), geometry_id=args.geometry_id, channel=args.channel,
        reverse_rays=N, jobs=args.jobs,
        kernel_xy_cm=args.kernel_xy_cm, kernel_z_cm=args.kernel_z_cm,
        terminal_core_area_cm2=area, n_port=n_port, n_scint=n_sc,
        analytic_reciprocity_prefactor_cm2=reciprocity_norm,
        reached_scintillator_fraction=rr["reached_scint"] / N,
        failed_fraction=rr["failed"] / N,
        truncated_fraction=rr["truncated"] / N,
        max_flight_time_ns=args.max_flight_time_ns,
        mean_tracking_steps=rr["steps"] / max(N, 1),
        final_status_counts=status_named,
        best_fit_global_scale_forward_over_adjoint=fit_scale,
        source_map_correlation=corr,
        source_fractional_rms=rel_rms,
        median_abs_z=float(np.nanmedian(np.abs(z))),
        max_abs_z=float(np.nanmax(np.abs(z))),
        median_binned_timing_cdf_maxdiff=float(np.nanmedian(ks_bin)),
        note="Adjoint source: uniform terminal CORE area + cosine-weighted inward hemisphere. Track-length kernel estimator.",
    )
    json_path.write_text(json.dumps(summary, indent=2))

    print("\naggregate adjoint diagnostic:")
    print(f"  reverse rays reaching scintillator = {rr['reached_scint']}/{N} ({100*rr['reached_scint']/N:.2f}%)")
    print(f"  reverse tracking failures          = {rr['failed']}/{N} ({100*rr['failed']/N:.3f}%)")
    print(f"  truncated at {args.max_flight_time_ns:g} ns           = {rr['truncated']}/{N} ({100*rr['truncated']/N:.3f}%)")
    print(f"  mean tracking steps/ray           = {rr['steps']/max(N,1):.2f}")
    print(f"  final statuses                    = {status_named}")
    print(f"  best-fit scale forward/adjoint    = {fit_scale:.4f}  (analytic reciprocity wants ~1)")
    print(f"  source-map correlation            = {corr:.4f}")
    print(f"  fractional RMS source residual    = {100*rel_rms:.2f}%")
    print(f"  median/max |combined z|           = {np.nanmedian(np.abs(z)):.2f} / {np.nanmax(np.abs(z)):.2f}")
    print(f"  median binned timing CDF maxdiff  = {np.nanmedian(ks_bin):.3f}")
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {json_path}")

    # We deliberately do not print PASS/FAIL yet.  This is the experiment that
    # determines whether the adjoint construction is correct in the real backend.
    print("Stage-6A ADJOINT PROTOTYPE COMPLETE (inspect scale, spatial agreement, and timing)")


if __name__ == "__main__":
    main()
