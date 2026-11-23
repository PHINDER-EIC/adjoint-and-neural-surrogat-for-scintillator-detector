#!/usr/bin/env python3
"""
PHINDER stage 5: canonical fixed-curvature bend laboratory.

Purpose
-------
Measure the optical penalty of a bend independently of the scintillator/source
geometry.  A guide is represented by straight input/output legs joined by a
circular centerline arc of radius R.  The circular sweep is approximated by a
chain of FrustumVolume pieces whose shared cross-sections are perpendicular to
the analytic tangent of the ideal circle.  Increasing --facets therefore gives
a controlled convergence toward a smooth fixed-radius bend without rewriting
Lorenzo's ray/surface engine.

For every bent guide, a *matched segmented straight control* is simulated with
exactly the same number of guide pieces and exactly the same sequence of
centerline segment lengths.  The reported bend transmission is

    T_bend = efficiency(bent) / efficiency(matched straight control),

so bulk absorption from the extra path length and artificial segmentation are
largely factored out.

Photons are injected directly inside the first guide core, using one common
source ensemble for all cases.  The default source is uniform over the entrance
cross-section and isotropic over the forward hemisphere.  This is deliberately
a canonical guide test, not yet the final PHINDER entrance phase space.

Typical first scans
-------------------
# Cheap geometry/transport smoke test:
python phinder_bridge/scripts/phinder_stage5_bend_lab.py \
  --angles 30,90 --radii 1.2 --facets 4,12 --photons 500 \
  --jobs 1 --prefix stage5_smoke

# Angle map at fixed R = 1.2 cm and 16 facets:
time python phinder_bridge/scripts/phinder_stage5_bend_lab.py \
  --angles 15,30,45,60,90 --radii 1.2 --facets 16 \
  --photons 10000 --jobs 10 --prefix stage5_angle_R1p2

# Faceting convergence for a 90-degree bend:
time python phinder_bridge/scripts/phinder_stage5_bend_lab.py \
  --angles 90 --radii 1.2 --facets 2,4,8,16,32 \
  --photons 10000 --jobs 10 --prefix stage5_convergence_90deg

# Radius scan for a 90-degree bend:
time python phinder_bridge/scripts/phinder_stage5_bend_lab.py \
  --angles 90 --radii 0.6,1.2,2.4,4.8 --facets 24 \
  --photons 10000 --jobs 10 --prefix stage5_radius_90deg
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Photon_Simulator import (
    AIR,
    CLAD,
    CORE,
    EJ204,
    BoxVolume,
    FrustumVolume,
    Scene,
    ReadoutPort,
    ApproximateEngine,
    PhotonBatch,
    PhotonStatus,
)
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatchSimulator, ParallelConfig


DEFAULT_GUIDE_WIDTH = 0.60       # cm, core width in the bend laboratory
DEFAULT_GUIDE_THICKNESS = 0.10   # cm, total clad thickness
DEFAULT_CLADDING = 0.001         # cm
DEFAULT_INPUT_LENGTH = 3.0       # cm
DEFAULT_OUTPUT_LENGTH = 3.0      # cm


@dataclass(frozen=True)
class GuidePath:
    points: np.ndarray      # shape (N+1, 2)
    tangents: np.ndarray    # shape (N+1, 2), unit vectors
    kind: str
    angle_deg: float
    radius_cm: float
    facets: int

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.points, axis=0), axis=1)

    @property
    def centerline_length(self) -> float:
        return float(self.segment_lengths.sum())


def _unit2(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError(f"Bad 2D direction {v}")
    return a / n


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated value")
    return vals


def _parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated integer")
    return vals


def make_circular_bend_path(
    angle_deg: float,
    radius_cm: float,
    facets: int,
    input_length: float,
    output_length: float,
) -> GuidePath:
    """Input straight + faceted circular arc + output straight, turning left.

    The ideal circular arc starts tangent to +x.  Cross-section tangents stored
    at the arc nodes are the *analytic* circle tangents, not chord directions.
    This lets each FrustumVolume connect two correctly oriented cross-sections.
    """
    if not (0.0 < angle_deg < 180.0):
        raise ValueError("bend angle must lie in (0, 180) degrees")
    if radius_cm <= 0.0:
        raise ValueError("bend radius must be positive")
    if facets < 1:
        raise ValueError("facets must be >= 1")
    if input_length <= 0.0 or output_length <= 0.0:
        raise ValueError("input/output lengths must be positive")

    theta = math.radians(angle_deg)
    p_start = np.array([0.0, 0.0])
    p_arc0 = np.array([input_length, 0.0])
    center = np.array([input_length, radius_cm])

    points = [p_start, p_arc0]
    tangents = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]

    for j in range(1, facets + 1):
        frac = j / facets
        alpha = -0.5 * math.pi + frac * theta
        p = center + radius_cm * np.array([math.cos(alpha), math.sin(alpha)])
        t = np.array([-math.sin(alpha), math.cos(alpha)])
        points.append(p)
        tangents.append(_unit2(t))

    t_end = tangents[-1]
    p_out = points[-1] + output_length * t_end
    points.append(p_out)
    tangents.append(t_end.copy())

    return GuidePath(
        points=np.asarray(points, dtype=float),
        tangents=np.asarray(tangents, dtype=float),
        kind="bend",
        angle_deg=float(angle_deg),
        radius_cm=float(radius_cm),
        facets=int(facets),
    )


def make_matched_straight_path(bend: GuidePath) -> GuidePath:
    """Straight path with the same segment lengths and number of pieces."""
    lengths = bend.segment_lengths
    x = np.concatenate([[0.0], np.cumsum(lengths)])
    points = np.column_stack([x, np.zeros_like(x)])
    tangents = np.tile(np.array([1.0, 0.0]), (len(points), 1))
    return GuidePath(
        points=points,
        tangents=tangents,
        kind="straight_control",
        angle_deg=bend.angle_deg,
        radius_cm=bend.radius_cm,
        facets=bend.facets,
    )


def cross_section(center_xy, tangent_xy, width: float, thickness: float) -> np.ndarray:
    """Rectangle whose outward +face normal follows tangent_xy.

    Vertex order is chosen so cross((B-A),(C-A)) = +tangent, matching the
    FrustumVolume convention: bottom is reversed and top keeps this order.
    """
    t = _unit2(tangent_xy)
    n = np.array([-t[1], t[0], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    c = np.array([float(center_xy[0]), float(center_xy[1]), 0.0])
    hw = 0.5 * float(width)
    hz = 0.5 * float(thickness)
    return np.asarray([
        c - hw*n - hz*z,
        c + hw*n - hz*z,
        c + hw*n + hz*z,
        c - hw*n + hz*z,
    ])


def _dummy_scintillator() -> BoxVolume:
    # Scene currently requires a semantic volume named "scintillator".  This
    # tiny box is deliberately remote and never participates in the bend test.
    return BoxVolume(
        "scintillator", EJ204,
        [-1001.0, -1001.0, -1001.0],
        [-1000.0, -1000.0, -1000.0],
        priority=0,
    )


def build_guide_scene(
    path: GuidePath,
    guide_width: float,
    guide_thickness: float,
    cladding_thickness: float,
    cladding_model: str = "loss",
) -> Scene:
    """Build the canonical faceted guide.

    cladding_model="loss" (default): track only the core and use CLAD as the
    outside material.  A photon transmitted through the core/clad interface is
    then immediately classified LOST_AT_Guide.  This is the natural and much
    faster definition for a *guided-mode bend-loss* measurement: cladding modes
    are treated as lost and we isolate what fraction of initially guided core
    modes survives the bend.

    cladding_model="full": explicitly track a thin cladding shell and AIR
    outside.  This permits cladding-mode propagation/re-entry and is useful for
    selected validation points, but is substantially slower.
    """
    if guide_width <= 0 or guide_thickness <= 0:
        raise ValueError("guide dimensions must be positive")
    if cladding_thickness < 0:
        raise ValueError("cladding thickness cannot be negative")
    if 2.0 * cladding_thickness >= min(guide_width, guide_thickness):
        raise ValueError("cladding is too thick for the requested guide cross-section")
    if cladding_model not in ("loss", "full"):
        raise ValueError("cladding_model must be 'loss' or 'full'")

    # guide_width/thickness are outer manufactured dimensions.  The optical
    # core is inset by the cladding thickness on all four sides.
    core_width = guide_width - 2.0 * cladding_thickness
    core_thickness = guide_thickness - 2.0 * cladding_thickness

    volumes = [_dummy_scintillator()]
    nseg = len(path.points) - 1
    for i in range(nseg):
        p0, p1 = path.points[i], path.points[i+1]
        t0, t1 = path.tangents[i], path.tangents[i+1]
        core0 = cross_section(p0, t0, core_width, core_thickness)
        core1 = cross_section(p1, t1, core_width, core_thickness)

        if cladding_model == "full":
            outer0 = cross_section(p0, t0, guide_width, guide_thickness)
            outer1 = cross_section(p1, t1, guide_width, guide_thickness)
            volumes.append(FrustumVolume(
                f"guide_clad_{i:03d}", CLAD,
                bottom_vertices=outer0,
                top_vertices=outer1,
                priority=10,
            ))

        core_name = "guide_terminal" if i == nseg - 1 else f"guide_core_{i:03d}"
        volumes.append(FrustumVolume(
            core_name, CORE,
            bottom_vertices=core0,
            top_vertices=core1,
            priority=20,
        ))

    final_t = _unit2(path.tangents[-1])
    port = ReadoutPort(
        channel=0,
        volume_name="guide_terminal",
        point=(float(path.points[-1,0]), float(path.points[-1,1]), 0.0),
        outward_normal=(float(final_t[0]), float(final_t[1]), 0.0),
        label=f"stage5 {path.kind}",
    )
    return Scene(
        volumes=volumes,
        outside_material=(CLAD if cladding_model == "loss" else AIR),
        scintillator_material=EJ204,
        readout_ports=[port],
    )


def make_source_batch(
    n: int,
    seed: int,
    guide_width: float,
    guide_thickness: float,
    cladding_thickness: float,
    source_offset: float,
    min_forward_cos: float,
    source_mode: str = "guided",
) -> PhotonBatch:
    """Common entrance ensemble for all bend/control comparisons.

    source_mode="guided" samples an isotropic forward hemisphere and rejects
    directions that would fail total internal reflection at either transverse
    core/cladding wall in an ideal straight guide.  Thus the bend test starts
    from a population of bona-fide straight-guide modes rather than spending
    CPU on photons that were never guided in the first place.

    source_mode="forward-isotropic" keeps the full forward hemisphere and is
    available as a robustness study.
    """
    if n < 1:
        raise ValueError("--photons must be >= 1")
    if not (0.0 <= min_forward_cos < 1.0):
        raise ValueError("--min-forward-cos must lie in [0,1)")
    if source_mode not in ("guided", "forward-isotropic"):
        raise ValueError("source_mode must be 'guided' or 'forward-isotropic'")

    core_width = guide_width - 2.0 * cladding_thickness
    core_thickness = guide_thickness - 2.0 * cladding_thickness
    if core_width <= 0 or core_thickness <= 0:
        raise ValueError("non-positive core dimensions")

    rng = np.random.default_rng(seed)
    frac = 0.999
    y = rng.uniform(-0.5*core_width*frac, 0.5*core_width*frac, n)
    z = rng.uniform(-0.5*core_thickness*frac, 0.5*core_thickness*frac, n)
    x = np.full(n, float(source_offset))
    points = np.column_stack([x, y, z])

    if source_mode == "forward-isotropic":
        mu = rng.uniform(min_forward_cos, 1.0, n)
        phi = rng.uniform(0.0, 2.0*math.pi, n)
        transverse = np.sqrt(np.maximum(0.0, 1.0 - mu*mu))
        directions = np.column_stack([
            mu,
            transverse*np.cos(phi),
            transverse*np.sin(phi),
        ])
    else:
        # TIR at a transverse wall requires |d·n| < cos(theta_c), with
        # theta_c = asin(n_clad/n_core).  Rejection sampling from the forward
        # isotropic hemisphere therefore gives the isotropic distribution
        # conditioned on being a guided mode in an ideal straight guide.
        crit = math.sqrt(1.0 - (CLAD.refractive_index/CORE.refractive_index)**2)
        accepted = []
        need = n
        while need > 0:
            m = max(1024, 4*need)
            mu = rng.uniform(min_forward_cos, 1.0, m)
            phi = rng.uniform(0.0, 2.0*math.pi, m)
            tr = np.sqrt(np.maximum(0.0, 1.0 - mu*mu))
            cand = np.column_stack([mu, tr*np.cos(phi), tr*np.sin(phi)])
            keep = (np.abs(cand[:,1]) < crit) & (np.abs(cand[:,2]) < crit)
            if np.any(keep):
                take = cand[keep][:need]
                accepted.append(take)
                need -= len(take)
        directions = np.vstack(accepted)

    statuses = np.full(n, int(PhotonStatus.RUNNING_INSIDE), dtype=np.int8)
    times = np.zeros(n, dtype=float)
    return PhotonBatch(points, directions, times, statuses)


def _status_counts(batch: PhotonBatch) -> dict[str, int]:
    vals, counts = np.unique(np.asarray(batch.statuses, dtype=int), return_counts=True)
    out: dict[str, int] = {}
    for v, c in zip(vals, counts):
        try:
            name = PhotonStatus(int(v)).name
        except Exception:
            name = f"UNKNOWN_{int(v)}"
        out[name] = int(c)
    return out


def run_case(
    path: GuidePath,
    source: PhotonBatch,
    guide_width: float,
    guide_thickness: float,
    cladding_thickness: float,
    jobs: int,
    transport_seed: int,
    cladding_model: str,
):
    scene = build_guide_scene(path, guide_width, guide_thickness, cladding_thickness, cladding_model)

    # The current simulator's multiprocessing workers seed from process IDs.
    # For jobs=1 the following makes the transport stochasticity reproducible.
    # With jobs>1 the injected phase-space ensemble is still exactly fixed, but
    # the detailed Fresnel/absorption random sequence is not bitwise repeatable.
    if jobs == 1:
        np.random.seed(int(transport_seed))

    engine = ApproximateEngine(scene=scene)
    simulator = PhotonBatchSimulator(engine, parallel_config=ParallelConfig(n_workers=jobs))
    end = simulator.run(source, parallel=(jobs != 1), show_progress=False)

    caught = np.asarray(end.statuses, dtype=int) == int(PhotonStatus.CAUGHT)
    n = source.size
    nc = int(caught.sum())
    eff = nc / n
    eff_err = math.sqrt(eff * (1.0 - eff) / n) if n else float("nan")
    times = np.asarray(end.times, dtype=float)[caught]
    result = {
        "generated": n,
        "caught": nc,
        "efficiency": eff,
        "efficiency_err": eff_err,
        "mean_time_ns": float(np.mean(times)) if len(times) else float("nan"),
        "rms_time_ns": float(np.std(times)) if len(times) else float("nan"),
        "median_time_ns": float(np.median(times)) if len(times) else float("nan"),
        "p90_time_ns": float(np.quantile(times, 0.90)) if len(times) else float("nan"),
    }
    return result, _status_counts(end), end


def validate_path(path: GuidePath, guide_width: float) -> None:
    if len(path.points) != len(path.tangents):
        raise ValueError("points/tangents length mismatch")
    if len(path.points) < 3:
        raise ValueError("path too short")
    if np.any(path.segment_lengths <= 1e-10):
        raise ValueError("zero-length path segment")
    norms = np.linalg.norm(path.tangents, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("non-unit tangent")
    # A fixed-radius centerline smaller than half the guide width would imply a
    # non-positive inner bend radius and is not a sensible swept guide.
    if path.kind == "bend" and path.radius_cm <= 0.5 * guide_width:
        raise ValueError(
            f"R={path.radius_cm:g} cm is <= guide_width/2={0.5*guide_width:g} cm; "
            "choose a larger bend radius"
        )


def _geometry_rows(path: GuidePath) -> list[dict]:
    rows = []
    for i, (p, t) in enumerate(zip(path.points, path.tangents)):
        rows.append({
            "kind": path.kind,
            "angle_deg": path.angle_deg,
            "radius_cm": path.radius_cm,
            "facets": path.facets,
            "node": i,
            "x_cm": p[0],
            "y_cm": p[1],
            "tx": t[0],
            "ty": t[1],
        })
    return rows


def _plot_geometry(ax, path: GuidePath, width: float, label: str):
    ax.plot(path.points[:,0], path.points[:,1], "o-", markersize=2.5, linewidth=1.0, label=label)
    # Draw a few cross-section width markers to make curvature/faceting visible.
    stride = max(1, (len(path.points)-1)//8)
    for i in range(0, len(path.points), stride):
        p = path.points[i]
        t = path.tangents[i]
        n = np.array([-t[1], t[0]])
        a = p - 0.5*width*n
        b = p + 0.5*width*n
        ax.plot([a[0],b[0]],[a[1],b[1]], linewidth=0.5, alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.grid(alpha=0.2)


def make_pdf(prefix: Path, results: pd.DataFrame, geometry: dict[tuple, GuidePath], guide_width: float):
    pdf_path = prefix.with_name(prefix.name + "_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        # Representative geometries: lowest and highest bend angles/radii/facets.
        bend_paths = [p for p in geometry.values() if p.kind == "bend"]
        picks = []
        if bend_paths:
            picks.append(bend_paths[0])
            if len(bend_paths) > 1:
                picks.append(bend_paths[-1])
        fig, ax = plt.subplots(figsize=(8.5,6.0))
        for p in picks:
            _plot_geometry(ax, p, guide_width,
                           f"bend {p.angle_deg:g}°, R={p.radius_cm:g} cm, N={p.facets}")
        if picks:
            ax.legend()
        ax.set_title("Stage 5 bend laboratory: faceted fixed-radius centerlines")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Bend transmission vs angle, grouped by R and facets.
        fig, ax = plt.subplots(figsize=(8.5,5.5))
        for (R,N), grp in results.groupby(["radius_cm","facets"], sort=True):
            g = grp.sort_values("angle_deg")
            ax.errorbar(g.angle_deg, g.bend_transmission, yerr=g.bend_transmission_err,
                        marker="o", linewidth=1.0, capsize=2, label=f"R={R:g} cm, N={int(N)}")
        ax.axhline(1.0, linewidth=0.8, linestyle="--")
        ax.set_xlabel("bend angle [deg]")
        ax.set_ylabel("bend transmission / matched straight")
        ax.set_title("Excess optical loss from the bend")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Absolute efficiencies: bend and matched control.
        fig, ax = plt.subplots(figsize=(8.5,5.5))
        for (R,N), grp in results.groupby(["radius_cm","facets"], sort=True):
            g = grp.sort_values("angle_deg")
            ax.plot(g.angle_deg, 100*g.bend_efficiency, "o-", linewidth=1.0,
                    label=f"bend R={R:g}, N={int(N)}")
            ax.plot(g.angle_deg, 100*g.control_efficiency, "x--", linewidth=0.8,
                    label=f"straight control R={R:g}, N={int(N)}")
        ax.set_xlabel("bend angle [deg]")
        ax.set_ylabel("caught fraction [%]")
        ax.set_title("Absolute collection from common injected phase space")
        ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Timing penalty.
        fig, ax = plt.subplots(figsize=(8.5,5.5))
        for (R,N), grp in results.groupby(["radius_cm","facets"], sort=True):
            g = grp.sort_values("angle_deg")
            ax.plot(g.angle_deg, g.excess_mean_time_ns, "o-", linewidth=1.0,
                    label=f"R={R:g} cm, N={int(N)}")
        ax.axhline(0.0, linewidth=0.8, linestyle="--")
        ax.set_xlabel("bend angle [deg]")
        ax.set_ylabel("mean caught-photon time: bend - control [ns]")
        ax.set_title("Timing cost of the bend")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Faceting convergence where multiple N exist for same (angle,R).
        fig, ax = plt.subplots(figsize=(8.5,5.5))
        ncurves = 0
        for (A,R), grp in results.groupby(["angle_deg","radius_cm"], sort=True):
            if grp.facets.nunique() < 2:
                continue
            g = grp.sort_values("facets")
            ax.errorbar(g.facets, g.bend_transmission, yerr=g.bend_transmission_err,
                        marker="o", linewidth=1.0, capsize=2,
                        label=f"{A:g}°, R={R:g} cm")
            ncurves += 1
        if ncurves:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("number of arc facets")
            ax.set_ylabel("bend transmission / matched straight")
            ax.set_title("Faceting convergence toward a smooth fixed-radius bend")
            ax.grid(alpha=0.25); ax.legend(fontsize=8)
        else:
            ax.text(0.5,0.5,"Run multiple --facets values to populate convergence scan",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Radius dependence where multiple radii exist for same (angle,N).
        fig, ax = plt.subplots(figsize=(8.5,5.5))
        ncurves = 0
        for (A,N), grp in results.groupby(["angle_deg","facets"], sort=True):
            if grp.radius_cm.nunique() < 2:
                continue
            g = grp.sort_values("radius_cm")
            ax.errorbar(g.radius_cm/guide_width, g.bend_transmission,
                        yerr=g.bend_transmission_err, marker="o", linewidth=1.0,
                        capsize=2, label=f"{A:g}°, N={int(N)}")
            ncurves += 1
        if ncurves:
            ax.set_xlabel("R / guide width")
            ax.set_ylabel("bend transmission / matched straight")
            ax.set_title("Bend loss versus curvature radius")
            ax.grid(alpha=0.25); ax.legend(fontsize=8)
        else:
            ax.text(0.5,0.5,"Run multiple --radii values to populate radius scan",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return pdf_path


def write_summary(prefix: Path, args, results: pd.DataFrame) -> Path:
    path = prefix.with_name(prefix.name + "_summary.txt")
    with path.open("w", encoding="utf-8") as f:
        f.write("PHINDER stage 5: fixed-curvature bend laboratory\n")
        f.write(f"guide outer width x thickness = {args.guide_width:g} x {args.guide_thickness:g} cm\n")
        f.write(f"cladding thickness = {args.cladding_thickness:g} cm on all guide sides\n")
        f.write(f"input/output straight lengths = {args.input_length:g} / {args.output_length:g} cm\n")
        f.write(f"injected photons per geometry = {args.photons}\n")
        f.write(f"source offset = {args.source_offset:g} cm inside input guide\n")
        f.write(f"source mode = {args.source_mode}; min cos(theta) = {args.min_forward_cos:g}\n")
        f.write(f"cladding transport model = {args.cladding_model}\n")
        f.write(f"source seed = {args.seed}; photon-tracking workers = {args.jobs}\n")
        if args.jobs != 1:
            f.write("NOTE: current simulator seeds multiprocessing workers from PIDs; source ensemble is fixed,\n")
            f.write("      but Fresnel/absorption stochasticity is not bitwise reproducible for jobs>1.\n")
        f.write("normalization = matched segmented straight guide with identical segment lengths\n")
        f.write("\nresults:\n")
        for _, r in results.sort_values(["radius_cm","angle_deg","facets"]).iterrows():
            f.write(
                f"  angle={r.angle_deg:g} deg  R={r.radius_cm:g} cm (R/w={r.radius_over_width:.3g}) "
                f"N={int(r.facets):d}: bend={100*r.bend_efficiency:.3f}% "
                f"control={100*r.control_efficiency:.3f}%  "
                f"T_bend={r.bend_transmission:.5f} +/- {r.bend_transmission_err:.5f}  "
                f"dt={r.excess_mean_time_ns:+.4f} ns\n"
            )
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default="15,30,45,60,90",
                    help="comma-separated bend angles in degrees")
    ap.add_argument("--radii", default="1.2",
                    help="comma-separated centerline bend radii in cm")
    ap.add_argument("--facets", default="16",
                    help="comma-separated numbers of facets used for the circular arc")
    ap.add_argument("--photons", type=int, default=10000,
                    help="directly injected photons per bent/control geometry")
    ap.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=24681357,
                    help="seed for the common injected photon phase-space ensemble")
    ap.add_argument("--transport-seed", type=int, default=97531,
                    help="transport RNG seed when --jobs 1")
    ap.add_argument("--guide-width", type=float, default=DEFAULT_GUIDE_WIDTH)
    ap.add_argument("--guide-thickness", type=float, default=DEFAULT_GUIDE_THICKNESS)
    ap.add_argument("--cladding-thickness", type=float, default=DEFAULT_CLADDING)
    ap.add_argument("--input-length", type=float, default=DEFAULT_INPUT_LENGTH)
    ap.add_argument("--output-length", type=float, default=DEFAULT_OUTPUT_LENGTH)
    ap.add_argument("--source-offset", type=float, default=0.10)
    ap.add_argument("--min-forward-cos", type=float, default=0.0)
    ap.add_argument("--source-mode", choices=("guided","forward-isotropic"), default="guided")
    ap.add_argument("--cladding-model", choices=("loss","full"), default="loss",
                    help="loss: core->clad transmission is terminal loss (fast); full: explicitly track cladding shell")
    ap.add_argument("--output-dir", type=Path, default=Path("phinder_bridge/data"))
    ap.add_argument("--prefix", default="stage5_bend")
    args = ap.parse_args()

    angles = _parse_float_list(args.angles)
    radii = _parse_float_list(args.radii)
    facets_list = _parse_int_list(args.facets)
    if args.source_offset <= 0.0 or args.source_offset >= args.input_length:
        raise ValueError("--source-offset must lie strictly inside the input straight leg")

    source = make_source_batch(
        args.photons, args.seed, args.guide_width, args.guide_thickness,
        args.cladding_thickness, args.source_offset, args.min_forward_cos, args.source_mode,
    )

    rows = []
    status_rows = []
    geometry_rows = []
    geometry: dict[tuple, GuidePath] = {}
    case_index = 0

    for radius in radii:
        for angle in angles:
            for facets in facets_list:
                bend = make_circular_bend_path(
                    angle, radius, facets, args.input_length, args.output_length
                )
                validate_path(bend, args.guide_width)
                control = make_matched_straight_path(bend)
                validate_path(control, args.guide_width)

                key = (float(angle), float(radius), int(facets))
                geometry[key] = bend
                geometry_rows.extend(_geometry_rows(bend))

                print(
                    f"[case {case_index+1}] angle={angle:g} deg R={radius:g} cm "
                    f"facets={facets} segments={len(bend.points)-1} "
                    f"L={bend.centerline_length:.5f} cm"
                )

                ctrl_result, ctrl_status, _ = run_case(
                    control, source, args.guide_width, args.guide_thickness,
                    args.cladding_thickness, args.jobs,
                    args.transport_seed + 2*case_index, args.cladding_model,
                )
                bend_result, bend_status, _ = run_case(
                    bend, source, args.guide_width, args.guide_thickness,
                    args.cladding_thickness, args.jobs,
                    args.transport_seed + 2*case_index + 1, args.cladding_model,
                )

                eb = bend_result["efficiency"]
                es = ctrl_result["efficiency"]
                sb = bend_result["efficiency_err"]
                ss = ctrl_result["efficiency_err"]
                ratio = eb / es if es > 0 else float("nan")
                if eb > 0 and es > 0:
                    ratio_err = ratio * math.sqrt((sb/eb)**2 + (ss/es)**2)
                else:
                    ratio_err = float("nan")

                rows.append({
                    "angle_deg": angle,
                    "radius_cm": radius,
                    "radius_over_width": radius / args.guide_width,
                    "facets": facets,
                    "n_segments": len(bend.points)-1,
                    "polygonal_centerline_cm": bend.centerline_length,
                    "ideal_centerline_cm": args.input_length + math.radians(angle)*radius + args.output_length,
                    "generated": args.photons,
                    "control_caught": ctrl_result["caught"],
                    "control_efficiency": es,
                    "control_efficiency_err": ss,
                    "control_mean_time_ns": ctrl_result["mean_time_ns"],
                    "control_rms_time_ns": ctrl_result["rms_time_ns"],
                    "bend_caught": bend_result["caught"],
                    "bend_efficiency": eb,
                    "bend_efficiency_err": sb,
                    "bend_mean_time_ns": bend_result["mean_time_ns"],
                    "bend_rms_time_ns": bend_result["rms_time_ns"],
                    "bend_median_time_ns": bend_result["median_time_ns"],
                    "bend_p90_time_ns": bend_result["p90_time_ns"],
                    "bend_transmission": ratio,
                    "bend_transmission_err": ratio_err,
                    "excess_mean_time_ns": bend_result["mean_time_ns"] - ctrl_result["mean_time_ns"],
                })

                for kind, table in (("control",ctrl_status),("bend",bend_status)):
                    for status, count in sorted(table.items()):
                        status_rows.append({
                            "angle_deg": angle, "radius_cm": radius, "facets": facets,
                            "kind": kind, "status": status, "count": count,
                            "fraction": count/args.photons,
                        })

                print(
                    f"    control {ctrl_result['caught']}/{args.photons} = {100*es:.3f}%\n"
                    f"    bend    {bend_result['caught']}/{args.photons} = {100*eb:.3f}%\n"
                    f"    T_bend = {ratio:.5f} +/- {ratio_err:.5f}; "
                    f"delta<t> = {bend_result['mean_time_ns']-ctrl_result['mean_time_ns']:+.4f} ns"
                )
                case_index += 1

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / args.prefix
    results = pd.DataFrame(rows)
    statuses = pd.DataFrame(status_rows)
    geom_df = pd.DataFrame(geometry_rows)

    results_path = prefix.with_name(prefix.name + "_results.csv")
    status_path = prefix.with_name(prefix.name + "_status_counts.csv")
    geom_path = prefix.with_name(prefix.name + "_geometry.csv")
    results.to_csv(results_path, index=False)
    statuses.to_csv(status_path, index=False)
    geom_df.to_csv(geom_path, index=False)
    summary_path = write_summary(prefix, args, results)
    pdf_path = make_pdf(prefix, results, geometry, args.guide_width)

    print("\nPHINDER stage-5 bend-lab outputs")
    for p in (results_path, status_path, geom_path, summary_path, pdf_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()
