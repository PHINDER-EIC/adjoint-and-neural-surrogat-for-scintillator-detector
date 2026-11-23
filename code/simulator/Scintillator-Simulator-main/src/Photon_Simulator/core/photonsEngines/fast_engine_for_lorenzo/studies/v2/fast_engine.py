"""Numba batch propagation for the v2 scene family (scintillator polyhedron +
fiber clad/core boxes, no reflective wrap).

Faithful port of ApproximateEngine._next_event for the volumes/materials that
actually occur in the v2 scenes, with the same stepping order and tolerances:
  1. nearest boundary among all volumes (first hit per volume, t > 1e-9);
     from-material = located (by priority) at the segment midpoint
  2. edge guard on the from-volume (eps 1e-9)          -> ABSORBED_AT_EDGES
  3. bulk absorption, d ~ Exp(mu_from)                 -> ABSORBED
  4. catch test (in fiber_core box, dir_y > 0,
     |y - core_ymax| <= 1e-8)                          -> CAUGHT
  5. unpolarized Fresnel between from/to materials (TIR => R = 1);
     reflect and continue, or refract; refracting to the outside -> LOST
Geometry arrays are EXTRACTED from the real build_scene_v2 Scene object so the
two engines can never disagree on geometry. Only the propagation RNG differs
(counter-seeded xorshift64* per photon instead of global np.random), which is
exactly the seed-to-seed noise the matched-event validation measures against.

The kernel is single-threaded on purpose: event-level ProcessPoolExecutor
parallelism (the existing pattern) composes with it without oversubscription.

Usage:  from fast_engine import run_event_v2_fast   # drop-in for run_event_v2
Validate first: .venv-srso/Scripts/python.exe studies/v2/validate_fast_engine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
import sim_v2  # noqa: E402
from sim_v2 import BETA, DX, LY, build_scene_v2, local_depth  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402
from feature_lab import f_abs, f_early  # noqa: E402
from sim import scaled_scintillator  # noqa: E402
from Photon_Simulator.core.physics.particles import MIP  # noqa: E402
from Photon_Simulator.core.physics.physics import ParticleInteraction  # noqa: E402

C_CM_NS = 29.9792458
EPS_RAY = 1e-9          # min ray-hit distance (engine eps)
EPS_EDGE = 1e-9         # point_on_edge tolerance
EPS_CONTAIN = 1e-9      # contains() tolerance
CATCH_TOL = 1e-8        # TrackingConfig.catch_tolerance
MAX_STEPS = 1_000_000
# status codes (subset of PhotonStatus, same values)
ST_ABSORBED, ST_LOST, ST_CAUGHT, ST_EDGE = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# scene compilation: pull arrays out of the REAL Scene object
# ---------------------------------------------------------------------------
def compile_scene(theta, material):
    scene = build_scene_v2(theta, material)
    poly, clad, core = scene.volumes
    faces = np.array(poly.faces, dtype=np.int64)            # (6, 4)
    return dict(
        face_p=np.ascontiguousarray(poly.face_points),
        face_n=np.ascontiguousarray(poly.face_normals),
        verts=np.ascontiguousarray(poly.vertices),
        faces=faces,
        clad_min=np.ascontiguousarray(clad.box_min),
        clad_max=np.ascontiguousarray(clad.box_max),
        core_min=np.ascontiguousarray(core.box_min),
        core_max=np.ascontiguousarray(core.box_max),
        # volume ids: 0 scint, 1 clad, 2 core, -1 outside(air)
        n_mat=np.array([material.refractive_index,
                        clad.material.refractive_index,
                        core.material.refractive_index, 1.0]),
        mu_mat=np.array([material.absorption_mu,
                         clad.material.absorption_mu,
                         core.material.absorption_mu, 0.0]),
    )


# ---------------------------------------------------------------------------
# RNG: splitmix64 seeding + xorshift64* draws (deterministic per photon)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) \
        & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) \
        & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _rand(state):
    s = state[0]
    s ^= s << np.uint64(13)
    s &= np.uint64(0xFFFFFFFFFFFFFFFF)
    s ^= s >> np.uint64(7)
    s ^= s << np.uint64(17)
    s &= np.uint64(0xFFFFFFFFFFFFFFFF)
    state[0] = s
    return ((s * np.uint64(0x2545F4914F6CDD1D))
            >> np.uint64(11)) * (1.0 / 9007199254740992.0)


# ---------------------------------------------------------------------------
# geometry kernels
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _box_first_hit(bmin, bmax, p, d):
    """First ray hit (t > EPS_RAY) on an axis-aligned box.
    Returns (t, axis, side); t = inf if none."""
    best_t = np.inf
    best_ax = -1
    best_sd = 0
    for ax in range(3):
        if abs(d[ax]) < 1e-12:
            continue
        for sd in range(2):
            plane = bmin[ax] if sd == 0 else bmax[ax]
            t = (plane - p[ax]) / d[ax]
            if t <= EPS_RAY or t >= best_t:
                continue
            ok = True
            for i in range(3):
                if i == ax:
                    continue
                ci = p[i] + t * d[i]
                if ci < bmin[i] - EPS_RAY or ci > bmax[i] + EPS_RAY:
                    ok = False
                    break
            if ok:
                best_t = t
                best_ax = ax
                best_sd = sd
    return best_t, best_ax, best_sd


@njit(cache=True, fastmath=True)
def _poly_first_hit(face_p, face_n, verts, faces, p, d):
    """First ray hit (t > EPS_RAY) on the convex polyhedron.
    Returns (t, face_idx); t = inf if none."""
    best_t = np.inf
    best_f = -1
    for f in range(face_p.shape[0]):
        denom = (d[0] * face_n[f, 0] + d[1] * face_n[f, 1]
                 + d[2] * face_n[f, 2])
        if abs(denom) < EPS_RAY:
            continue
        t = ((face_p[f, 0] - p[0]) * face_n[f, 0]
             + (face_p[f, 1] - p[1]) * face_n[f, 1]
             + (face_p[f, 2] - p[2]) * face_n[f, 2]) / denom
        if t <= EPS_RAY or t >= best_t:
            continue
        hx = p[0] + t * d[0]
        hy = p[1] + t * d[1]
        hz = p[2] + t * d[2]
        inside = True
        m = faces.shape[1]
        for i in range(m):
            a = faces[f, i]
            b = faces[f, (i + 1) % m]
            ex = verts[b, 0] - verts[a, 0]
            ey = verts[b, 1] - verts[a, 1]
            ez = verts[b, 2] - verts[a, 2]
            vx = hx - verts[a, 0]
            vy = hy - verts[a, 1]
            vz = hz - verts[a, 2]
            cx = ey * vz - ez * vy
            cy = ez * vx - ex * vz
            cz = ex * vy - ey * vx
            if (cx * face_n[f, 0] + cy * face_n[f, 1]
                    + cz * face_n[f, 2]) < -EPS_RAY:
                inside = False
                break
        if inside:
            best_t = t
            best_f = f
    return best_t, best_f


@njit(cache=True, fastmath=True)
def _in_box(bmin, bmax, p):
    for i in range(3):
        if p[i] < bmin[i] - EPS_CONTAIN or p[i] > bmax[i] + EPS_CONTAIN:
            return False
    return True


@njit(cache=True, fastmath=True)
def _in_poly(face_p, face_n, p):
    for f in range(face_p.shape[0]):
        if ((p[0] - face_p[f, 0]) * face_n[f, 0]
                + (p[1] - face_p[f, 1]) * face_n[f, 1]
                + (p[2] - face_p[f, 2]) * face_n[f, 2]) > EPS_CONTAIN:
            return False
    return True


@njit(cache=True, fastmath=True)
def _locate(face_p, face_n, clad_min, clad_max, core_min, core_max, p):
    """Priority order of the v2 scene: core(20) > clad(10) > scint(0)."""
    if _in_box(core_min, core_max, p):
        return 2
    if _in_box(clad_min, clad_max, p):
        return 1
    if _in_poly(face_p, face_n, p):
        return 0
    return -1


@njit(cache=True, fastmath=True)
def _box_on_edge(bmin, bmax, p):
    nb = 0
    for i in range(3):
        if abs(p[i] - bmin[i]) <= EPS_EDGE or abs(p[i] - bmax[i]) <= EPS_EDGE:
            nb += 1
    return nb >= 2


@njit(cache=True, fastmath=True)
def _poly_on_edge(verts, faces, p):
    for f in range(faces.shape[0]):
        m = faces.shape[1]
        for i in range(m):
            a = faces[f, i]
            b = faces[f, (i + 1) % m]
            abx = verts[b, 0] - verts[a, 0]
            aby = verts[b, 1] - verts[a, 1]
            abz = verts[b, 2] - verts[a, 2]
            apx = p[0] - verts[a, 0]
            apy = p[1] - verts[a, 1]
            apz = p[2] - verts[a, 2]
            denom = abx * abx + aby * aby + abz * abz
            t = (apx * abx + apy * aby + apz * abz) / denom
            if t < -EPS_EDGE or t > 1.0 + EPS_EDGE:
                continue
            cx = verts[a, 0] + t * abx - p[0]
            cy = verts[a, 1] + t * aby - p[1]
            cz = verts[a, 2] + t * abz - p[2]
            if cx * cx + cy * cy + cz * cz <= EPS_EDGE * EPS_EDGE:
                return True
    return False


# ---------------------------------------------------------------------------
# the propagation kernel
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def propagate_batch(pts, dirs, times, seeds,
                    face_p, face_n, verts, faces,
                    clad_min, clad_max, core_min, core_max,
                    n_mat, mu_mat):
    n = pts.shape[0]
    out_status = np.full(n, ST_LOST, dtype=np.int8)
    out_time = times.copy()
    core_ymax = core_max[1]
    state = np.empty(1, dtype=np.uint64)

    for ph in range(n):
        state[0] = _splitmix64(seeds[ph])
        px, py, pz = pts[ph, 0], pts[ph, 1], pts[ph, 2]
        dx, dy, dz = dirs[ph, 0], dirs[ph, 1], dirs[ph, 2]
        t_now = times[ph]
        status = ST_LOST
        p = np.empty(3)
        d = np.empty(3)

        for _ in range(MAX_STEPS):
            p[0] = px; p[1] = py; p[2] = pz
            d[0] = dx; d[1] = dy; d[2] = dz

            # ---- 1. nearest boundary among the three volumes -------------
            t_poly, f_poly = _poly_first_hit(face_p, face_n, verts, faces, p, d)
            t_clad, ax_c, sd_c = _box_first_hit(clad_min, clad_max, p, d)
            t_core, ax_k, sd_k = _box_first_hit(core_min, core_max, p, d)
            t_hit = t_poly
            which = 0
            if t_clad < t_hit:
                t_hit = t_clad; which = 1
            if t_core < t_hit:
                t_hit = t_core; which = 2
            if t_hit > 1e30:          # no boundary ahead (fastmath-safe inf test)
                status = ST_LOST
                break

            # winning-face outward normal
            nx = ny = nz = 0.0
            if which == 0:
                nx = face_n[f_poly, 0]; ny = face_n[f_poly, 1]
                nz = face_n[f_poly, 2]
            elif which == 1:
                if ax_c == 0:
                    nx = 1.0 if sd_c == 1 else -1.0
                elif ax_c == 1:
                    ny = 1.0 if sd_c == 1 else -1.0
                else:
                    nz = 1.0 if sd_c == 1 else -1.0
            else:
                if ax_k == 0:
                    nx = 1.0 if sd_k == 1 else -1.0
                elif ax_k == 1:
                    ny = 1.0 if sd_k == 1 else -1.0
                else:
                    nz = 1.0 if sd_k == 1 else -1.0

            # from-volume at segment midpoint (engine convention)
            p[0] = px + 0.5 * t_hit * dx
            p[1] = py + 0.5 * t_hit * dy
            p[2] = pz + 0.5 * t_hit * dz
            v_from = _locate(face_p, face_n, clad_min, clad_max,
                             core_min, core_max, p)
            n_from = n_mat[v_from] if v_from >= 0 else n_mat[3]
            mu_from = mu_mat[v_from] if v_from >= 0 else mu_mat[3]

            hx = px + t_hit * dx
            hy = py + t_hit * dy
            hz = pz + t_hit * dz

            # ---- 2. edge guard on the from-volume ------------------------
            p[0] = hx; p[1] = hy; p[2] = hz
            on_edge = False
            if v_from == 0:
                on_edge = _poly_on_edge(verts, faces, p)
            elif v_from == 1:
                on_edge = _box_on_edge(clad_min, clad_max, p)
            elif v_from == 2:
                on_edge = _box_on_edge(core_min, core_max, p)
            if on_edge:
                t_now += t_hit * n_from / C_CM_NS
                status = ST_EDGE
                break

            # ---- 3. bulk absorption --------------------------------------
            if mu_from > 0.0:
                d_abs = -np.log(_rand(state)) / mu_from
                if d_abs < t_hit:
                    t_now += d_abs * n_from / C_CM_NS
                    status = ST_ABSORBED
                    break
            t_now += t_hit * n_from / C_CM_NS

            # oriented normal (against direction)
            if dx * nx + dy * ny + dz * nz > 0.0:
                nx = -nx; ny = -ny; nz = -nz

            # ---- 4. catch test (core box, +y face, moving +y) ------------
            if v_from == 2 and dy > 0.0 and abs(hy - core_ymax) <= CATCH_TOL:
                status = ST_CAUGHT
                break

            # ---- 5. Fresnel at the boundary ------------------------------
            # to-material: located at midpoint of the NEXT segment
            p[0] = hx; p[1] = hy; p[2] = hz
            t2_poly, _f2 = _poly_first_hit(face_p, face_n, verts, faces, p, d)
            t2_clad, _a2, _s2 = _box_first_hit(clad_min, clad_max, p, d)
            t2_core, _a3, _s3 = _box_first_hit(core_min, core_max, p, d)
            t2 = t2_poly
            if t2_clad < t2:
                t2 = t2_clad
            if t2_core < t2:
                t2 = t2_core
            if t2 > 1e30:
                v_to = -1
            else:
                p[0] = hx + 0.5 * t2 * dx
                p[1] = hy + 0.5 * t2 * dy
                p[2] = hz + 0.5 * t2 * dz
                v_to = _locate(face_p, face_n, clad_min, clad_max,
                               core_min, core_max, p)
            n_to = n_mat[v_to] if v_to >= 0 else n_mat[3]

            cos_i = -(dx * nx + dy * ny + dz * nz)      # > 0 (oriented normal)
            if cos_i > 1.0:
                cos_i = 1.0
            sin_i2 = 1.0 - cos_i * cos_i
            eta = n_from / n_to
            sin_t2 = eta * eta * sin_i2

            reflect = False
            cos_t = 0.0
            if sin_t2 > 1.0:                            # TIR
                reflect = True
            else:
                cos_t = np.sqrt(1.0 - sin_t2)
                rs = (n_from * cos_i - n_to * cos_t) / (n_from * cos_i
                                                        + n_to * cos_t)
                rp = (n_to * cos_i - n_from * cos_t) / (n_to * cos_i
                                                        + n_from * cos_t)
                R = 0.5 * (rs * rs + rp * rp)
                if _rand(state) < R:
                    reflect = True

            if reflect:
                dot = dx * nx + dy * ny + dz * nz       # < 0
                rx = dx - 2.0 * dot * nx
                ry = dy - 2.0 * dot * ny
                rz = dz - 2.0 * dot * nz
                inv = 1.0 / np.sqrt(rx * rx + ry * ry + rz * rz)
                dx = rx * inv; dy = ry * inv; dz = rz * inv
                px, py, pz = hx, hy, hz
                continue

            # transmitted (Snell)
            tx = eta * dx + (eta * cos_i - cos_t) * nx
            ty = eta * dy + (eta * cos_i - cos_t) * ny
            tz = eta * dz + (eta * cos_i - cos_t) * nz
            inv = 1.0 / np.sqrt(tx * tx + ty * ty + tz * tz)
            dx = tx * inv; dy = ty * inv; dz = tz * inv
            px, py, pz = hx, hy, hz
            if v_to == -1:
                status = ST_LOST
                break

        out_status[ph] = status
        out_time[ph] = t_now
    return out_status, out_time


# ---------------------------------------------------------------------------
# drop-in event runner (photon GENERATION identical to run_event_v2)
# ---------------------------------------------------------------------------
def run_event_v2_fast(theta, mips, seed: int, light_yield: float = LY):
    """Same signature/returns as sim_v2.run_event_v2, propagation via numba."""
    u, v, E, t0 = mips
    mat = scaled_scintillator("EJ200", light_yield)
    scene_arrays = compile_scene(theta, mat)
    bw, bh = theta[0], theta[1]
    dedx = 2.0 * mat.density_g_cm3
    rng = np.random.default_rng(seed)
    inter = ParticleInteraction(build_scene_v2(theta, mat), MIP())

    pts_all, dirs_all, times_all = [], [], []
    for i in range(len(u)):
        x, y = float(u[i] * bw), float(v[i] * bh)
        L = min(float(E[i]) / dedx, 0.95 * local_depth(theta, x))
        pts, ptimes = inter.generate_photon_source(
            np.array([x, y, 1e-4]), np.array([0.0, 0.0, 1.0]), L, DX,
            BETA, dedx, light_yield, rng)
        dirs, phot_t = inter.photon_generator(
            pts, ptimes, mat.rise_time_ns, mat.decay_time_ns, rng)
        pts_all.append(pts); dirs_all.append(dirs)
        times_all.append(phot_t + t0[i])
    pts = np.ascontiguousarray(np.concatenate(pts_all), dtype=np.float64)
    dirs = np.ascontiguousarray(np.concatenate(dirs_all), dtype=np.float64)
    ptimes = np.ascontiguousarray(np.concatenate(times_all), dtype=np.float64)
    n_gen = len(ptimes)
    base = (seed % (2 ** 63)) * 0x9E3779B97F4A7C15 % (2 ** 64)  # python-int math
    seeds = np.uint64(base) + np.arange(n_gen, dtype=np.uint64)

    status, times = propagate_batch(
        pts, dirs, ptimes, seeds,
        scene_arrays["face_p"], scene_arrays["face_n"],
        scene_arrays["verts"], scene_arrays["faces"],
        scene_arrays["clad_min"], scene_arrays["clad_max"],
        scene_arrays["core_min"], scene_arrays["core_max"],
        scene_arrays["n_mat"], scene_arrays["mu_mat"])

    t = times[status == ST_CAUGHT]
    feats = np.array(
        f_abs(t) + f_early(t)
        + [np.log1p(len(t)), len(t) / max(1, n_gen), np.log1p(n_gen)],
        dtype=np.float32)
    return feats, t.astype(np.float32), n_gen
