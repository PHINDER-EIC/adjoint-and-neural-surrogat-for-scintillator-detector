"""Numba photon-transport engine with the physics of the JAX engine (engine B).

Why a new engine
----------------
* Engine A (``ApproximateEngine``) is the reference, but it is pure Python
  (~11 ms per photon) and lacks Lambertian reflection.
* Engine B (``JaxApproximateEngine``) has the most complete physics, but moves
  all photons in lock-step, so most of its work is wasted on CPU.
* Engine C (``fast_engine_for_lorenzo``) is fast (~9 us per photon) but only
  supports one fixed three-volume layout and an older step order.

This engine keeps C's structure (one compiled loop per photon, early exit, no
allocation) and implements B's physics for any scene B accepts:

* any number of convex volumes (``BoxVolume``, ``ConvexPolyhedronVolume`` and
  subclasses) with priorities; outside material from the ``Scene``;
* explicit ``ReadoutPort``s, or the legacy +y face of a ``fiber_core`` box;
* step order of the current A and B: bulk absorption -> readout -> edge guard
  (skipped between volumes of the same material) -> surface;
* ``ReflectiveMaterial`` with reflectance, specular and **Lambertian** parts;
* unpolarised Fresnel reflection, total internal reflection, Snell refraction;
* full ``PhotonStatus`` codes (``*_AT_Scintillator`` / ``*_AT_Guide``),
  final position / direction / time, step count and a ``truncated`` flag.

Choices where A and B differ
----------------------------
* Tolerances default to A's float64 values (1e-9 cm geometry, 1e-8 cm
  readout); B's looser values can be set in ``NumbaTrackingConfig``.
* Directions are normalised before tracking (B's behaviour; A does not).
* A photon with no boundary ahead, or outside every volume, ends as ``LOST``
  (B's behaviour; A raises).
* "Same material" means the same material object (B's behaviour).

Random numbers: every photon has its own xorshift64* stream seeded from
``(seed, photon index)``. Results are bit-identical for any number of threads
and any call order, and for any batch split when each chunk passes its start
index as ``index_offset``. They differ photon-by-photon from A and B
(which use other generators); they agree statistically.

Usage::

    from numba_engine import NumbaEngine
    engine = NumbaEngine(scene)                 # compiles the scene to arrays once
    result = engine.track_photon_batch(batch, seed=42)
    result.batch.statuses, result.batch.times, result.truncated, result.steps
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from Photon_Simulator.core.detectorSetup.material import ReflectiveMaterial
from Photon_Simulator.core.detectorSetup.Shapes.box import BoxVolume
from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import ConvexPolyhedronVolume
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch, PhotonEvent, PhotonStatus
from Photon_Simulator.core.photonsEngines.batchTransport import TransportResult

__all__ = ["NumbaEngine", "NumbaTrackingConfig", "compile_scene", "SceneArrays"]

# Status codes (same values as PhotonStatus)
_RUN_T, _RUN_I = int(PhotonStatus.RUNNING_TRANSMITTED), int(PhotonStatus.RUNNING_INSIDE)
_LOST, _CAUGHT, _EDGE = int(PhotonStatus.LOST), int(PhotonStatus.CAUGHT), int(PhotonStatus.ABSORBED_AT_EDGES)
_REFL_S, _REFL_G = int(PhotonStatus.ABSORBED_REFL_AT_Scintillator), int(PhotonStatus.ABSORBED_REFL_AT_Guide)
_LAMB_S, _LAMB_G = int(PhotonStatus.ABSORBED_LAMBDA_AT_Scintillator), int(PhotonStatus.ABSORBED_LAMBDA_AT_Guide)
_LOST_S, _LOST_G = int(PhotonStatus.LOST_AT_Scintillator), int(PhotonStatus.LOST_AT_Guide)
_MASK64 = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class NumbaTrackingConfig:
    """Tracking settings. Defaults reproduce engine A's tolerances."""

    max_steps: int = 1_000_000
    c_cm_ns: float = 29.9792458
    ray_epsilon: float = 1e-9            # minimum distance to the next boundary (A: 1e-9, B: 1e-6)
    parallel_epsilon: float = 1e-12      # |cos| below which a face is treated as parallel (A box: 1e-12)
    containment_tolerance: float = 1e-9  # point-in-volume / point-on-face slack (A: 1e-9, B: 1e-6)
    edge_tolerance: float = 1e-9         # distance to two faces that counts as an edge (A: 1e-9, B: 2e-6)
    catch_tolerance: float = 1e-8        # readout-plane slack; ports use max(this, port.tolerance)

    def __post_init__(self):
        if self.max_steps <= 0 or self.c_cm_ns <= 0:
            raise ValueError("max_steps and c_cm_ns must be positive")
        for name in ("ray_epsilon", "parallel_epsilon", "containment_tolerance", "edge_tolerance", "catch_tolerance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class SceneArrays:
    """Flat float64/int arrays describing a Scene (built once by ``compile_scene``)."""

    face_points: np.ndarray     # (V, F, 3) one point per face plane, padded
    face_normals: np.ndarray    # (V, F, 3) outward normals, padded
    face_count: np.ndarray      # (V,) number of real faces per volume
    locate_order: np.ndarray    # (V,) volume indices by descending priority (ties: scene order)
    volume_material: np.ndarray  # (V,) index into the material table
    volume_is_scint: np.ndarray  # (V,) True for volumes named "scintillator"
    refractive_index: np.ndarray  # (M,)
    absorption_mu: np.ndarray     # (M,) 1/cm
    reflective: np.ndarray        # (M,) bool
    reflectance: np.ndarray       # (M,)
    specular_fraction: np.ndarray  # (M,)
    outside_material: int
    port_volume: np.ndarray       # (P,) volume index, -1 = disabled
    port_point: np.ndarray        # (P, 3)
    port_normal: np.ndarray       # (P, 3) outward, unit
    port_tolerance: np.ndarray    # (P,)
    volume_names: tuple


def compile_scene(scene) -> SceneArrays:
    """Convert a ``Scene`` to flat arrays. Mirrors ``JaxEngine.adapters.scene_to_jax``."""
    volumes = list(scene.volumes)
    if not volumes:
        raise ValueError("scene has no volumes")
    bad = [v.name for v in volumes if not isinstance(v, ConvexPolyhedronVolume)]
    if bad:
        raise TypeError(f"only convex polyhedra are supported; unsupported volume(s): {', '.join(bad)}")

    n_vol = len(volumes)
    max_f = max(len(v.face_points) for v in volumes)
    face_points = np.zeros((n_vol, max_f, 3))
    face_normals = np.zeros((n_vol, max_f, 3))
    face_count = np.zeros(n_vol, dtype=np.int64)
    materials: list = []

    def mat_index(m):                      # identity, like B (avoids dataclass ==)
        for i, c in enumerate(materials):
            if c is m:
                return i
        materials.append(m)
        return len(materials) - 1

    volume_material = np.zeros(n_vol, dtype=np.int64)
    for i, v in enumerate(volumes):
        k = len(v.face_points)
        face_points[i, :k] = np.asarray(v.face_points, dtype=float)
        face_normals[i, :k] = np.asarray(v.face_normals, dtype=float)
        face_count[i] = k
        volume_material[i] = mat_index(v.material)
    outside = mat_index(scene.outside_material)

    priorities = np.array([v.priority for v in volumes])
    locate_order = np.array(sorted(range(n_vol), key=lambda i: -priorities[i]), dtype=np.int64)  # stable
    volume_is_scint = np.array([v.name == "scintillator" for v in volumes])

    n_idx, mu, refl, rfl, spec = [], [], [], [], []
    for m in materials:
        is_refl = isinstance(m, ReflectiveMaterial)
        n = getattr(m, "refractive_index", None)
        a = getattr(m, "absorption_mu", None)
        n_idx.append(1.0 if n is None else float(n))
        mu.append(0.0 if a is None else float(a))
        refl.append(is_refl)
        rfl.append(float(getattr(m, "reflectance", 0.0)) if is_refl else 0.0)
        spec.append(float(getattr(m, "specular_fraction", 1.0)) if is_refl else 1.0)

    ports = list(getattr(scene, "readout_ports", ()) or ())
    pv, pp, pn, pt = [], [], [], []
    if ports:
        for port in ports:
            pv.append(next(i for i, v in enumerate(volumes) if v.name == port.volume_name))
            pp.append(port.point); pn.append(port.outward_normal); pt.append(port.tolerance)
    else:                                   # legacy: +y face of a BoxVolume named fiber_core
        for i, v in enumerate(volumes):
            if v.name == "fiber_core" and isinstance(v, BoxVolume):
                pv.append(i); pp.append((0.0, float(v.box_max[1]), 0.0)); pn.append((0.0, 1.0, 0.0)); pt.append(0.0)
    if not pv:
        pv, pp, pn, pt = [-1], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [0.0]

    return SceneArrays(
        face_points=face_points, face_normals=face_normals, face_count=face_count,
        locate_order=locate_order, volume_material=volume_material, volume_is_scint=volume_is_scint,
        refractive_index=np.array(n_idx), absorption_mu=np.array(mu), reflective=np.array(refl),
        reflectance=np.array(rfl), specular_fraction=np.array(spec), outside_material=int(outside),
        port_volume=np.array(pv, dtype=np.int64), port_point=np.array(pp, dtype=float),
        port_normal=np.array(pn, dtype=float), port_tolerance=np.array(pt, dtype=float),
        volume_names=tuple(v.name for v in volumes),
    )


# ---------------------------------------------------------------------------
# Random numbers: splitmix64 seeding, xorshift64* stream per photon
# ---------------------------------------------------------------------------
@njit(cache=True)
def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(_MASK64)
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(_MASK64)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(_MASK64)
    z = z ^ (z >> np.uint64(31))
    if z == np.uint64(0):
        z = np.uint64(0x9E3779B97F4A7C15)
    return z


@njit(cache=True)
def _next_uniform(s):
    """Advance xorshift64* state s; return (new_state, u) with u in (0, 1]."""
    s ^= s << np.uint64(13)
    s &= np.uint64(_MASK64)
    s ^= s >> np.uint64(7)
    s ^= s << np.uint64(17)
    s &= np.uint64(_MASK64)
    r = (s * np.uint64(0x2545F4914F6CDD1D)) & np.uint64(_MASK64)
    u = (float(r >> np.uint64(11)) + 1.0) * (1.0 / 9007199254740992.0)
    return s, u


# ---------------------------------------------------------------------------
# Geometry kernels
# ---------------------------------------------------------------------------
@njit(cache=True, error_model="numpy")
def _inside(v, x, y, z, fp, fn, fc, tol):
    for f in range(fc[v]):
        if ((x - fp[v, f, 0]) * fn[v, f, 0] + (y - fp[v, f, 1]) * fn[v, f, 1]
                + (z - fp[v, f, 2]) * fn[v, f, 2]) > tol:
            return False
    return True


@njit(cache=True, error_model="numpy")
def _locate(x, y, z, fp, fn, fc, order, tol):
    """Highest-priority volume containing the point, -1 if none."""
    for k in range(order.shape[0]):
        v = order[k]
        if _inside(v, x, y, z, fp, fn, fc, tol):
            return v
    return -1


@njit(cache=True, error_model="numpy")
def _nearest_hit(px, py, pz, dx, dy, dz, fp, fn, fc, ray_eps, par_eps, tol):
    """Nearest face crossing among all volumes: (t, volume, face); t < 0 if none."""
    best_t = 1e300
    best_v = -1
    best_f = -1
    for v in range(fc.shape[0]):
        for f in range(fc[v]):
            nx, ny, nz = fn[v, f, 0], fn[v, f, 1], fn[v, f, 2]
            den = dx * nx + dy * ny + dz * nz
            if abs(den) < par_eps:
                continue
            t = ((fp[v, f, 0] - px) * nx + (fp[v, f, 1] - py) * ny + (fp[v, f, 2] - pz) * nz) / den
            if t <= ray_eps or t >= best_t:
                continue
            hx, hy, hz = px + t * dx, py + t * dy, pz + t * dz
            on_face = True                     # inside every other face plane of this volume
            for g in range(fc[v]):
                if g == f:
                    continue
                if ((hx - fp[v, g, 0]) * fn[v, g, 0] + (hy - fp[v, g, 1]) * fn[v, g, 1]
                        + (hz - fp[v, g, 2]) * fn[v, g, 2]) > tol:
                    on_face = False
                    break
            if on_face:
                best_t, best_v, best_f = t, v, f
    if best_v < 0:
        return -1.0, -1, -1
    return best_t, best_v, best_f


@njit(cache=True, error_model="numpy")
def _on_edge(v, x, y, z, fp, fn, fc, tol):
    """Point lies on >= 2 face planes of volume v (edge or vertex)."""
    count = 0
    for f in range(fc[v]):
        s = ((x - fp[v, f, 0]) * fn[v, f, 0] + (y - fp[v, f, 1]) * fn[v, f, 1]
             + (z - fp[v, f, 2]) * fn[v, f, 2])
        if abs(s) <= tol:
            count += 1
            if count >= 2:
                return True
    return False


# ---------------------------------------------------------------------------
# The transport kernel
# ---------------------------------------------------------------------------
@njit(cache=True, error_model="numpy")
def _track_photon(i, base, st, px, py, pz, dx, dy, dz, t_now,
                  fp, fn, fc, order, vol_mat, vol_scint,
                  n_mat, mu_mat, refl_mat, rfl_mat, spec_mat, outside,
                  port_vol, port_pt, port_n, port_tol,
                  max_steps, c, ray_eps, par_eps, tol, edge_tol, catch_tol):
    """Track one photon to the end. Returns final point, direction, time, status, steps, still-running."""
    rng = _splitmix64(base ^ _splitmix64(np.uint64(i)))
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    if norm > 0.0:
        dx, dy, dz = dx / norm, dy / norm, dz / norm
    running = True
    k = 0
    while k < max_steps:
        k += 1
        # 1. nearest boundary and the volume we are in (segment midpoint)
        t_hit, hv, hf = _nearest_hit(px, py, pz, dx, dy, dz, fp, fn, fc, ray_eps, par_eps, tol)
        if t_hit < 0.0:
            st = _LOST; running = False; break
        v_from = _locate(px + 0.5 * t_hit * dx, py + 0.5 * t_hit * dy, pz + 0.5 * t_hit * dz,
                         fp, fn, fc, order, tol)
        if v_from < 0:
            st = _LOST; running = False; break
        m_from = vol_mat[v_from]
        n_from = n_mat[m_from]
        scint = vol_scint[v_from]

        # 2. bulk absorption
        mu = mu_mat[m_from]
        if mu > 0.0:
            rng, u = _next_uniform(rng)
            d_abs = -np.log(u) / mu
            if d_abs < t_hit:
                px += d_abs * dx; py += d_abs * dy; pz += d_abs * dz
                t_now += d_abs * n_from / c
                st = _LAMB_S if scint else _LAMB_G
                running = False; break

        hx, hy, hz = px + t_hit * dx, py + t_hit * dy, pz + t_hit * dz
        t_now += t_hit * n_from / c
        px, py, pz = hx, hy, hz

        # volume on the other side: midpoint of the next segment
        t2, _v2, _f2 = _nearest_hit(hx, hy, hz, dx, dy, dz, fp, fn, fc, ray_eps, par_eps, tol)
        if t2 < 0.0:
            v_to = -1
        else:
            v_to = _locate(hx + 0.5 * t2 * dx, hy + 0.5 * t2 * dy, hz + 0.5 * t2 * dz,
                           fp, fn, fc, order, tol)
        m_to = vol_mat[v_to] if v_to >= 0 else outside

        # 3. readout ports
        caught = False
        for q in range(port_vol.shape[0]):
            if port_vol[q] != v_from:
                continue
            pnx, pny, pnz = port_n[q, 0], port_n[q, 1], port_n[q, 2]
            if dx * pnx + dy * pny + dz * pnz <= 0.0:
                continue
            s = (hx - port_pt[q, 0]) * pnx + (hy - port_pt[q, 1]) * pny + (hz - port_pt[q, 2]) * pnz
            if abs(s) <= max(catch_tol, port_tol[q]):
                caught = True
                break
        if caught:
            st = _CAUGHT; running = False; break

        # 4. edge guard (not between two volumes of the same material)
        same_material = v_to >= 0 and m_to == m_from
        if not same_material and _on_edge(v_from, hx, hy, hz, fp, fn, fc, edge_tol):
            st = _EDGE; running = False; break

        # surface normal oriented against the ray
        nx, ny, nz = fn[hv, hf, 0], fn[hv, hf, 1], fn[hv, hf, 2]
        dot = dx * nx + dy * ny + dz * nz
        if dot > 0.0:
            nx, ny, nz, dot = -nx, -ny, -nz, -dot

        # 5a. reflector
        if refl_mat[m_to]:
            rng, u = _next_uniform(rng)
            if not (u < rfl_mat[m_to]):
                st = _REFL_S if scint else _REFL_G
                running = False; break
            rng, u = _next_uniform(rng)
            if u <= spec_mat[m_to]:
                rx, ry, rz = dx - 2.0 * dot * nx, dy - 2.0 * dot * ny, dz - 2.0 * dot * nz
            else:                                   # Lambertian, cosine-weighted around the normal
                rng, u1 = _next_uniform(rng)
                rng, u2 = _next_uniform(rng)
                r = np.sqrt(u1)
                phi = 2.0 * np.pi * u2
                lx, ly, lz = r * np.cos(phi), r * np.sin(phi), np.sqrt(max(0.0, 1.0 - u1))
                if abs(nz) < 0.9:
                    hx_, hy_, hz_ = 0.0, 0.0, 1.0
                else:
                    hx_, hy_, hz_ = 1.0, 0.0, 0.0
                tx, ty, tz = hy_ * nz - hz_ * ny, hz_ * nx - hx_ * nz, hx_ * ny - hy_ * nx
                tn = np.sqrt(tx * tx + ty * ty + tz * tz)
                tx, ty, tz = tx / tn, ty / tn, tz / tn
                bx, by, bz = ny * tz - nz * ty, nz * tx - nx * tz, nx * ty - ny * tx
                rx = lx * tx + ly * bx + lz * nx
                ry = lx * ty + ly * by + lz * ny
                rz = lx * tz + ly * bz + lz * nz
            rn = np.sqrt(rx * rx + ry * ry + rz * rz)
            dx, dy, dz = rx / rn, ry / rn, rz / rn
            st = _RUN_I
            continue

        # 5b. dielectric interface: Fresnel / TIR / Snell
        n_to = n_mat[m_to]
        cos_i = min(1.0, -dot)
        eta = n_from / n_to
        sin_t2 = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
        reflect = True
        cos_t = 0.0
        if sin_t2 <= 1.0:
            cos_t = np.sqrt(1.0 - sin_t2)
            rs = (n_from * cos_i - n_to * cos_t) / (n_from * cos_i + n_to * cos_t)
            rp = (n_to * cos_i - n_from * cos_t) / (n_to * cos_i + n_from * cos_t)
            big_r = 0.5 * (rs * rs + rp * rp)
            rng, u = _next_uniform(rng)
            reflect = u < big_r
        if reflect:
            rx, ry, rz = dx - 2.0 * dot * nx, dy - 2.0 * dot * ny, dz - 2.0 * dot * nz
            rn = np.sqrt(rx * rx + ry * ry + rz * rz)
            dx, dy, dz = rx / rn, ry / rn, rz / rn
            st = _RUN_I
            continue
        tx = eta * dx + (eta * cos_i - cos_t) * nx
        ty = eta * dy + (eta * cos_i - cos_t) * ny
        tz = eta * dz + (eta * cos_i - cos_t) * nz
        tn = np.sqrt(tx * tx + ty * ty + tz * tz)
        dx, dy, dz = tx / tn, ty / tn, tz / tn
        if v_to < 0:
            st = _LOST_S if scint else _LOST_G
            running = False; break
        st = _RUN_T
    return px, py, pz, dx, dy, dz, t_now, st, k, running


@njit(cache=True, parallel=True, error_model="numpy")
def _transport(points, dirs, times, statuses, seed, index_offset,
               fp, fn, fc, order, vol_mat, vol_scint,
               n_mat, mu_mat, refl_mat, rfl_mat, spec_mat, outside,
               port_vol, port_pt, port_n, port_tol,
               max_steps, c, ray_eps, par_eps, tol, edge_tol, catch_tol):
    n = points.shape[0]
    out_p = points.copy()
    out_d = dirs.copy()
    out_t = times.copy()
    out_s = statuses.copy()
    steps = np.zeros(n, dtype=np.int64)
    truncated = np.zeros(n, dtype=np.bool_)
    base = _splitmix64(np.uint64(seed))
    for i in prange(n):
        st = statuses[i]
        if st == _RUN_I or st == _RUN_T:
            r = _track_photon(i + index_offset, base, st, points[i, 0], points[i, 1], points[i, 2],
                              dirs[i, 0], dirs[i, 1], dirs[i, 2], times[i],
                              fp, fn, fc, order, vol_mat, vol_scint,
                              n_mat, mu_mat, refl_mat, rfl_mat, spec_mat, outside,
                              port_vol, port_pt, port_n, port_tol,
                              max_steps, c, ray_eps, par_eps, tol, edge_tol, catch_tol)
            out_p[i, 0], out_p[i, 1], out_p[i, 2] = r[0], r[1], r[2]
            out_d[i, 0], out_d[i, 1], out_d[i, 2] = r[3], r[4], r[5]
            out_t[i] = r[6]
            out_s[i] = r[7]
            steps[i] = r[8]
            truncated[i] = r[9]
    return out_p, out_d, out_t, out_s, steps, truncated


class NumbaEngine:
    """Fast CPU photon transport with engine B's physics.

    Parameters
    ----------
    scene : Scene
        Any scene made of convex volumes (what engine B accepts).
    config : NumbaTrackingConfig, optional
        Tolerances and step limit; defaults match engine A.

    Notes
    -----
    Threads: uses all numba threads (``numba.set_num_threads(k)`` to limit).
    Results do not depend on the thread count. The scene is converted once at
    construction; rebuild the engine after changing the scene.
    """

    def __init__(self, scene, config: NumbaTrackingConfig | None = None):
        if scene is None:
            raise ValueError("NumbaEngine: missing scene")
        self.scene = scene
        self.config = config or NumbaTrackingConfig()
        self.arrays = compile_scene(scene)

    def track_photon_batch(self, batch: PhotonBatch, *, seed: int = 0, index_offset: int = 0) -> TransportResult:
        """Track a ``PhotonBatch``; returns ``TransportResult(batch, truncated, steps, None)``.

        Photon ``i`` of the batch uses random stream ``(seed, index_offset + i)``.
        To split one big batch into chunks and get exactly the same result,
        pass each chunk's start index as ``index_offset``.
        """
        if not 0 <= int(seed) < 2 ** 64:
            raise ValueError("seed must fit an unsigned 64-bit integer")
        if int(index_offset) < 0:
            raise ValueError("index_offset must be >= 0")
        a, c = self.arrays, self.config
        pts = np.ascontiguousarray(batch.interaction_points, dtype=np.float64)
        dirs = np.ascontiguousarray(batch.directions, dtype=np.float64)
        times = np.ascontiguousarray(batch.times, dtype=np.float64)
        stat = np.ascontiguousarray(batch.statuses, dtype=np.int64)
        p, d, t, s, steps, trunc = _transport(
            pts, dirs, times, stat, np.uint64(int(seed)), np.int64(int(index_offset)),
            a.face_points, a.face_normals, a.face_count, a.locate_order, a.volume_material, a.volume_is_scint,
            a.refractive_index, a.absorption_mu, a.reflective, a.reflectance, a.specular_fraction,
            a.outside_material, a.port_volume, a.port_point, a.port_normal, a.port_tolerance,
            c.max_steps, c.c_cm_ns, c.ray_epsilon, c.parallel_epsilon, c.containment_tolerance,
            c.edge_tolerance, c.catch_tolerance)
        out = PhotonBatch(p, d, t, s.astype(np.int32))
        return TransportResult(out, trunc, steps.astype(np.int32), None)

    def track_one(self, photon: PhotonEvent, *, seed: int = 0) -> PhotonEvent:
        """Track one photon; returns its final ``PhotonEvent``."""
        b = PhotonBatch(np.asarray(photon.interaction_point, float)[None], np.asarray(photon.direction, float)[None],
                        np.array([float(photon.time)]), np.array([int(photon.status)], dtype=np.int32))
        r = self.track_photon_batch(b, seed=seed)
        return PhotonEvent(r.batch.interaction_points[0], r.batch.directions[0],
                           PhotonStatus(int(r.batch.statuses[0])), float(r.batch.times[0]))
