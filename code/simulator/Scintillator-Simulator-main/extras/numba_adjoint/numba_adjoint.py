"""Numba reverse (adjoint) tracer for the Stage-6 voxel optical field.

A compiled, multi-threaded replacement for the Python root tracer in
``phinder_bridge/scripts/phinder_stage6_adjoint_field.py`` (``_track_root``),
with the same physics, the same random numbers and the same output sums.

What one "root" does (unchanged from the repo tracer)
-----------------------------------------------------
* Start on the readout CORE face of one guide, cosine-weighted inward.
* Bulk absorption is a continuous Beer-Lambert weight, never a random kill.
* At a dielectric boundary the ray splits into a reflected child (weight R)
  and a transmitted child (weight T) when both are >= ``split_min_child_weight``
  and the branch budget allows; otherwise one child is chosen at random
  (unbiased). The random number is the repo's keyed counter-based draw
  (seed, root, branch key, step), so the two tracers make the same choices.
* Specular reflectors multiply the weight by their reflectance.
* Every weighted segment inside the volume named ``scintillator`` is split
  exactly at the voxel planes and scored as attenuated track length
  (``L``), time-weighted track length (``Lt``) and a time histogram.

Differences from the repo tracer
--------------------------------
* Compiled with numba; roots run in parallel over ``nblocks`` blocks, each
  with its own accumulators (no per-root 6 MB histogram).
* Ray/volume tests are skipped for volumes whose bounding box the ray cannot
  reach first. This only removes work; it does not change any hit.
* A root that fails (start outside the terminal core, a hit point outside
  every volume, a ray in a gap with no volume, two ports matching one hit)
  is dropped, as in the repo. Its partial time histogram is removed by
  re-tracing it with weight -1 (the draws are deterministic).
* Lambertian reflectors are refused when the tracer is built. The repo
  tracer raises inside each root that meets one, so those roots are counted
  as failed and the map is silently biased.
* The 18 historical source-box probes of the field script are not scored
  (the geometry atlas does not use them either).

Output: a ``merged`` dict with the same keys as the repo's
``phinder_stage6_adjoint_field._empty_merge``, so the atlas's
``_compute_products`` and ``_write_geometry_group`` can be used unchanged.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from numba import njit, prange
from scipy.stats import qmc

from Photon_Simulator.core.detectorSetup.material import ReflectiveMaterial
from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import ConvexPolyhedronVolume

__all__ = ["NumbaAdjointField", "TERMINAL_NAMES", "from_stage6_cfg"]

# terminal weight categories (names are the repo's Counter keys)
_T_MAX_STEPS, _T_MAX_TIME, _T_NO_HIT, _T_ZERO, _T_EDGE, _T_REFL, _T_IFACE, _T_ESC = range(8)
_T_PORT0 = 8                       # + port index
TERMINAL_NAMES = ["UNRESOLVED_MAX_STEPS", "UNRESOLVED_MAX_TIME", "NO_BOUNDARY_HIT", "ZERO_DISTANCE",
                  "AMBIGUOUS_MATERIAL_EDGE", "REFLECTOR_ABSORBED", "INTERFACE_ABSORBED", "ESCAPED_OUTSIDE"]

# per-root diagnostic columns
_D_OK, _D_REACHED, _D_BULK, _D_SPLITS, _D_FALLBACKS, _D_NODES, _D_MAXLIVE, _D_EVENTS, _D_LVOL, _D_LTVOL, _D_FAIL = range(11)
_D_TERM = 11                       # + terminal category

# failure codes (diag column _D_FAIL)
FAIL_NAMES = {1: "start not inside terminal core", 2: "hit point outside every volume",
              3: "segment in no volume (gap)", 4: "hit matches several readout ports"}


# ---------------------------------------------------------------------------
# keyed random numbers: bit-identical to the repo's _mix64 / _keyed_uniform01
# ---------------------------------------------------------------------------
@njit(cache=True)
def _mix64(x):
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return x


@njit(cache=True)
def _child_key(parent, step, side):
    x = parent ^ _mix64(step + np.uint64(0x9E3779B97F4A7C15))
    if side == 0:
        x = x ^ np.uint64(0xD1B54A32D192ED03)
    else:
        x = x ^ np.uint64(0x94D049BB133111EB)
    return _mix64(x)


@njit(cache=True)
def _keyed_u01(seed, ray_id, key, step):
    x = _mix64(seed ^ np.uint64(0xA0761D6478BD642F))
    x = x ^ _mix64(ray_id + np.uint64(0xE7037ED1A0B428DB))
    x = x ^ _mix64(key + np.uint64(0x8EBC6AF09C88C6E3))
    x = x ^ _mix64(step + np.uint64(0x589965CC75374CC3))
    z = _mix64(x)
    return float(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


# ---------------------------------------------------------------------------
# geometry (same tests as extras/numba_engine, plus bounding-box culling)
# ---------------------------------------------------------------------------
@njit(cache=True, error_model="numpy")
def _inside(v, x, y, z, fp, fn, fc, tol):
    for f in range(fc[v]):
        if ((x - fp[v, f, 0]) * fn[v, f, 0] + (y - fp[v, f, 1]) * fn[v, f, 1]
                + (z - fp[v, f, 2]) * fn[v, f, 2]) > tol:
            return False
    return True


@njit(cache=True, error_model="numpy")
def _locate(x, y, z, fp, fn, fc, order, bbmin, bbmax, tol):
    """Highest-priority volume containing the point, -1 if none."""
    for k in range(order.shape[0]):
        v = order[k]
        if (x < bbmin[v, 0] or x > bbmax[v, 0] or y < bbmin[v, 1] or y > bbmax[v, 1]
                or z < bbmin[v, 2] or z > bbmax[v, 2]):
            continue
        if _inside(v, x, y, z, fp, fn, fc, tol):
            return v
    return -1


@njit(cache=True, error_model="numpy")
def _nearest_hit(px, py, pz, dx, dy, dz, fp, fn, fc, bbmin, bbmax, ray_eps, par_eps, tol):
    """Nearest face crossing among all volumes: (t, volume, face); t < 0 if none."""
    best_t = 1e300
    best_v = -1
    best_f = -1
    for v in range(fc.shape[0]):
        # slab test against the padded bounding box: skip volumes that cannot give a closer hit
        t_in = -1e300
        t_out = 1e300
        miss = False
        for a in range(3):
            if a == 0:
                p, d = px, dx
            elif a == 1:
                p, d = py, dy
            else:
                p, d = pz, dz
            lo = bbmin[v, a]
            hi = bbmax[v, a]
            if d == 0.0:
                if p < lo or p > hi:
                    miss = True
                    break
            else:
                t1 = (lo - p) / d
                t2 = (hi - p) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_in:
                    t_in = t1
                if t2 < t_out:
                    t_out = t2
        if miss or t_out < ray_eps or t_in > best_t or t_in > t_out:
            continue
        for f in range(fc[v]):
            nx, ny, nz = fn[v, f, 0], fn[v, f, 1], fn[v, f, 2]
            den = dx * nx + dy * ny + dz * nz
            if abs(den) < par_eps:
                continue
            t = ((fp[v, f, 0] - px) * nx + (fp[v, f, 1] - py) * ny + (fp[v, f, 2] - pz) * nz) / den
            if t <= ray_eps or t >= best_t:
                continue
            hx, hy, hz = px + t * dx, py + t * dy, pz + t * dz
            on_face = True
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
# scoring: exact voxel partition + time bins (same arithmetic as the repo)
# ---------------------------------------------------------------------------
@njit(cache=True, error_model="numpy")
def _exp_integral(mu, s0, s1):
    if s1 <= s0:
        return 0.0, 0.0
    if mu <= 0.0:
        return s1 - s0, 0.5 * (s1 * s1 - s0 * s0)
    e0 = math.exp(-mu * s0)
    e1 = math.exp(-mu * s1)
    i0 = (e0 - e1) / mu
    i1 = (s0 / mu + 1.0 / (mu * mu)) * e0 - (s1 / mu + 1.0 / (mu * mu)) * e1
    return i0, i1


@njit(cache=True, error_model="numpy")
def _add_time(hist, vox, t0, noc, w0, mu, s0, s1, dt, tmax, nb, scale):
    ta = t0 + noc * s0
    tb = t0 + noc * s1
    if tb <= 0.0 or ta >= tmax:
        return
    aa = max(ta, 0.0)
    bb = min(tb, tmax)
    if bb <= aa:
        return
    ss0 = max(s0, (aa - t0) / noc)
    ss1 = min(s1, (bb - t0) / noc)
    i0 = max(0, int(math.floor(aa / dt)))
    i1 = min(nb - 1, int(math.floor(np.nextafter(bb, -np.inf) / dt)))
    for ib in range(i0, i1 + 1):
        tl = max(aa, ib * dt)
        tr = min(bb, (ib + 1) * dt)
        if tr <= tl:
            continue
        sl = max(ss0, (tl - t0) / noc)
        sr = min(ss1, (tr - t0) / noc)
        I0, _ = _exp_integral(mu, sl, sr)
        hist[vox, ib] += scale * w0 * I0


@njit(cache=True, error_model="numpy")
def _score_segment(ax, ay, az, bx, by, bz, t0, w0, mu, noc, spacing, shape,
                   Lg, Ltg, hist, dt, tmax, nb, scale, us):
    """Deposit one weighted segment [a, b] inside the scintillator."""
    d0, d1, d2 = bx - ax, by - ay, bz - az
    L = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    if L <= 0.0 or w0 <= 0.0:
        return
    m = 2
    us[0] = 0.0
    us[1] = 1.0
    for a in range(3):
        if a == 0:
            da, x0, x1 = d0, ax, bx
        elif a == 1:
            da, x0, x1 = d1, ay, by
        else:
            da, x0, x1 = d2, az, bz
        if abs(da) < 1e-15:
            continue
        lo = min(x0, x1)
        hi = max(x0, x1)
        h = spacing[a]
        k0 = max(1, int(math.floor(lo / h)) + 1)
        k1 = min(shape[a] - 1, int(math.floor((hi - 1e-14) / h)))
        for k in range(k0, k1 + 1):
            u = (k * h - x0) / da
            if 1e-13 < u < 1.0 - 1e-13:
                us[m] = u
                m += 1
    # insertion sort (m is small)
    for i in range(1, m):
        key = us[i]
        j = i - 1
        while j >= 0 and us[j] > key:
            us[j + 1] = us[j]
            j -= 1
        us[j + 1] = key
    # drop numerical duplicates, comparing with the previous ORIGINAL entry (as np.diff does)
    prev_orig = us[0]
    k = 1
    for i in range(1, m):
        cur = us[i]
        if cur - prev_orig > 1e-12:
            us[k] = cur
            k += 1
        prev_orig = cur
    ny_, nz_ = shape[1], shape[2]
    for i in range(k - 1):
        ua = us[i]
        ub = us[i + 1]
        if ub <= ua:
            continue
        um = 0.5 * (ua + ub)
        ix = int(math.floor((ax + um * d0) / spacing[0]))
        iy = int(math.floor((ay + um * d1) / spacing[1]))
        iz = int(math.floor((az + um * d2) / spacing[2]))
        ix = min(max(ix, 0), shape[0] - 1)
        iy = min(max(iy, 0), shape[1] - 1)
        iz = min(max(iz, 0), shape[2] - 1)
        s0 = ua * L
        s1 = ub * L
        I0, I1 = _exp_integral(mu, s0, s1)
        wl = w0 * I0
        if wl <= 0.0:
            continue
        vox = (ix * ny_ + iy) * nz_ + iz
        Lg[vox] += scale * wl
        Ltg[vox] += scale * w0 * (t0 * I0 + noc * I1)
        _add_time(hist, vox, t0, noc, w0, mu, s0, s1, dt, tmax, nb, scale)


# ---------------------------------------------------------------------------
# one root
# ---------------------------------------------------------------------------
@njit(cache=True, error_model="numpy")
def _trace_root(ray_id, sx, sy, sz, ux, uy, uz, scale, seed, start_vol,
                fp, fn, fc, order, bbmin, bbmax, vol_mat, vol_scint,
                n_mat, mu_mat, refl_mat, rfl_mat, outside,
                port_vol, port_pt, port_n, port_tol,
                c, ray_eps, par_eps, tol, edge_tol, catch_tol,
                max_steps, max_time, split_min, max_live, max_nodes,
                spacing, shape, dt, tmax, nb,
                Lg, Ltg, hist, us, sp, sd, sf, sk, diag):
    """Trace one root. Returns 0 on success or a failure code (see FAIL_NAMES)."""
    # stack: sp (pos), sd (dir), sf[:, 0]=t, sf[:, 1]=w, sf[:, 2]=steps, sk=branch key
    if _locate(sx, sy, sz, fp, fn, fc, order, bbmin, bbmax, tol) != start_vol:
        return 1
    ns = 1
    sp[0, 0], sp[0, 1], sp[0, 2] = sx, sy, sz
    sd[0, 0], sd[0, 1], sd[0, 2] = ux, uy, uz
    sf[0, 0], sf[0, 1], sf[0, 2] = 0.0, 1.0, 0.0
    sk[0] = _mix64(ray_id ^ np.uint64(0x243F6A8885A308D3))
    nodes = 1
    maxlive = 1
    splits = 0
    fallbacks = 0
    events = 0
    reached = 0
    bulk = 0.0

    while ns > 0:
        ns -= 1
        px, py, pz = sp[ns, 0], sp[ns, 1], sp[ns, 2]
        dx, dy, dz = sd[ns, 0], sd[ns, 1], sd[ns, 2]
        t = sf[ns, 0]
        w = sf[ns, 1]
        steps = int(sf[ns, 2])
        key = sk[ns]
        if w <= 0.0:
            continue
        while True:
            if steps >= max_steps:
                diag[_D_TERM + _T_MAX_STEPS] += w
                break
            if t >= max_time:
                diag[_D_TERM + _T_MAX_TIME] += w
                break
            # Scene.first_boundary_hit normalises the direction
            dn = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx, dy, dz = dx / dn, dy / dn, dz / dn
            L, hv, hf = _nearest_hit(px, py, pz, dx, dy, dz, fp, fn, fc, bbmin, bbmax, ray_eps, par_eps, tol)
            if L < 0.0:
                diag[_D_TERM + _T_NO_HIT] += w
                break
            v_from = _locate(px + 0.5 * L * dx, py + 0.5 * L * dy, pz + 0.5 * L * dz,
                             fp, fn, fc, order, bbmin, bbmax, tol)
            hx, hy, hz = px + L * dx, py + L * dy, pz + L * dz
            if _locate(hx, hy, hz, fp, fn, fc, order, bbmin, bbmax, tol) < 0:
                return 2
            m_from = vol_mat[v_from] if v_from >= 0 else outside
            # volume on the other side (midpoint of the next segment)
            L2, _v2, _f2 = _nearest_hit(hx, hy, hz, dx, dy, dz, fp, fn, fc, bbmin, bbmax, ray_eps, par_eps, tol)
            if L2 < 0.0:
                v_to = -1
            else:
                v_to = _locate(hx + 0.5 * L2 * dx, hy + 0.5 * L2 * dy, hz + 0.5 * L2 * dz,
                               fp, fn, fc, order, bbmin, bbmax, tol)
            m_to = vol_mat[v_to] if v_to >= 0 else outside

            n = n_mat[m_from]
            mu = mu_mat[m_from]
            noc = n / c
            dtseg = L * noc
            scint = v_from >= 0 and vol_scint[v_from]

            if t + dtseg > max_time:
                lcap = max(0.0, (max_time - t) * c / n)
                if scint and lcap > 0.0:
                    reached = 1
                    _score_segment(px, py, pz, px + lcap * dx, py + lcap * dy, pz + lcap * dz, t, w, mu, noc,
                                   spacing, shape, Lg, Ltg, hist, dt, tmax, nb, scale, us)
                wcap = w * math.exp(-mu * lcap)
                bulk += max(0.0, w - wcap)
                diag[_D_TERM + _T_MAX_TIME] += wcap
                break

            t_hit = t + dtseg
            if scint:
                reached = 1
                _score_segment(px, py, pz, hx, hy, hz, t, w, mu, noc,
                               spacing, shape, Lg, Ltg, hist, dt, tmax, nb, scale, us)
            wb = w * math.exp(-mu * L)
            bulk += max(0.0, w - wb)
            if wb <= 0.0:
                break
            events += 1
            steps += 1

            # readout ports (Scene.readout_channel)
            nmatch = 0
            qhit = -1
            if v_from >= 0:
                for q in range(port_vol.shape[0]):
                    if port_vol[q] != v_from:
                        continue
                    pnx, pny, pnz = port_n[q, 0], port_n[q, 1], port_n[q, 2]
                    s = (hx - port_pt[q, 0]) * pnx + (hy - port_pt[q, 1]) * pny + (hz - port_pt[q, 2]) * pnz
                    if abs(s) <= max(catch_tol, port_tol[q]) and dx * pnx + dy * pny + dz * pnz > 0.0:
                        nmatch += 1
                        qhit = q
            if nmatch > 1:
                return 4
            if nmatch == 1:
                diag[_D_TERM + _T_PORT0 + qhit] += wb
                break
            if v_from < 0:
                return 3          # repo: from_volume is None -> AttributeError -> root dropped

            if _on_edge(v_from, hx, hy, hz, fp, fn, fc, edge_tol):
                if v_to >= 0 and m_from == m_to:
                    px, py, pz, t, w = hx, hy, hz, t_hit, wb
                    continue
                diag[_D_TERM + _T_EDGE] += wb
                break

            nx, ny, nz = fn[hv, hf, 0], fn[hv, hf, 1], fn[hv, hf, 2]
            dot = dx * nx + dy * ny + dz * nz
            if dot > 0.0:
                nx, ny, nz, dot = -nx, -ny, -nz, -dot
            rx, ry, rz = dx - 2.0 * dot * nx, dy - 2.0 * dot * ny, dz - 2.0 * dot * nz

            if refl_mat[m_to]:
                wr = wb * rfl_mat[m_to]
                diag[_D_TERM + _T_REFL] += wb - wr
                if wr <= 0.0:
                    break
                px, py, pz, dx, dy, dz, t, w = hx, hy, hz, rx, ry, rz, t_hit, wr
                continue

            # Fresnel, written as in ApproximateEngine._fresnel_coefficients
            n_from = n_mat[m_from]
            n_to = n_mat[m_to]
            theta_in = math.acos(min(1.0, max(-1.0, abs(dot))))
            sin_t = n_from * math.sin(theta_in) / n_to
            if abs(sin_t) > 1.0:
                R, T = 1.0, 0.0
            else:
                theta_t = math.asin(sin_t)
                ci = math.cos(theta_in)
                ct = math.cos(theta_t)
                nfc = n_from * ci
                ntc = n_to * ct
                den_s = nfc + ntc
                den_p = n_to * ci + n_from * ct
                r_s = (nfc - ntc) / den_s
                r_p = (n_to * ci - n_from * ct) / den_p
                t_s = 2.0 * nfc / den_s
                t_p = 2.0 * nfc / den_p
                pref = ntc / nfc
                R = 0.5 * (r_s * r_s + r_p * r_p)
                T = 0.5 * (pref * t_s * t_s + pref * t_p * t_p)
            R = max(0.0, R)
            T = max(0.0, T)
            srt = R + T
            if srt <= 0.0:
                diag[_D_TERM + _T_IFACE] += wb
                break
            R /= srt
            T /= srt

            # transmitted direction (ApproximateEngine._transmitted_direction)
            cos_i = -dot
            eta = n_from / n_to
            sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
            t_ok = sin_t2 <= 1.0
            tx = ty = tz = 0.0
            if t_ok:
                cos_t = math.sqrt(1.0 - sin_t2)
                tx = eta * dx + (eta * cos_i - cos_t) * nx
                ty = eta * dy + (eta * cos_i - cos_t) * ny
                tz = eta * dz + (eta * cos_i - cos_t) * nz
                tn = math.sqrt(tx * tx + ty * ty + tz * tz)
                tx, ty, tz = tx / tn, ty / tn, tz / tn

            if T <= 1e-15:
                px, py, pz, dx, dy, dz, t, w = hx, hy, hz, rx, ry, rz, t_hit, wb
                continue
            if R <= 1e-15:
                if not t_ok:
                    px, py, pz, dx, dy, dz, t, w = hx, hy, hz, rx, ry, rz, t_hit, wb
                    continue
                if v_to < 0:
                    diag[_D_TERM + _T_ESC] += wb
                    break
                px, py, pz, dx, dy, dz, t, w = hx, hy, hz, tx, ty, tz, t_hit, wb
                continue

            wr = wb * R
            wt = wb * T
            if min(wr, wt) >= split_min and ns + 2 <= max_live and nodes + 1 <= max_nodes:
                if not t_ok:
                    px, py, pz, dx, dy, dz, t, w = hx, hy, hz, rx, ry, rz, t_hit, wb
                    continue
                splits += 1
                nodes += 1
                ust = np.uint64(steps)
                kr = _child_key(key, ust, 0)
                kt = _child_key(key, ust, 1)
                if v_to < 0:
                    diag[_D_TERM + _T_ESC] += wt
                    px, py, pz, dx, dy, dz, t, w, key = hx, hy, hz, rx, ry, rz, t_hit, wr, kr
                    continue
                sp[ns, 0], sp[ns, 1], sp[ns, 2] = hx, hy, hz
                sf[ns, 0], sf[ns, 2] = t_hit, float(steps)
                if wr >= wt:      # push transmitted, continue reflected
                    sd[ns, 0], sd[ns, 1], sd[ns, 2] = tx, ty, tz
                    sf[ns, 1] = wt
                    sk[ns] = kt
                    px, py, pz, dx, dy, dz, t, w, key = hx, hy, hz, rx, ry, rz, t_hit, wr, kr
                else:
                    sd[ns, 0], sd[ns, 1], sd[ns, 2] = rx, ry, rz
                    sf[ns, 1] = wr
                    sk[ns] = kr
                    px, py, pz, dx, dy, dz, t, w, key = hx, hy, hz, tx, ty, tz, t_hit, wt, kt
                ns += 1
                if ns + 1 > maxlive:
                    maxlive = ns + 1
                continue

            # unbiased analog fallback with the keyed draw
            fallbacks += 1
            u = _keyed_u01(seed, ray_id, key, np.uint64(steps))
            if u < R or not t_ok:
                px, py, pz, dx, dy, dz, t, w = hx, hy, hz, rx, ry, rz, t_hit, wb
                continue
            if v_to < 0:
                diag[_D_TERM + _T_ESC] += wb
                break
            px, py, pz, dx, dy, dz, t, w = hx, hy, hz, tx, ty, tz, t_hit, wb

    diag[_D_REACHED] = reached
    diag[_D_BULK] = bulk
    diag[_D_SPLITS] = splits
    diag[_D_FALLBACKS] = fallbacks
    diag[_D_NODES] = nodes
    diag[_D_MAXLIVE] = maxlive
    diag[_D_EVENTS] = events
    return 0


@njit(cache=True, parallel=True, error_model="numpy")
def _run_roots(starts, dirs, ray_ids, seed, start_vol, nblocks,
               fp, fn, fc, order, bbmin, bbmax, vol_mat, vol_scint,
               n_mat, mu_mat, refl_mat, rfl_mat, outside,
               port_vol, port_pt, port_n, port_tol,
               c, ray_eps, par_eps, tol, edge_tol, catch_tol,
               max_steps, max_time, split_min, max_live, max_nodes,
               spacing, shape, dt, tmax, nb, ndiag):
    N = starts.shape[0]
    V = shape[0] * shape[1] * shape[2]
    sumL = np.zeros((nblocks, V))
    sumL2 = np.zeros((nblocks, V))
    sumLt = np.zeros((nblocks, V))
    hits = np.zeros((nblocks, V), dtype=np.int64)
    hist = np.zeros((nblocks, V, nb))
    diag = np.zeros((N, ndiag))
    nus = shape[0] + shape[1] + shape[2] + 4
    for b in prange(nblocks):
        Lg = np.zeros(V)
        Ltg = np.zeros(V)
        us = np.zeros(nus)
        sp = np.zeros((max_live + 2, 3))
        sd = np.zeros((max_live + 2, 3))
        sf = np.zeros((max_live + 2, 3))
        sk = np.zeros(max_live + 2, dtype=np.uint64)
        hb = hist[b]
        for j in range(b, N, nblocks):
            args_ok = _trace_root(ray_ids[j], starts[j, 0], starts[j, 1], starts[j, 2],
                                  dirs[j, 0], dirs[j, 1], dirs[j, 2], 1.0, seed, start_vol,
                                  fp, fn, fc, order, bbmin, bbmax, vol_mat, vol_scint,
                                  n_mat, mu_mat, refl_mat, rfl_mat, outside,
                                  port_vol, port_pt, port_n, port_tol,
                                  c, ray_eps, par_eps, tol, edge_tol, catch_tol,
                                  max_steps, max_time, split_min, max_live, max_nodes,
                                  spacing, shape, dt, tmax, nb,
                                  Lg, Ltg, hb, us, sp, sd, sf, sk, diag[j])
            if args_ok == 0:
                lv = 0.0
                ltv = 0.0
                for v in range(V):
                    x = Lg[v]
                    if x != 0.0:
                        sumL[b, v] += x
                        sumL2[b, v] += x * x
                        if x > 0.0:
                            hits[b, v] += 1
                        lv += x
                    y = Ltg[v]
                    if y != 0.0:
                        sumLt[b, v] += y
                        ltv += y
                diag[j, _D_OK] = 1.0
                diag[j, _D_LVOL] = lv
                diag[j, _D_LTVOL] = ltv
            else:
                # undo this root's time-histogram deposits: same deterministic path, weight -1
                _trace_root(ray_ids[j], starts[j, 0], starts[j, 1], starts[j, 2],
                            dirs[j, 0], dirs[j, 1], dirs[j, 2], -1.0, seed, start_vol,
                            fp, fn, fc, order, bbmin, bbmax, vol_mat, vol_scint,
                            n_mat, mu_mat, refl_mat, rfl_mat, outside,
                            port_vol, port_pt, port_n, port_tol,
                            c, ray_eps, par_eps, tol, edge_tol, catch_tol,
                            max_steps, max_time, split_min, max_live, max_nodes,
                            spacing, shape, dt, tmax, nb,
                            Lg, Ltg, hb, us, sp, sd, sf, sk, diag[j])
                for k in range(ndiag):
                    diag[j, k] = 0.0
                diag[j, _D_FAIL] = args_ok
            for v in range(V):
                Lg[v] = 0.0
                Ltg[v] = 0.0
    return sumL, sumL2, sumLt, hits, hist, diag


# ---------------------------------------------------------------------------
# Python driver
# ---------------------------------------------------------------------------
class NumbaAdjointField:
    """Reverse tracer for one readout channel of one scene.

    Parameters
    ----------
    scene : Scene
        Convex volumes only; the scored volume must be named ``scintillator``
        and is assumed to fill the box [0, cell] (as in the Stage-6 scenes).
    terminal_volume : str
        Name of the volume the roots start in (``guide0_terminal``).
    endpoint, outward, lateral, vertical : array (3,)
        Centre of the readout face, its outward normal and two in-face axes.
    core_w, core_h : float
        Readout face size along ``lateral`` and ``vertical`` (cm).
    cell : array (3,), grid_shape : (nx, ny, nz)
        Scintillator box and voxel grid.
    cfg : dict
        Tracking settings with the repo's keys: split_min_child_weight,
        max_live_branches, max_branch_nodes, max_steps, max_flight_time_ns,
        start_epsilon_cm, dt_ns, tmax_ns.
    """

    def __init__(self, scene, *, terminal_volume, endpoint, outward, lateral, vertical,
                 core_w, core_h, cell, grid_shape, cfg, c_cm_ns=29.9792458,
                 ray_epsilon=1e-9, parallel_epsilon=1e-12, tolerance=1e-9, edge_tolerance=1e-9,
                 catch_tolerance=1e-8, bbox_pad=1e-6):
        vols = list(scene.volumes)
        bad = [v.name for v in vols if not isinstance(v, ConvexPolyhedronVolume)]
        if bad:
            raise TypeError(f"only convex polyhedra are supported: {bad}")
        names = [v.name for v in vols]
        if terminal_volume not in names:
            raise ValueError(f"no volume named {terminal_volume!r}")
        if names.count("scintillator") != 1:
            raise ValueError("exactly one volume must be named 'scintillator'")
        self.scene, self.cfg = scene, dict(cfg)
        self.endpoint, self.outward = np.asarray(endpoint, float), np.asarray(outward, float)
        self.lateral, self.vertical = np.asarray(lateral, float), np.asarray(vertical, float)
        self.inward = -self.outward
        self.core_w, self.core_h = float(core_w), float(core_h)
        self.cell = np.asarray(cell, float)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.spacing = self.cell / np.asarray(self.grid_shape, float)
        self.nb = int(math.ceil(float(cfg["tmax_ns"]) / float(cfg["dt_ns"])))
        self.c = float(c_cm_ns)
        self.tols = (float(ray_epsilon), float(parallel_epsilon), float(tolerance),
                     float(edge_tolerance), float(catch_tolerance))

        # the scored box must sit at the origin, as the voxel indexing assumes
        sc = vols[names.index("scintillator")]
        lo, hi = sc.vertices.min(axis=0), sc.vertices.max(axis=0)
        if not (np.allclose(lo, 0.0, atol=1e-9) and np.allclose(hi, self.cell, atol=1e-9)):
            raise ValueError(f"scintillator box {lo}..{hi} is not [0, cell={self.cell}]")

        # materials: deduplicated by == (the repo compares materials with ==)
        mats = []

        def midx(m):
            for i, x in enumerate(mats):
                if x is m or x == m:
                    return i
            mats.append(m)
            return len(mats) - 1

        V = len(vols)
        F = max(len(v.face_points) for v in vols)
        fp, fn = np.zeros((V, F, 3)), np.zeros((V, F, 3))
        fc = np.zeros(V, np.int64)
        vm = np.zeros(V, np.int64)
        bbmin, bbmax = np.zeros((V, 3)), np.zeros((V, 3))
        for i, v in enumerate(vols):
            k = len(v.face_points)
            fp[i, :k], fn[i, :k], fc[i] = np.asarray(v.face_points, float), np.asarray(v.face_normals, float), k
            vm[i] = midx(v.material)
            bbmin[i] = v.vertices.min(axis=0) - bbox_pad
            bbmax[i] = v.vertices.max(axis=0) + bbox_pad
        outside = midx(scene.outside_material)
        for m in mats:
            if isinstance(m, ReflectiveMaterial) and float(m.specular_fraction) < 1.0 - 1e-12:
                raise NotImplementedError("Lambertian reflector parts are not supported by the weighted tracer")
        pr = np.array([v.priority for v in vols])
        order = np.array(sorted(range(V), key=lambda i: -pr[i]), np.int64)
        ports = list(getattr(scene, "readout_ports", ()) or ())
        if not ports:
            raise ValueError("scene has no readout ports")
        self.port_channels = [int(p.channel) for p in ports]
        self.arrays = dict(
            fp=fp, fn=fn, fc=fc, order=order, bbmin=bbmin, bbmax=bbmax, vol_mat=vm,
            vol_scint=np.array([n == "scintillator" for n in names]),
            n_mat=np.array([float(getattr(m, "refractive_index", 1.0) or 1.0) for m in mats]),
            mu_mat=np.array([float(getattr(m, "absorption_mu", 0.0) or 0.0) for m in mats]),
            refl_mat=np.array([isinstance(m, ReflectiveMaterial) for m in mats]),
            rfl_mat=np.array([float(m.reflectance) if isinstance(m, ReflectiveMaterial) else 0.0 for m in mats]),
            outside=int(outside),
            port_vol=np.array([names.index(p.volume_name) for p in ports], np.int64),
            port_pt=np.array([p.point for p in ports], float),
            port_n=np.array([p.outward_normal for p in ports], float),
            port_tol=np.array([float(p.tolerance) for p in ports], float),
        )
        self.start_vol = names.index(terminal_volume)
        self.ndiag = _D_TERM + _T_PORT0 + len(ports)

    # -- launches ---------------------------------------------------------
    def launch_rays(self, u):
        """Root start points and directions from 4-D uniforms (repo's _initial_reverse_ray)."""
        u = np.atleast_2d(np.asarray(u, float))
        edge = 1.0 - 2.0e-10
        a = (u[:, 0] - 0.5) * self.core_w * edge
        b = (u[:, 1] - 0.5) * self.core_h * edge
        start = self.endpoint + a[:, None] * self.lateral + b[:, None] * self.vertical
        start = start - float(self.cfg["start_epsilon_cm"]) * self.outward
        mu = np.sqrt(u[:, 2])
        st = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
        phi = 2.0 * np.pi * u[:, 3]
        d = (mu[:, None] * self.inward + (st * np.cos(phi))[:, None] * self.lateral
             + (st * np.sin(phi))[:, None] * self.vertical)
        return start, d / np.linalg.norm(d, axis=1, keepdims=True)

    @staticmethod
    def halton(n, launch_seed, start=0):
        s = qmc.Halton(d=4, scramble=True, seed=int(launch_seed))
        if start:
            s.fast_forward(int(start))
        return s.random(int(n))

    # -- transport --------------------------------------------------------
    def trace(self, u, ray_ids, physics_seed, nblocks=None):
        """Trace roots; returns raw block sums and per-root diagnostics."""
        import numba
        starts, dirs = self.launch_rays(u)
        ids = np.ascontiguousarray(np.asarray(ray_ids, dtype=np.uint64))
        nb_ = int(nblocks or numba.get_num_threads())
        nb_ = max(1, min(nb_, len(ids)))
        a, cf = self.arrays, self.cfg
        rt = _run_roots(
            np.ascontiguousarray(starts), np.ascontiguousarray(dirs), ids, np.uint64(int(physics_seed)),
            self.start_vol, nb_,
            a["fp"], a["fn"], a["fc"], a["order"], a["bbmin"], a["bbmax"], a["vol_mat"], a["vol_scint"],
            a["n_mat"], a["mu_mat"], a["refl_mat"], a["rfl_mat"], a["outside"],
            a["port_vol"], a["port_pt"], a["port_n"], a["port_tol"],
            self.c, *self.tols,
            int(cf["max_steps"]), float(cf["max_flight_time_ns"]), float(cf["split_min_child_weight"]),
            int(cf["max_live_branches"]), int(cf["max_branch_nodes"]),
            self.spacing, np.asarray(self.grid_shape, np.int64), float(cf["dt_ns"]), float(cf["tmax_ns"]),
            self.nb, self.ndiag)
        return rt

    def run(self, n_roots, launch_seed, physics_seed, start=0, nblocks=None):
        """Trace ``n_roots`` Halton roots from prefix index ``start``.

        Returns the repo-compatible ``merged`` dict (see module docstring)
        and the per-root diagnostic array.
        """
        u = self.halton(n_roots, launch_seed, start)
        ids = np.arange(start, start + n_roots, dtype=np.uint64)
        sumL, sumL2, sumLt, hits, hist, diag = self.trace(u, ids, physics_seed, nblocks)
        return self.merge(sumL, sumL2, sumLt, hits, hist, diag, start), diag

    def merge(self, sumL, sumL2, sumLt, hits, hist, diag, start=0):
        shape = self.grid_shape
        ok = diag[:, _D_OK] > 0
        term = Counter()
        for k, name in enumerate(TERMINAL_NAMES):
            s = float(diag[ok, _D_TERM + k].sum())
            if s:
                term[name] = s
        for q, ch in enumerate(self.port_channels):
            s = float(diag[ok, _D_TERM + _T_PORT0 + q].sum())
            if s:
                term[f"READOUT_PORT_{ch}"] += s
        lv = diag[ok, _D_LVOL]
        return dict(
            n=int(ok.sum()), attempted=int(len(diag)), next_id=int(start + len(diag)),
            sumL=sumL.sum(0).reshape(shape), sumL2=sumL2.sum(0).reshape(shape),
            sumLt=sumLt.sum(0).reshape(shape), hist=hist.sum(0).reshape(shape + (self.nb,)),
            hit_roots=hits.sum(0).reshape(shape),
            sumLvol=float(lv.sum()), sumLvol2=float((lv * lv).sum()), sumLtvol=float(diag[ok, _D_LTVOL].sum()),
            reached_scint=int(diag[ok, _D_REACHED].sum()),
            bulk_absorbed_weight=float(diag[ok, _D_BULK].sum()),
            terminal_weight=term,
            splits=int(diag[ok, _D_SPLITS].sum()), analog_fallbacks=int(diag[ok, _D_FALLBACKS].sum()),
            nodes_created=int(diag[ok, _D_NODES].sum()),
            max_live=int(diag[ok, _D_MAXLIVE].max()) if ok.any() else 0,
            boundary_events=int(diag[ok, _D_EVENTS].sum()),
            failed=int((~ok).sum()),
            failure_codes={FAIL_NAMES[int(c)]: int((diag[:, _D_FAIL] == c).sum())
                           for c in np.unique(diag[~ok, _D_FAIL])} if (~ok).any() else {},
        )


def from_stage6_cfg(cfg, **kw):
    """Build the tracer from a Stage-6 cfg dict (as made by the atlas's ``_geometry_cfg``).

    Needs ``phinder_bridge/scripts`` on ``sys.path``. Mirrors the repo's
    ``phinder_stage6_adjoint_field._worker_init``.
    """
    import phinder_stage6_adjoint_prototype as proto
    scene, paths, core_w, core_h = proto._build_from_cfg(cfg)
    ch = int(cfg["channel"])
    pts, tans = paths[ch]
    Z = float(cfg["cell_cm"][2])
    th = float(cfg["guide_thickness_cm"])
    zc = Z + 0.5 * th if ch == 0 else -0.5 * th
    endpoint = np.array([pts[-1, 0], pts[-1, 1], zc], float)
    tf = np.asarray(tans[-1], float)
    tf = tf / np.linalg.norm(tf)
    outward = np.array([tf[0], tf[1], 0.0])
    lateral = np.array([-tf[1], tf[0], 0.0])
    return NumbaAdjointField(scene, terminal_volume=f"guide{ch}_terminal", endpoint=endpoint, outward=outward,
                             lateral=lateral, vertical=np.array([0.0, 0.0, 1.0]), core_w=core_w, core_h=core_h,
                             cell=cfg["cell_cm"], grid_shape=cfg["grid_shape"], cfg=cfg, **kw)
