"""Test scenes and photon sources used by validate_numba_engine.py and the tests."""
from __future__ import annotations

import numpy as np

from Photon_Simulator import AIR, CLAD, CORE, BoxVolume, Scene
from Photon_Simulator.config.defaults import DX, EJ200, EJ204
from Photon_Simulator.core.detectorSetup.material import Material, ReflectiveMaterial
from Photon_Simulator.core.detectorSetup.readout import ReadoutPort
from Photon_Simulator.core.detectorSetup.referenceScene import build_reference_scene, make_source_batch
from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import ConvexPolyhedronVolume
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch
from Photon_Simulator.core.physics.physics import ParticleInteraction

# Design 0 of fast_engine_for_lorenzo/studies/v2/configs.npz: [bottom_w, bottom_h, depth, taper, x_shear, fiber_t]
V2_THETA = (3.5517564015462995, 7.656830024905503, 1.7254351805895567,
            0.6187300961464643, 0.0839672802016139, 0.19017563592642545)


def v2_scene(theta=V2_THETA):
    """Frustum EJ-200 scintillator + fibre clad + core (same as sim_v2.build_scene_v2)."""
    bw, bh, d, taper, shear, ft = theta
    tw = taper * bw
    tx0 = 0.5 * (1.0 - shear) * (bw - tw)
    tx1 = tx0 + tw
    v = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0],
                  [tx0, 0, d], [tx1, 0, d], [tx1, bh, d], [tx0, bh, d]], dtype=float)
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    fy1 = bh + 0.2
    return Scene([
        ConvexPolyhedronVolume("scintillator", EJ200, v, faces, priority=0),
        BoxVolume("fiber_clad", CLAD, [tx0, 0, d], [tx1, fy1, d + ft], priority=10),
        BoxVolume("fiber_core", CORE, [tx0, 0, d + 0.001], [tx1, fy1, d + ft - 0.001], priority=20),
    ], outside_material=AIR, scintillator_material=EJ200)


def v2_photons(n, seed=3, theta=V2_THETA):
    """Scintillation photons from one MIP track through the middle of the v2 scintillator."""
    bw, bh, d = theta[0], theta[1], theta[2]
    inter = ParticleInteraction(v2_scene(theta))
    rng = np.random.default_rng(seed)
    dedx = 2.0 * EJ200.density_g_cm3
    length = min(1.2 / dedx, 0.95 * d)
    ly = 1.2 * n / (dedx * length)
    pts, t0 = [], []
    while sum(len(p) for p in pts) < n:
        p, t = inter.generate_photon_source(np.array([0.5 * bw, 0.5 * bh, 1e-4]), np.array([0, 0, 1.0]),
                                            length, DX, 0.998, dedx, ly, rng)
        pts.append(p); t0.append(t)
    pts = np.concatenate(pts)[:n]; t0 = np.concatenate(t0)[:n]
    dirs, times = inter.photon_generator(pts, t0, EJ200.rise_time_ns, EJ200.decay_time_ns, rng)
    return PhotonBatch(pts, dirs, times, np.full(n, -1, dtype=np.int32))


def lambertian_scene(specular_fraction=0.3):
    """Reference scene with a partly diffuse wrap (engine A cannot run this)."""
    wrap = ReflectiveMaterial(name="diffuse wrap", refractive_index=None, absorption_mu=None,
                              reflectance=0.95, specular_fraction=specular_fraction)
    return Scene([
        BoxVolume("reflector", wrap, [-0.1, -0.1, -0.1], [1.1, 5.1, 0.6], priority=0),
        BoxVolume("scintillator", EJ204, [0, 0, 0], [1, 3, 0.5], priority=10),
        BoxVolume("guide_terminal", CORE, [0, 3, 0], [1, 5, 0.5], priority=10),
    ], AIR, EJ204, readout_ports=[ReadoutPort(0, "guide_terminal", (0, 5, 0), (0, 1, 0))])


def air_gap_scene(with_air_volume=True):
    """Scintillator and guide separated by 1 mm; optionally an explicit air box in the gap."""
    vols = [BoxVolume("scintillator", EJ204, [0, 0, 0], [1, 3, 0.5], priority=10),
            BoxVolume("guide", CORE, [0, 3.1, 0], [1, 5, 0.5], priority=10)]
    if with_air_volume:
        vols.append(BoxVolume("gap_air", AIR, [0, 3, 0], [1, 3.1, 0.5], priority=5))
    return Scene(vols, AIR, EJ204, readout_ports=[ReadoutPort(0, "guide", (0, 5, 0), (0, 1, 0))])


def deterministic_scene():
    """No randomness affects the outcome: mu = 0, one refractive index, perfect specular mirror.

    Every photon then follows a fixed path, so engines can be compared photon by photon.
    """
    clear = Material("clear", refractive_index=1.5, absorption_mu=0.0)
    mirror = ReflectiveMaterial(name="mirror", refractive_index=None, absorption_mu=None,
                                reflectance=1.0, specular_fraction=1.0)
    return Scene([
        BoxVolume("mirror", mirror, [-0.1, -0.1, -0.1], [1.1, 5.0, 0.6], priority=0),
        BoxVolume("scintillator", clear, [0, 0, 0], [1, 3, 0.5], priority=10),
        BoxVolume("guide", clear, [0, 3, 0], [1, 5, 0.5], priority=10),
    ], AIR, clear, readout_ports=[ReadoutPort(0, "guide", (0, 5, 0), (0, 1, 0))])


def uniform_batch(n, seed, lo=(0.02, 0.02, 0.02), hi=(0.98, 2.98, 0.48)):
    """Isotropic photons uniformly inside a box region (default: the 1x3x0.5 scintillator)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, hi, size=(n, 3))
    v = rng.normal(size=(n, 3))
    return PhotonBatch(pts, v / np.linalg.norm(v, axis=1, keepdims=True), np.zeros(n),
                       np.full(n, -1, dtype=np.int32))


__all__ = ["V2_THETA", "v2_scene", "v2_photons", "lambertian_scene", "air_gap_scene",
           "deterministic_scene", "uniform_batch", "build_reference_scene", "make_source_batch"]
