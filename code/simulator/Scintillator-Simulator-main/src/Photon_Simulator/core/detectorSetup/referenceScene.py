"""Small forward-transport fixture in cm, with an explicit terminal readout."""
import numpy as np

from Photon_Simulator.config.defaults import AIR, EJ204, CORE, GENERIC_REFLECTIVE
from .Shapes.box import BoxVolume
from .scene import Scene
from .readout import ReadoutPort


def build_reference_scene():
    # Scintillator 1 x 3 x 0.5 cm, followed by a 2-cm guide. An outer
    # reflector tests both Fresnel transport and the reflector's survival RNG.
    return Scene([
        BoxVolume("reflector", GENERIC_REFLECTIVE, [-0.1, -0.1, -0.1], [1.1, 5.1, 0.6], priority=0),
        BoxVolume("scintillator", EJ204, [0, 0, 0], [1, 3, 0.5], priority=10),
        BoxVolume("guide_terminal", CORE, [0, 3, 0], [1, 5, 0.5], priority=10),
    ], AIR, EJ204, readout_ports=[ReadoutPort(0, "guide_terminal", (0, 5, 0), (0, 1, 0))])


def make_source_batch(n, seed, source="distributed"):
    from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatch, PhotonStatus
    rng = np.random.default_rng(seed)
    fixed = {"center": [0.5, 1.5, 0.25], "near_readout": [0.5, 2.85, 0.25],
             "far_corner": [0.08, 0.15, 0.08]}
    if source == "distributed":
        points = rng.uniform([0.02, 0.02, 0.02], [0.98, 2.98, 0.48], size=(n, 3))
    else:
        points = np.tile(fixed[source], (n, 1))
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    radial = np.sqrt(1 - z*z)
    directions = np.column_stack((radial*np.cos(phi), radial*np.sin(phi), z))
    return PhotonBatch(points, directions, np.zeros(n),
                       np.full(n, int(PhotonStatus.RUNNING_INSIDE), dtype=np.int32))
