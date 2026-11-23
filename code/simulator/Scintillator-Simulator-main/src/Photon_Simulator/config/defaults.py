import os

from Photon_Simulator.core.detectorSetup.material import (
    Material,
    PlasticScintillator,
    ReflectiveMaterial,
)


# ---------------------------------------------------------------------
# SIMULATION PARAMETERS
# ---------------------------------------------------------------------

DE_DX = 2.0  # MeV / (g cm^-2)
DX = 0.01  # cm

N_CORE = max(1, os.cpu_count() // 2)
CHUNK_SIZE = 8000

EPS_NEW_VOLUME_DETECTION = 1e-7

FASTMATH = True


# ---------------------------------------------------------------------
# INTERNAL STATUS CODES
# ---------------------------------------------------------------------

RUNNING_TRANSMITTED = -2
RUNNING_INSIDE = -1
ABSORBED = 0
LOST = 1
CAUGHT = 2


# ---------------------------------------------------------------------
# DEFAULT MATERIALS
# ---------------------------------------------------------------------

AIR = Material(
    name="air",
    refractive_index=1.0,
    absorption_mu=0.0,
)

EJ228 = PlasticScintillator.ej228(absorption_mu=0.02)
EJ200 = PlasticScintillator.ej200()
EJ204 = PlasticScintillator.ej204()
EJ230 = PlasticScintillator.ej230()
EJ260 = PlasticScintillator.ej260()

BC404 = PlasticScintillator.bc404()
BC408 = PlasticScintillator.bc408()
BC420 = PlasticScintillator.bc420()
BC422 = PlasticScintillator.bc422()

SCSN81 = PlasticScintillator.scsn81()

CORE = Material(
    name="fiber_core",
    refractive_index=1.59,
    absorption_mu=0.005,
)

CLAD = Material(
    name="fiber_clad",
    refractive_index=1.49,
    absorption_mu=0.002,
)

# PTFE = ReflectiveMaterial(
#     name="PTFE",
#     refractive_index=None,
#     absorption_mu=None,
#     reflectance=0.98,
#     transmittance=0.0,
#     specular_fraction=0.0,
# )

GENERIC_REFLECTIVE = ReflectiveMaterial(
    name="Reflective (r = 0.98)",
    refractive_index=None,
    absorption_mu=None,
    reflectance=0.98,
    transmittance=0.0,
    specular_fraction=1.0,
)


__all__ = [
    # Simulation parameters
    "DE_DX",
    "DX",
    "N_CORE",
    "CHUNK_SIZE",
    "EPS_NEW_VOLUME_DETECTION",
    "FASTMATH",

    # Internal status codes
    "RUNNING_TRANSMITTED",
    "RUNNING_INSIDE",
    "ABSORBED",
    "LOST",
    "CAUGHT",

    # Default materials
    "AIR",
    "EJ228",
    "EJ200",
    "EJ204",
    "EJ230",
    "EJ260",
    "BC404",
    "BC408",
    "BC420",
    "BC422",
    "SCSN81",
    "CORE",
    "CLAD",
    # "PTFE",
    "GENERIC_REFLECTIVE",
]