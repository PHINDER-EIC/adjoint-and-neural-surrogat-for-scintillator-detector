"""JAX array containers, tracking configuration, and photon status constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np

from Photon_Simulator.core.photonsEngines.approximateEngine import (
    PhotonBatch, PhotonEvent, PhotonStatus,
)

Array = jax.Array

#Compatibility with the existing ApproximateEngine status codes

RUNNING_TRANSMITTED = int(PhotonStatus.RUNNING_TRANSMITTED)
RUNNING_INSIDE = int(PhotonStatus.RUNNING_INSIDE)
ABSORBED = int(PhotonStatus.ABSORBED)
LOST = int(PhotonStatus.LOST)
CAUGHT = int(PhotonStatus.CAUGHT)
ABSORBED_AT_EDGES = int(PhotonStatus.ABSORBED_AT_EDGES)
ABSORBED_REFL_AT_SCINTILLATOR = int(PhotonStatus.ABSORBED_REFL_AT_Scintillator)
ABSORBED_REFL_AT_GUIDE = int(PhotonStatus.ABSORBED_REFL_AT_Guide)
ABSORBED_LAMBDA_AT_SCINTILLATOR = int(PhotonStatus.ABSORBED_LAMBDA_AT_Scintillator)
ABSORBED_LAMBDA_AT_GUIDE = int(PhotonStatus.ABSORBED_LAMBDA_AT_Guide)
LOST_AT_SCINTILLATOR = int(PhotonStatus.LOST_AT_Scintillator)
LOST_AT_GUIDE = int(PhotonStatus.LOST_AT_Guide)


@dataclass(frozen=True)
class JaxTrackingConfig:
    """Static configuration for JAX photon transport.

    `max_steps` is intentionally finite because a JAX loop has a compiled
    upper bound.  Photons still active at that bound are returned with
    `truncated=True` rather than raising from inside compiled code
    """

    max_steps: int = 1200
    c_cm_ns: float = 29.9792458
    ray_epsilon: float = 1.0e-6
    containment_tolerance: float = 1.0e-6
    edge_tolerance: float = 2.0e-6
    catch_tolerance: float = 2.0e-6
    soft_interactions: bool = False
    interaction_temperature: float = 0.05
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.c_cm_ns <= 0.0:
            raise ValueError("c_cm_ns must be positive")
        if self.ray_epsilon <= 0.0:
            raise ValueError("ray_epsilon must be positive")
        if self.interaction_temperature <= 0.0:
            raise ValueError("interaction_temperature must be positive")


class JaxSceneGeometry(NamedTuple):
    """Padded convex-polyhedron representation of a detector scene.

    Volumes follow the order of `scene.volumes`. Each volume is described
    by its face planes, defined by points and outward-facing normals.
    Arrays are padded to `max_faces`, the largest face count among all
    volumes, to provide fixed shapes for JAX JIT compilation.

    Attributes
    ----------
    face_points : Array, shape (n_volumes, max_faces, 3)
        A point on each face plane, expressed in global scene coordinates
        in centimetres. Entries corresponding to padded faces are ignored.
    face_normals : Array, shape (n_volumes, max_faces, 3)
        Outward-facing normals defining the face planes. A point lies
        inside a volume when, for every valid face,
        `dot(point - face_point, face_normal) <= tolerance`,
        where `tolerance` is the configured containment tolerance.
    face_mask : Array, shape (n_volumes, max_faces)
        Boolean mask identifying valid faces. True entries correspond
        to actual faces; False entries mark ignored padding.
    priorities : Array, shape (n_volumes,)
        Integer priorities used to resolve overlapping volumes. Among
        volumes containing a point, the one with the highest priority
        is selected. Ties select the first volume in scene order.
    material_ids : Array, shape (n_volumes,)
        Integer indices associating each volume with its material
        properties in the corresponding `JaxMaterialTable`.
    scintillator_mask : Array, shape (n_volumes,)
        Boolean mask identifying scintillator volumes for the assignment
        of material-specific transport statuses. `scene_to_jax` sets
        this mask to True for volumes named "scintillator".
    catch_axes : Array, shape (n_volumes,)
        Integer coordinate axis normal to each photon collection plane:
        0 for x, 1 for y, and 2 for z. A value of -1 disables collection
        for that volume. `scene_to_jax` assigns axis 1 to volumes named
        "fiber_core" and axis 0 to volumes named "fiber_core_x".
    catch_ends : Array, shape (n_volumes,)
        Collection-plane coordinate, in centimetres, along the axis
        specified by ``catch_axes``. A photon reaching a boundary from
        inside the associated volume is caught if its coordinate is
        within the configured catch tolerance of this value and its
        direction has a positive component along the collection axis.
        Ignored when ``catch_axes`` is -1. ``scene_to_jax`` uses the
        maximum vertex coordinate along the selected axis.

    Notes
    -----
    All fields are JAX arrays, and this NamedTuple is a JAX pytree.
    Face points and normals can therefore be supplied as continuous
    geometry parameters to JAX transformations. Masks, material indices,
    priorities, and collection-axis indices encode discrete choices.
    """

    face_points: Array  # (n_volumes, max_faces, 3)
    face_normals: Array  # (n_volumes, max_faces, 3)
    face_mask: Array  # (n_volumes, max_faces)
    priorities: Array  # (n_volumes,)
    material_ids: Array  # (n_volumes,)
    scintillator_mask: Array  # (n_volumes,)
    catch_axes: Array  # (n_volumes,), -1 if the volume is not a sensor
    catch_ends: Array  # (n_volumes,)
    # General readout planes, including rotated ports and multiple ports/volume.
    port_volume_ids: Array
    port_points: Array
    port_normals: Array
    port_tolerances: Array


class JaxMaterialTable(NamedTuple):
    """Continuous and discrete material properties used by transport."""

    refractive_indices: Array
    absorption_mu: Array
    reflective: Array
    reflectance: Array
    specular_fraction: Array
    outside_material_id: Array


class JaxPhotonBatchInput(NamedTuple):
    """ photon batch accepted by :func:`transport_photons`."""

    interaction_points: Array  # (n_photons, 3)
    directions: Array  # (n_photons, 3)
    times: Array  # (n_photons,)
    statuses: Array  # (n_photons,)


class JaxPhotonEvent(NamedTuple):
    """Single-photon result whose fields remain JAX values."""

    interaction_point: Array
    direction: Array
    status: Array
    time: Array
    truncated: Array
    steps: Array


class JaxPhotonBatchOutput(NamedTuple):
    """Final JAX arrays and per-photon truncation information."""

    interaction_points: Array
    directions: Array
    times: Array
    statuses: Array
    active: Array
    truncated: Array
    steps: Array

    def to_photon_batch(self) -> PhotonBatch:
        """Copy the result to the existing NumPy ``PhotonBatch`` type."""

        return PhotonBatch(
            interaction_points=np.asarray(self.interaction_points),
            directions=np.asarray(self.directions),
            times=np.asarray(self.times),
            statuses=np.asarray(self.statuses),
        )


class JaxPhotonTrackOutput(NamedTuple):
    """Batch result plus every state, including the initial one."""

    output: JaxPhotonBatchOutput
    interaction_points: Array
    directions: Array
    times: Array
    statuses: Array

    def to_tracks(self) -> list[list[PhotonEvent]]:
        """Convert the padded history to the tracks used by visualizers."""
        points = np.asarray(self.interaction_points)
        directions = np.asarray(self.directions)
        times = np.asarray(self.times)
        statuses = np.asarray(self.statuses)
        steps = np.asarray(self.output.steps)
        return [
            [
                PhotonEvent(
                    interaction_point=points[step, photon_index],
                    direction=directions[step, photon_index],
                    status=PhotonStatus(int(statuses[step, photon_index])),
                    time=float(times[step, photon_index]),
                )
                for step in range(int(step_count) + 1)
            ]
            for photon_index, step_count in enumerate(steps)
        ]


class _TransportState(NamedTuple):
    points: Array
    directions: Array
    times: Array
    statuses: Array
    active: Array
    steps: Array
    key: Array
