"""Convert detector scenes and photon containers to JAX arrays."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import ConvexPolyhedronVolume
from Photon_Simulator.core.detectorSetup.material import ReflectiveMaterial
from Photon_Simulator.core.detectorSetup.Shapes.box import BoxVolume
from Photon_Simulator.core.detectorSetup.scene import Scene
from Photon_Simulator.core.photonsEngines.approximateEngine import (
    PhotonBatch, PhotonEvent, PhotonStatus,
)
from .types import (
    Array,
    JaxSceneGeometry,
    JaxMaterialTable,
    JaxPhotonBatchInput,
)


def _material_identity_index(materials: list[object], material: object) -> int:
    """Return an identity-based index, avoiding surprising dataclass equality."""

    for index, candidate in enumerate(materials):
        if candidate is material:
            return index
    materials.append(material)
    return len(materials) - 1


def scene_to_jax(
    scene: Scene,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[JaxSceneGeometry, JaxMaterialTable]:
    """Convert an existing ``Scene`` to fixed-shape JAX arrays.

    Face normals and points are deliberately exposed in the returned pytree.
    Callers may therefore optimize them (or build a parameterized geometry
    from them) without reconstructing Python ``Volume`` objects.
    """

    if not scene.volumes:
        raise ValueError("A JAX scene must contain at least one volume")

    unsupported = [
        volume.name
        for volume in scene.volumes
        if not isinstance(volume, ConvexPolyhedronVolume)
    ]
    if unsupported:
        names = ", ".join(unsupported)
        raise TypeError(
            "JaxApproximateEngine supports convex polyhedra only; "
            f"unsupported volume(s): {names}"
        )

    n_volumes = len(scene.volumes)
    max_faces = max(len(volume.face_points) for volume in scene.volumes)
    face_points = np.zeros((n_volumes, max_faces, 3), dtype=np.float64)
    face_normals = np.zeros_like(face_points)
    face_mask = np.zeros((n_volumes, max_faces), dtype=bool)

    priorities = np.empty(n_volumes, dtype=np.int32)
    scintillator_mask = np.zeros(n_volumes, dtype=bool)
    catch_axes = np.full(n_volumes, -1, dtype=np.int32)
    catch_ends = np.zeros(n_volumes, dtype=np.float64)

    materials: list[object] = []
    material_ids = np.empty(n_volumes, dtype=np.int32)

    for volume_index, volume in enumerate(scene.volumes):
        count = len(volume.face_points)
        face_points[volume_index, :count] = np.asarray(volume.face_points)
        face_normals[volume_index, :count] = np.asarray(volume.face_normals)
        face_mask[volume_index, :count] = True
        priorities[volume_index] = int(volume.priority)
        material_ids[volume_index] = _material_identity_index(
            materials, volume.material
        )
        scintillator_mask[volume_index] = volume.name == "scintillator"

        if not getattr(scene, "readout_ports", ()) and volume.name == "fiber_core" and isinstance(volume, BoxVolume):
            catch_axes[volume_index] = 1
            catch_ends[volume_index] = float(np.max(volume.vertices[:, 1]))

    ports = list(getattr(scene, "readout_ports", ()))
    port_ids, port_points, port_normals, port_tolerances = [], [], [], []
    for port in ports:
        port_ids.append(next(i for i, v in enumerate(scene.volumes) if v.name == port.volume_name))
        port_points.append(port.point)
        port_normals.append(port.outward_normal)
        port_tolerances.append(port.tolerance)
    if not ports:
        for i, axis in enumerate(catch_axes):
            if axis >= 0:
                point, normal = np.zeros(3), np.zeros(3)
                point[axis], normal[axis] = catch_ends[i], 1.0
                port_ids.append(i)
                port_points.append(point)
                port_normals.append(normal)
                port_tolerances.append(0.0)
    # A disabled plane keeps reductions well-defined for scenes with no sensor.
    if not port_ids:
        port_ids, port_points, port_normals, port_tolerances = [-1], [[0., 0., 0.]], [[0., 0., 0.]], [0.]

    outside_material_id = _material_identity_index(materials, scene.outside_material)

    refractive_indices = []
    absorption_mu = []
    reflective = []
    reflectance = []
    specular_fraction = []

    for material in materials:
        is_reflective = isinstance(material, ReflectiveMaterial)
        n = getattr(material, "refractive_index", None)
        mu = getattr(material, "absorption_mu", None)
        refractive_indices.append(1.0 if n is None else float(n))
        absorption_mu.append(0.0 if mu is None else float(mu))
        reflective.append(is_reflective)
        reflectance.append(
            float(getattr(material, "reflectance", 0.0)) if is_reflective else 0.0
        )
        specular_fraction.append(
            float(getattr(material, "specular_fraction", 1.0)) if is_reflective else 1.0
        )

    geometry = JaxSceneGeometry(
        face_points=jnp.asarray(face_points, dtype=dtype),
        face_normals=jnp.asarray(face_normals, dtype=dtype),
        face_mask=jnp.asarray(face_mask),
        priorities=jnp.asarray(priorities),
        material_ids=jnp.asarray(material_ids),
        scintillator_mask=jnp.asarray(scintillator_mask),
        catch_axes=jnp.asarray(catch_axes),
        catch_ends=jnp.asarray(catch_ends, dtype=dtype),
        port_volume_ids=jnp.asarray(port_ids, dtype=jnp.int32),
        port_points=jnp.asarray(port_points, dtype=dtype),
        port_normals=jnp.asarray(port_normals, dtype=dtype),
        port_tolerances=jnp.asarray(port_tolerances, dtype=dtype),
    )
    material_table = JaxMaterialTable(
        refractive_indices=jnp.asarray(refractive_indices, dtype=dtype),
        absorption_mu=jnp.asarray(absorption_mu, dtype=dtype),
        reflective=jnp.asarray(reflective),
        reflectance=jnp.asarray(reflectance, dtype=dtype),
        specular_fraction=jnp.asarray(specular_fraction, dtype=dtype),
        outside_material_id=jnp.asarray(outside_material_id, dtype=jnp.int32),
    )
    return geometry, material_table


def as_jax_photon_batch(
    photons: JaxPhotonBatchInput | PhotonBatch | PhotonEvent,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> JaxPhotonBatchInput:
    """Convert simulator photon containers to a JAX-native batch."""

    if isinstance(photons, JaxPhotonBatchInput):
        return JaxPhotonBatchInput(
            interaction_points=jnp.asarray(photons.interaction_points, dtype=dtype),
            directions=jnp.asarray(photons.directions, dtype=dtype),
            times=jnp.asarray(photons.times, dtype=dtype),
            statuses=jnp.asarray(photons.statuses, dtype=jnp.int32),
        )

    if isinstance(photons, PhotonBatch):
        return JaxPhotonBatchInput(
            interaction_points=jnp.asarray(photons.interaction_points, dtype=dtype),
            directions=jnp.asarray(photons.directions, dtype=dtype),
            times=jnp.asarray(photons.times, dtype=dtype),
            statuses=jnp.asarray(photons.statuses, dtype=jnp.int32),
        )

    if isinstance(photons, PhotonEvent):
        return JaxPhotonBatchInput(
            interaction_points=jnp.asarray(
                np.asarray(photons.interaction_point)[None, :], dtype=dtype
            ),
            directions=jnp.asarray(np.asarray(photons.direction)[None, :], dtype=dtype),
            times=jnp.asarray([photons.time], dtype=dtype),
            statuses=jnp.asarray([int(photons.status)], dtype=jnp.int32),
        )

    raise TypeError("photons must be JaxPhotonBatchInput, PhotonBatch, or PhotonEvent")
