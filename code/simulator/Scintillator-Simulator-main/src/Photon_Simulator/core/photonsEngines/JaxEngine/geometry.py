"""Volume containment, boundary intersections, edge and collection tests."""

from __future__ import annotations

import jax.numpy as jnp

from .types import Array, JaxSceneGeometry, JaxTrackingConfig


def _locate_volumes(
    points: Array,
    geometry: JaxSceneGeometry,
    tolerance: float,
) -> Array:
    signed_distance = jnp.sum(
        (points[:, None, None, :] - geometry.face_points[None, :, :, :])
        * geometry.face_normals[None, :, :, :],
        axis=-1,
    )
    inside_each_face = (signed_distance <= tolerance) | (
        ~geometry.face_mask[None, :, :]
    )
    inside = jnp.all(inside_each_face, axis=-1)

    minimum_priority = jnp.iinfo(jnp.int32).min
    scores = jnp.where(
        inside,
        geometry.priorities[None, :],
        minimum_priority,
    )
    indices = jnp.argmax(scores, axis=1).astype(jnp.int32)
    return jnp.where(jnp.any(inside, axis=1), indices, -1)


def _nearest_boundary(
    points: Array,
    directions: Array,
    geometry: JaxSceneGeometry,
    config: JaxTrackingConfig,
) -> tuple[Array, Array, Array, Array]:
    """Return the nearest valid convex-face intersection for every ray."""

    denominator = jnp.einsum("nd,vfd->nvf", directions, geometry.face_normals)
    numerator = jnp.sum(
        (geometry.face_points[None, :, :, :] - points[:, None, None, :])
        * geometry.face_normals[None, :, :, :],
        axis=-1,
    )
    safe_denominator = jnp.where(
        jnp.abs(denominator) > config.ray_epsilon,
        denominator,
        1.0,
    )
    distances = numerator / safe_denominator
    candidates = (
        points[:, None, None, :]
        + distances[:, :, :, None] * directions[:, None, None, :]
    )

    # A point lies on a face polygon of a convex volume exactly when it lies
    # on the inner side of all of that volume's face planes.
    signed_to_all_faces = jnp.sum(
        (candidates[:, :, :, None, :] - geometry.face_points[None, :, None, :, :])
        * geometry.face_normals[None, :, None, :, :],
        axis=-1,
    )
    inside_volume = jnp.all(
        (signed_to_all_faces <= config.containment_tolerance)
        | (~geometry.face_mask[None, :, None, :]),
        axis=-1,
    )

    valid_candidates = (
        geometry.face_mask[None, :, :]
        & (jnp.abs(denominator) > config.ray_epsilon)
        & (distances > config.ray_epsilon)
        & inside_volume
    )
    candidate_distances = jnp.where(valid_candidates, distances, jnp.inf)

    n_photons = points.shape[0]
    flat_distances = candidate_distances.reshape((n_photons, -1))
    flat_index = jnp.argmin(flat_distances, axis=1)
    distance = jnp.take_along_axis(flat_distances, flat_index[:, None], axis=1)[:, 0]
    valid = jnp.isfinite(distance)

    normals = jnp.broadcast_to(
        geometry.face_normals[None, :, :, :],
        (n_photons,) + geometry.face_normals.shape,
    ).reshape((n_photons, -1, 3))
    normal = jnp.take_along_axis(normals, flat_index[:, None, None], axis=1)[:, 0, :]

    safe_distance = jnp.where(valid, distance, 0.0)
    hit_points = points + safe_distance[:, None] * directions
    normal = jnp.where(valid[:, None], normal, jnp.zeros_like(normal))
    return safe_distance, hit_points, normal, valid


def _boundary_context(
    points: Array,
    directions: Array,
    geometry: JaxSceneGeometry,
    config: JaxTrackingConfig,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Mirror ``Scene.first_boundary_hit`` using only JAX operations."""

    distance, hit_points, normal, valid_hit = _nearest_boundary(
        points, directions, geometry, config
    )
    before_points = points + 0.5 * distance[:, None] * directions
    from_volume = _locate_volumes(before_points, geometry, config.containment_tolerance)

    next_distance, _, _, valid_next = _nearest_boundary(
        hit_points, directions, geometry, config
    )
    after_points = hit_points + 0.5 * next_distance[:, None] * directions
    located_after = _locate_volumes(
        after_points, geometry, config.containment_tolerance
    )
    to_volume = jnp.where(valid_hit & valid_next, located_after, -1)
    return distance, hit_points, normal, valid_hit, from_volume, to_volume


def _hits_edge(
    hit_points: Array,
    from_volume: Array,
    geometry: JaxSceneGeometry,
    tolerance: float,
) -> Array:
    safe_volume = jnp.clip(from_volume, 0, geometry.face_points.shape[0] - 1)
    face_points = geometry.face_points[safe_volume]
    face_normals = geometry.face_normals[safe_volume]
    face_mask = geometry.face_mask[safe_volume]
    signed_distance = jnp.sum(
        (hit_points[:, None, :] - face_points) * face_normals,
        axis=-1,
    )
    faces_at_point = face_mask & (jnp.abs(signed_distance) <= tolerance)
    return (from_volume >= 0) & (jnp.sum(faces_at_point, axis=1) >= 2)


def _caught_mask(
    hit_points: Array,
    directions: Array,
    from_volume: Array,
    geometry: JaxSceneGeometry,
    tolerance: float,
) -> Array:
    signed_distance = jnp.sum(
        (hit_points[:, None, :] - geometry.port_points[None, :, :])
        * geometry.port_normals[None, :, :], axis=-1)
    moving_outward = directions @ geometry.port_normals.T > 0.0
    matches = (
        (from_volume[:, None] >= 0)
        & (from_volume[:, None] == geometry.port_volume_ids[None, :])
        & moving_outward
        & (jnp.abs(signed_distance) <= jnp.maximum(tolerance, geometry.port_tolerances)))
    return jnp.any(matches, axis=1)
