"""Vector normalization, reflection, refraction, and Fresnel sampling."""

from __future__ import annotations

import jax.numpy as jnp

from .types import Array


def _unit(vectors: Array, epsilon: float = 1.0e-12) -> Array:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, epsilon)


def _oriented_normal(directions: Array, normals: Array) -> Array:
    points_with_ray = jnp.sum(directions * normals, axis=-1) > 0.0
    return jnp.where(points_with_ray[:, None], -normals, normals)


def _reflected_direction(directions: Array, normals: Array) -> Array:
    projection = jnp.sum(directions * normals, axis=-1, keepdims=True)
    return _unit(directions - 2.0 * projection * normals)


def _transmitted_direction(
    directions: Array,
    normals: Array,
    n_from: Array,
    n_to: Array,
) -> tuple[Array, Array]:
    cos_i = -jnp.sum(directions * normals, axis=-1)
    eta = n_from / jnp.maximum(n_to, 1.0e-12)
    sin_t_squared = eta * eta * jnp.maximum(0.0, 1.0 - cos_i * cos_i)
    can_transmit = sin_t_squared <= 1.0
    cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin_t_squared))
    transmitted = eta[:, None] * directions + (eta * cos_i - cos_t)[:, None] * normals
    return can_transmit, _unit(transmitted)


def _fresnel_reflectance(
    directions: Array,
    normals: Array,
    n_from: Array,
    n_to: Array,
) -> Array:
    cos_i = jnp.clip(-jnp.sum(directions * normals, axis=-1), 0.0, 1.0)
    eta = n_from / jnp.maximum(n_to, 1.0e-12)
    sin_t_squared = eta * eta * jnp.maximum(0.0, 1.0 - cos_i * cos_i)
    total_internal_reflection = sin_t_squared > 1.0
    cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin_t_squared))

    denominator_s = n_from * cos_i + n_to * cos_t
    denominator_p = n_to * cos_i + n_from * cos_t
    r_s = (n_from * cos_i - n_to * cos_t) / jnp.maximum(denominator_s, 1.0e-12)
    r_p = (n_to * cos_i - n_from * cos_t) / jnp.maximum(denominator_p, 1.0e-12)
    reflectance = 0.5 * (r_s * r_s + r_p * r_p)
    return jnp.where(
        total_internal_reflection,
        1.0,
        jnp.clip(reflectance, 0.0, 1.0),
    )


def _cosine_hemisphere_direction(
    normals: Array,
    radial_sample: Array,
    angular_sample: Array,
) -> Array:
    """Cosine-weighted Lambertian sample around an inward normal."""

    radius = jnp.sqrt(radial_sample)
    phi = 2.0 * jnp.pi * angular_sample
    local_x = radius * jnp.cos(phi)
    local_y = radius * jnp.sin(phi)
    local_z = jnp.sqrt(jnp.maximum(0.0, 1.0 - radial_sample))

    helper_z = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), normals.shape)
    helper_x = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), normals.shape)
    helper = jnp.where((jnp.abs(normals[:, 2]) < 0.9)[:, None], helper_z, helper_x)
    tangent = _unit(jnp.cross(helper, normals))
    bitangent = jnp.cross(normals, tangent)
    return _unit(
        local_x[:, None] * tangent
        + local_y[:, None] * bitangent
        + local_z[:, None] * normals
    )
