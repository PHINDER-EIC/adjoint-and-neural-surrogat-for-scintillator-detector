"""Stochastic transport steps and compiled batch tracking loops."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .types import (
    Array,
    JaxTrackingConfig,
    JaxSceneGeometry,
    JaxMaterialTable,
    JaxPhotonBatchInput,
    JaxPhotonBatchOutput,
    JaxPhotonTrackOutput,
    _TransportState,
)
from .types import (
    RUNNING_TRANSMITTED,
    RUNNING_INSIDE,
    ABSORBED,
    LOST,
    CAUGHT,
    ABSORBED_AT_EDGES,
    ABSORBED_REFL_AT_SCINTILLATOR,
    ABSORBED_REFL_AT_GUIDE,
    ABSORBED_LAMBDA_AT_SCINTILLATOR,
    ABSORBED_LAMBDA_AT_GUIDE,
    LOST_AT_SCINTILLATOR,
    LOST_AT_GUIDE,
)
from .geometry import _boundary_context, _hits_edge, _caught_mask
from .optics import (
    _unit,
    _oriented_normal,
    _reflected_direction,
    _transmitted_direction,
    _fresnel_reflectance,
    _cosine_hemisphere_direction,
)


def _transport_step(
    state: _TransportState,
    geometry: JaxSceneGeometry,
    materials: JaxMaterialTable,
    config: JaxTrackingConfig,
) -> _TransportState:
    directions = _unit(state.directions)
    (
        boundary_distance,
        hit_points,
        raw_normal,
        valid_hit,
        from_volume,
        to_volume,
    ) = _boundary_context(state.points, directions, geometry, config)

    valid_event = state.active & valid_hit & (from_volume >= 0)
    no_hit = state.active & (~valid_hit | (from_volume < 0))

    safe_from = jnp.clip(from_volume, 0, geometry.material_ids.shape[0] - 1)
    from_material = geometry.material_ids[safe_from]
    n_from = materials.refractive_indices[from_material]
    absorption_mu = materials.absorption_mu[from_material]
    from_scintillator = geometry.scintillator_mask[safe_from]

    safe_to = jnp.clip(to_volume, 0, geometry.material_ids.shape[0] - 1)
    to_material = jnp.where(
        to_volume >= 0,
        geometry.material_ids[safe_to],
        materials.outside_material_id,
    )
    n_to = materials.refractive_indices[to_material]

    next_key, sample_key = jax.random.split(state.key)
    samples = jax.random.uniform(
        sample_key,
        shape=(state.points.shape[0], 6),
        dtype=(jnp.float32 if config.mixed_precision else state.points.dtype),
        minval=jnp.finfo(
            jnp.float32 if config.mixed_precision else state.points.dtype
        ).tiny,
        maxval=1.0,
    )

    absorption_distance = jnp.where(
        absorption_mu > 0.0,
        -jnp.log(samples[:, 0]) / jnp.maximum(absorption_mu, 1.0e-30),
        jnp.inf,
    )
    bulk_absorbed = valid_event & (absorption_distance < boundary_distance)
    boundary_event = valid_event & (~bulk_absorbed)
    catch_event = (
        boundary_event
        & _caught_mask(
            hit_points,
            directions,
            from_volume,
            geometry,
            config.catch_tolerance,
        )
    )
    # Match the patched CPU engine: readout wins over edge absorption;
    # artificial segmentation inside the same material is transparent.
    same_material = (to_volume >= 0) & (from_material == to_material)
    edge_absorbed = boundary_event & (~catch_event) & (~same_material) & _hits_edge(
        hit_points, from_volume, geometry, config.edge_tolerance)
    interface_event = boundary_event & (~edge_absorbed) & (~catch_event)

    normals = _oriented_normal(directions, raw_normal)
    reflected = _reflected_direction(directions, normals)
    can_transmit, transmitted = _transmitted_direction(
        directions, normals, n_from, n_to
    )
    fresnel_r = _fresnel_reflectance(directions, normals, n_from, n_to)

    reflective_interface = interface_event & materials.reflective[to_material]
    dielectric_interface = interface_event & (~materials.reflective[to_material])

    reflector_survives = samples[:, 1] < materials.reflectance[to_material]
    reflector_specular = samples[:, 2] <= materials.specular_fraction[to_material]
    diffuse = _cosine_hemisphere_direction(normals, samples[:, 3], samples[:, 4])
    reflector_direction = jnp.where(reflector_specular[:, None], reflected, diffuse)
    reflector_continues = reflective_interface & reflector_survives
    reflector_absorbed = reflective_interface & (~reflector_survives)

    hard_fresnel_reflection = (samples[:, 5] < fresnel_r) | (~can_transmit)
    dielectric_reflected = dielectric_interface & hard_fresnel_reflection
    dielectric_transmitted = dielectric_interface & (~hard_fresnel_reflection)
    dielectric_lost = dielectric_transmitted & (to_volume < 0)
    dielectric_transmitted_inside = dielectric_transmitted & (to_volume >= 0)

    if config.soft_interactions:
        reflection_weight = jax.nn.sigmoid(
            (fresnel_r - samples[:, 5]) / config.interaction_temperature
        )
        relaxed_direction = _unit(
            reflection_weight[:, None] * reflected
            + (1.0 - reflection_weight)[:, None] * transmitted
        )
        dielectric_direction = relaxed_direction
    else:
        dielectric_direction = jnp.where(
            hard_fresnel_reflection[:, None], reflected, transmitted
        )

    continues = (
        reflector_continues | dielectric_reflected | dielectric_transmitted_inside
    )

    traveled_distance = jnp.where(bulk_absorbed, absorption_distance, boundary_distance)
    event_points = jnp.where(
        bulk_absorbed[:, None],
        state.points + absorption_distance[:, None] * directions,
        hit_points,
    )
    event_times = state.times + traveled_distance * n_from / config.c_cm_ns

    output_points = jnp.where(valid_event[:, None], event_points, state.points)
    output_times = jnp.where(valid_event, event_times, state.times)
    output_directions = directions
    output_directions = jnp.where(
        reflector_continues[:, None], reflector_direction, output_directions
    )
    output_directions = jnp.where(
        dielectric_interface[:, None], dielectric_direction, output_directions
    )

    statuses = state.statuses
    statuses = jnp.where(no_hit, LOST, statuses)
    statuses = jnp.where(
        bulk_absorbed,
        jnp.where(
            from_scintillator,
            ABSORBED_LAMBDA_AT_SCINTILLATOR,
            ABSORBED_LAMBDA_AT_GUIDE,
        ),
        statuses,
    )
    statuses = jnp.where(edge_absorbed, ABSORBED_AT_EDGES, statuses)
    statuses = jnp.where(catch_event, CAUGHT, statuses)
    statuses = jnp.where(
        reflector_absorbed,
        jnp.where(
            from_scintillator,
            ABSORBED_REFL_AT_SCINTILLATOR,
            ABSORBED_REFL_AT_GUIDE,
        ),
        statuses,
    )
    statuses = jnp.where(
        reflector_continues | dielectric_reflected,
        RUNNING_INSIDE,
        statuses,
    )
    statuses = jnp.where(
        dielectric_transmitted_inside,
        RUNNING_TRANSMITTED,
        statuses,
    )
    statuses = jnp.where(
        dielectric_lost,
        jnp.where(from_scintillator, LOST_AT_SCINTILLATOR, LOST_AT_GUIDE),
        statuses,
    ).astype(jnp.int32)

    return _TransportState(
        points=output_points,
        directions=output_directions,
        times=output_times,
        statuses=statuses,
        active=continues,
        steps=state.steps + state.active.astype(jnp.int32),
        key=next_key,
    )


@partial(jax.jit, static_argnames=("config",))
def transport_photons(
    photons: JaxPhotonBatchInput,
    key: Array,
    geometry: JaxSceneGeometry,
    materials: JaxMaterialTable,
    config: JaxTrackingConfig = JaxTrackingConfig(),
) -> JaxPhotonBatchOutput:
    """Trace a photon batch in one compiled, differentiable JAX program."""

    points = jnp.asarray(photons.interaction_points)
    directions = _unit(jnp.asarray(photons.directions, dtype=points.dtype))
    times = jnp.asarray(photons.times, dtype=points.dtype)
    statuses = jnp.asarray(photons.statuses, dtype=jnp.int32)
    active = (statuses == RUNNING_INSIDE) | (statuses == RUNNING_TRANSMITTED)

    initial_state = _TransportState(
        points=points,
        directions=directions,
        times=times,
        statuses=statuses,
        active=active,
        steps=jnp.zeros_like(statuses, dtype=jnp.int32),
        key=key,
    )

    def body(_index: int, state: _TransportState) -> _TransportState:
        return _transport_step(state, geometry, materials, config)

    final_state = jax.lax.fori_loop(0, config.max_steps, body, initial_state)
    return JaxPhotonBatchOutput(
        interaction_points=final_state.points,
        directions=final_state.directions,
        times=final_state.times,
        statuses=final_state.statuses,
        active=final_state.active,
        truncated=final_state.active,
        steps=final_state.steps,
    )


@partial(jax.jit, static_argnames=("config",))
def transport_photons_dynamic(
    photons: JaxPhotonBatchInput,
    key: Array,
    geometry: JaxSceneGeometry,
    materials: JaxMaterialTable,
    config: JaxTrackingConfig = JaxTrackingConfig(),
) -> JaxPhotonBatchOutput:
    """Trace a batch and stop as soon as every photon is inactive.

    This fast while-loop path is not intended for reverse-mode differentiation.
    """
    points = jnp.asarray(photons.interaction_points)
    directions = _unit(jnp.asarray(photons.directions, dtype=points.dtype))
    times = jnp.asarray(photons.times, dtype=points.dtype)
    statuses = jnp.asarray(photons.statuses, dtype=jnp.int32)
    active = (statuses == RUNNING_INSIDE) | (statuses == RUNNING_TRANSMITTED)
    initial_state = _TransportState(
        points=points,
        directions=directions,
        times=times,
        statuses=statuses,
        active=active,
        steps=jnp.zeros_like(statuses, dtype=jnp.int32),
        key=key,
    )

    def condition(loop_state):
        iteration, state = loop_state
        return (iteration < config.max_steps) & jnp.any(state.active)

    def body(loop_state):
        iteration, state = loop_state
        return iteration + 1, _transport_step(state, geometry, materials, config)

    _, final_state = jax.lax.while_loop(
        condition, body, (jnp.asarray(0, dtype=jnp.int32), initial_state)
    )
    return JaxPhotonBatchOutput(
        interaction_points=final_state.points,
        directions=final_state.directions,
        times=final_state.times,
        statuses=final_state.statuses,
        active=final_state.active,
        truncated=final_state.active,
        steps=final_state.steps,
    )


@partial(jax.jit, static_argnames=("config",))
def transport_photons_with_tracks(
    photons: JaxPhotonBatchInput,
    key: Array,
    geometry: JaxSceneGeometry,
    materials: JaxMaterialTable,
    config: JaxTrackingConfig = JaxTrackingConfig(),
) -> JaxPhotonTrackOutput:
    """Trace a batch and retain every state for visualization/debugging."""
    points = jnp.asarray(photons.interaction_points)
    statuses = jnp.asarray(photons.statuses, dtype=jnp.int32)
    initial_state = _TransportState(
        points=points,
        directions=_unit(jnp.asarray(photons.directions, dtype=points.dtype)),
        times=jnp.asarray(photons.times, dtype=points.dtype),
        statuses=statuses,
        active=(statuses == RUNNING_INSIDE) | (statuses == RUNNING_TRANSMITTED),
        steps=jnp.zeros_like(statuses),
        key=key,
    )

    def body(state: _TransportState, _index: Array):
        next_state = _transport_step(state, geometry, materials, config)
        return next_state, (
            next_state.points,
            next_state.directions,
            next_state.times,
            next_state.statuses,
        )

    final_state, history = jax.lax.scan(
        body, initial_state, jnp.arange(config.max_steps)
    )
    output = JaxPhotonBatchOutput(
        interaction_points=final_state.points,
        directions=final_state.directions,
        times=final_state.times,
        statuses=final_state.statuses,
        active=final_state.active,
        truncated=final_state.active,
        steps=final_state.steps,
    )
    return JaxPhotonTrackOutput(
        output=output,
        interaction_points=jnp.concatenate(
            (initial_state.points[None, ...], history[0]), axis=0
        ),
        directions=jnp.concatenate(
            (initial_state.directions[None, ...], history[1]), axis=0
        ),
        times=jnp.concatenate((initial_state.times[None, ...], history[2]), axis=0),
        statuses=jnp.concatenate(
            (initial_state.statuses[None, ...], history[3]), axis=0
        ),
    )
