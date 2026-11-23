"""Active-photon compaction for the JAX transport engine (engine B).

Problem
-------
``JaxApproximateEngine.track_photon_batch`` advances every photon of a batch
in lock-step and keeps looping until the *slowest* photon has finished. Photon
path lengths have a long tail (most photons stop after a few bounces, a few
are trapped by total internal reflection for hundreds or thousands), so most
of the computed steps belong to photons that are already finished. On the v2
scene, only ~2% of the work is useful.

Fix
---
Run the existing compiled loop for a short *segment* (``segment_steps``
bounces), keep only the photons that are still active, pack them into a
smaller batch, and repeat. Batch sizes are rounded up to powers of two
(``min_bucket`` or larger) so JAX compiles one program per size class instead
of one per photon count.

The physics is untouched: each segment calls the engine's own
``transport_photons_dynamic`` with the engine's own geometry, materials and
tolerances. Only the random stream changes (each segment gets
``fold_in(key, segment_index)``), so results agree with the uncompacted engine
statistically, not photon by photon.

Usage
-----
Drop-in for BatchTransport (recommended)::

    from Photon_Simulator.core.photonsEngines.batchTransport import BatchTransport
    from compaction import enable_compaction

    runner = enable_compaction(BatchTransport(scene, engine="jax", seed=42, chunk_size=20000))
    result = runner.run(batch)            # TransportResult, same as before

Direct::

    out = track_photon_batch_compacted(jax_engine, batch, key=0)

Requirements: ``jax.config.update("jax_enable_x64", True)`` before any JAX
array is created if the engine uses float64 geometry (BatchTransport does
this for you).
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from Photon_Simulator.core.photonsEngines.JaxEngine import (
    JaxPhotonBatchInput,
    JaxPhotonBatchOutput,
    as_jax_photon_batch,
    transport_photons_dynamic,
)
from Photon_Simulator.core.photonsEngines.JaxEngine.types import (
    ABSORBED,
    RUNNING_INSIDE,
    RUNNING_TRANSMITTED,
)

__all__ = [
    "track_photon_batch_compacted",
    "CompactedJaxEngine",
    "enable_compaction",
]


def _bucket_size(n_active: int, min_bucket: int) -> int:
    """Smallest power of two >= n_active, but at least ``min_bucket``."""
    return max(min_bucket, 1 << max(0, n_active - 1).bit_length())


def _as_key(key):
    if isinstance(key, (int, np.integer)):
        return jax.random.PRNGKey(int(key))
    return jnp.asarray(key, dtype=jnp.uint32)


@partial(jax.jit, static_argnames=("config",))
def _run_segment(points, directions, times, statuses, steps, idx, key,
                 geometry, materials, config):
    """Advance the photons listed in ``idx`` by up to ``config.max_steps`` bounces.

    ``idx`` has a power-of-two length. Entries equal to ``n`` (the full batch
    size) are padding: they are gathered as a copy of the last photon, marked
    inactive, and their results are dropped when scattering back.
    """
    n = points.shape[0]
    valid = idx < n

    def take(a):
        return jnp.take(a, idx, axis=0, mode="clip")

    segment_input = JaxPhotonBatchInput(
        interaction_points=take(points),
        directions=take(directions),
        times=take(times),
        statuses=jnp.where(valid, take(statuses), ABSORBED).astype(jnp.int32),
    )
    out = transport_photons_dynamic(segment_input, key, geometry, materials, config)

    def put(full, part):
        return full.at[idx].set(part.astype(full.dtype), mode="drop")

    active = jnp.zeros(n, dtype=bool).at[idx].set(out.active & valid, mode="drop")
    return (
        put(points, out.interaction_points),
        put(directions, out.directions),
        put(times, out.times),
        put(statuses, out.statuses),
        steps.at[idx].add(jnp.where(valid, out.steps, 0), mode="drop"),
        active,
    )


def track_photon_batch_compacted(
    engine,
    photons,
    *,
    key=0,
    segment_steps: int = 16,
    min_bucket: int = 256,
    max_steps: int | None = None,
) -> JaxPhotonBatchOutput:
    """Compacted equivalent of ``JaxApproximateEngine.track_photon_batch``.

    Parameters
    ----------
    engine : JaxApproximateEngine
        Supplies geometry, materials, config and dtype. Not modified.
    photons : PhotonBatch | JaxPhotonBatchInput | PhotonEvent
        Same inputs as ``track_photon_batch``.
    key : int | jax PRNG key
        Same meaning as in ``track_photon_batch``. Fixed key + inputs +
        ``segment_steps`` + ``min_bucket`` reproduce the run exactly.
    segment_steps : int
        Bounces per segment before re-packing. Smaller = less wasted work but
        more host round-trips. 8-32 works well; 16 is the default.
    min_bucket : int
        Smallest packed batch size. Avoids many tiny compiled programs.
    max_steps : int | None
        Total bounce budget per photon; defaults to ``engine.config.max_steps``.
        Photons still active after it are returned with ``truncated=True``,
        exactly like the uncompacted engine.

    Returns
    -------
    JaxPhotonBatchOutput
        Same fields and photon order as ``track_photon_batch``.
    """
    if segment_steps <= 0 or min_bucket <= 0:
        raise ValueError("segment_steps and min_bucket must be positive")
    budget = engine.config.max_steps if max_steps is None else int(max_steps)
    if budget <= 0:
        raise ValueError("max_steps must be positive")

    batch = as_jax_photon_batch(photons, dtype=engine.dtype)
    points = jnp.asarray(batch.interaction_points)
    directions = jnp.asarray(batch.directions, dtype=points.dtype)
    times = jnp.asarray(batch.times, dtype=points.dtype)
    statuses = jnp.asarray(batch.statuses, dtype=jnp.int32)
    n = int(points.shape[0])
    steps = jnp.zeros(n, dtype=jnp.int32)
    key = _as_key(key)

    status_np = np.asarray(statuses)
    idx = np.flatnonzero((status_np == RUNNING_INSIDE) | (status_np == RUNNING_TRANSMITTED))

    done = 0
    segment = 0
    while idx.size and done < budget:
        this_steps = min(segment_steps, budget - done)
        config = dataclasses.replace(engine.config, max_steps=this_steps)
        size = _bucket_size(idx.size, min_bucket)
        idx_padded = np.full(size, n, dtype=np.int32)
        idx_padded[: idx.size] = idx
        points, directions, times, statuses, steps, active = _run_segment(
            points, directions, times, statuses, steps,
            jnp.asarray(idx_padded), jax.random.fold_in(key, segment),
            engine.geometry, engine.materials, config,
        )
        done += this_steps
        segment += 1
        idx = np.flatnonzero(np.asarray(active))   # one small host sync per segment

    still_active = np.zeros(n, dtype=bool)
    still_active[idx] = True
    still_active = jnp.asarray(still_active)
    return JaxPhotonBatchOutput(
        interaction_points=points,
        directions=directions,
        times=times,
        statuses=statuses,
        active=still_active,
        truncated=still_active,
        steps=steps,
    )


class CompactedJaxEngine:
    """Wrap a ``JaxApproximateEngine`` so ``track_photon_batch`` uses compaction.

    Every other attribute (``geometry``, ``materials``, ``config``, ``scene``,
    ``track_photon_batch_with_tracks``, ``track_one`` ...) is forwarded to the
    wrapped engine unchanged.
    """

    def __init__(self, engine, *, segment_steps: int = 16, min_bucket: int = 256):
        self.engine = engine
        self.segment_steps = segment_steps
        self.min_bucket = min_bucket

    def track_photon_batch(self, photons, *, key=0):
        return track_photon_batch_compacted(
            self.engine, photons, key=key,
            segment_steps=self.segment_steps, min_bucket=self.min_bucket,
        )

    def __getattr__(self, name):
        return getattr(self.engine, name)


def enable_compaction(runner, *, segment_steps: int = 16, min_bucket: int = 256):
    """Switch a ``BatchTransport(engine="jax")`` runner to compacted transport.

    Returns the same runner, for chaining. Chunking, seeding (``fold_in`` per
    chunk), output order and ``TransportResult`` are unchanged.
    ``run(..., return_tracks=True)`` still uses the original, uncompacted path.
    """
    if getattr(runner, "name", None) != "jax":
        raise ValueError("enable_compaction needs BatchTransport(engine='jax')")
    if isinstance(runner.engine, CompactedJaxEngine):
        runner.engine.segment_steps = segment_steps
        runner.engine.min_bucket = min_bucket
    else:
        runner.engine = CompactedJaxEngine(
            runner.engine, segment_steps=segment_steps, min_bucket=min_bucket)
    runner.device_info["compaction"] = {
        "segment_steps": segment_steps, "min_bucket": min_bucket}
    return runner
