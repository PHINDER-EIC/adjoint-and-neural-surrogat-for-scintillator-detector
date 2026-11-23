"""Selectable forward transport. Scene lengths are cm; photon times are ns.

Chunk scheduling belongs here, outside JaxApproximateEngine. NumPy conversion
occurs only at the boundary to the simulator's existing PhotonBatch consumers.
"""
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings

import numpy as np
from numba import njit

from .approximateEngine import ApproximateEngine, PhotonBatch, TrackingConfig


@njit
def _seed_numba(seed):
    # Numba's reflector sampler has a separate RNG from Python's NumPy RNG.
    np.random.seed(seed)


@dataclass
class TransportResult:
    batch: PhotonBatch
    truncated: np.ndarray
    steps: np.ndarray
    tracks: list | None = None


def _track_cpu_chunk(engine, batch, seed, return_tracks):
    output = PhotonBatch(np.empty((batch.size, 3)), np.empty((batch.size, 3)),
                         np.empty(batch.size), np.empty(batch.size, dtype=np.int32))
    truncated = np.zeros(batch.size, dtype=bool)
    steps = np.zeros(batch.size, dtype=np.int32)
    tracks = [] if return_tracks else None
    saved_rng = np.random.get_state()
    np.random.seed(seed)
    _seed_numba(seed)
    try:
        for i in range(batch.size):
            state = batch.state_of(i)
            history = [state] if return_tracks else None
            # Preserve the bounded active state instead of raising at max_steps.
            while steps[i] < engine.config.max_steps and engine._is_running(state.status):
                state = engine._next_event(state)
                steps[i] += 1
                if return_tracks:
                    history.append(state)
            truncated[i] = engine._is_running(state.status)
            output.interaction_points[i] = state.interaction_point
            output.directions[i] = state.direction
            output.times[i] = state.time
            output.statuses[i] = int(state.status)
            if return_tracks:
                tracks.append(history)
    finally:
        np.random.set_state(saved_rng)
    return TransportResult(output, truncated, steps, tracks)


_CPU_WORKER_ENGINE = None


def _init_cpu_worker(scene, config):
    global _CPU_WORKER_ENGINE
    _CPU_WORKER_ENGINE = ApproximateEngine(scene, config=config)


def _track_cpu_task(task):
    start, batch, seed, return_tracks = task
    return start, _track_cpu_chunk(_CPU_WORKER_ENGINE, batch, seed, return_tracks)


class BatchTransport:
    def __init__(self, scene, *, engine="approximate", seed=42,
                 max_steps=20000, chunk_size=1000, cpu_workers=1, cpu_chunk_size=1000):
        if engine not in ("approximate", "jax"):
            raise ValueError("engine must be 'approximate' or 'jax'")
        if max_steps <= 0 or chunk_size <= 0:
            raise ValueError("max_steps and chunk_size must be positive")
        if cpu_workers <= 0 or cpu_chunk_size <= 0:
            raise ValueError("cpu_workers and cpu_chunk_size must be positive")
        if not 0 <= seed < 2**32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        self.name = engine
        self.chunk_size = chunk_size
        self.seed = seed
        self.cpu_workers = cpu_workers
        self.cpu_chunk_size = cpu_chunk_size
        self._chunk_index = 0
        self._cpu_seeds = np.random.SeedSequence(seed)
        self.device_info = {"backend": "cpu", "precision": "float64",
                            "workers_requested": cpu_workers, "workers": 1,
                            "cpu_chunk_size": cpu_chunk_size, "start_method": "spawn"}
        if engine == "jax":
            try:
                import jax
            except ImportError as exc:
                raise RuntimeError("JAX missing: install requirements-jax.txt in your environment") from exc
            # Must precede scene adaptation, keys, and every JAX array creation.
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            import jaxlib
            from .JaxEngine import JaxApproximateEngine, JaxTrackingConfig
            self.engine = JaxApproximateEngine(
                scene, config=JaxTrackingConfig(max_steps=max_steps,
                    mixed_precision=True, soft_interactions=False), dtype=jnp.float64)
            self._key = jax.random.PRNGKey(seed)
            devices = jax.devices()
            self.device_info = {
                "backend": jax.default_backend(), "devices": [str(d) for d in devices],
                "jax_version": jax.__version__, "precision": "mixed",
                "jaxlib_version": jaxlib.__version__,
                "gpu_available": any(d.platform == "gpu" for d in devices),
                "geometry_dtype": str(self.engine.geometry.face_points.dtype),
                "material_dtype": str(self.engine.materials.refractive_indices.dtype),
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "workers": 1,
            }
            if not any(d.platform == "gpu" for d in devices):
                warnings.warn("Backend GPU JAX non disponibile: trasporto eseguito su "
                              + jax.default_backend(), RuntimeWarning, stacklevel=2)
        else:
            self.engine = ApproximateEngine(scene, config=TrackingConfig(max_steps=max_steps))

    def run(self, batch, *, return_tracks=False):
        """Preserve input order. Reusing this runner advances independent streams.

        No CPU multiprocessing surrounds JAX. Fixed seed, chunk size and call
        sequence reproduce a run; changing chunk size changes random streams.
        Histories are allocated only on explicit request.
        """
        output = PhotonBatch(np.empty((batch.size, 3), dtype=float),
                             np.empty((batch.size, 3), dtype=float),
                             np.empty(batch.size, dtype=float),
                             np.empty(batch.size, dtype=np.int32))
        truncated = np.zeros(batch.size, dtype=bool)
        steps = np.zeros(batch.size, dtype=np.int32)
        tracks = [] if return_tracks else None
        if self.name == "jax":
            import jax
            for start in range(0, batch.size, self.chunk_size):
                stop = min(start + self.chunk_size, batch.size)
                part = PhotonBatch(batch.interaction_points[start:stop],
                                   batch.directions[start:stop], batch.times[start:stop],
                                   batch.statuses[start:stop])
                if self._chunk_index >= 2**32:
                    raise OverflowError("JAX chunk key counter exhausted")
                key = jax.random.fold_in(self._key, self._chunk_index)
                self._chunk_index += 1
                if return_tracks:
                    history = self.engine.track_photon_batch_with_tracks(part, key=key)
                    jax.block_until_ready(history)
                    result = history.output
                    tracks.extend(history.to_tracks())
                else:
                    result = self.engine.track_photon_batch(part, key=key)
                    jax.block_until_ready(result)
                for field in ("interaction_points", "directions", "times", "statuses"):
                    getattr(output, field)[start:stop] = np.asarray(getattr(result, field))
                truncated[start:stop] = np.asarray(result.truncated)
                steps[start:stop] = np.asarray(result.steps)
        else:
            starts = range(0, batch.size, self.cpu_chunk_size)
            # Chunk seeds depend on seed/run/chunk, never PID or scheduling.
            seeds = self._cpu_seeds.spawn(1)[0].spawn(len(starts))
            def tasks():
                for start, seed in zip(starts, seeds):
                    stop = min(start + self.cpu_chunk_size, batch.size)
                    part = PhotonBatch(batch.interaction_points[start:stop], batch.directions[start:stop],
                                       batch.times[start:stop], batch.statuses[start:stop])
                    yield start, part, int(seed.generate_state(1)[0]), return_tracks
            workers = min(self.cpu_workers, len(starts))
            self.device_info["workers"] = workers
            def collect(results):
                for start, result in results:
                    stop = start + result.batch.size
                    for field in ("interaction_points", "directions", "times", "statuses"):
                        getattr(output, field)[start:stop] = getattr(result.batch, field)
                    truncated[start:stop] = result.truncated
                    steps[start:stop] = result.steps
                    if return_tracks:
                        tracks.extend(result.tracks)
            if workers > 1:
                # spawn is safe after JAX has initialized GPU/CPU threads in
                # the parent. Workers import only the NumPy transport module.
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                        initializer=_init_cpu_worker, initargs=(self.engine.scene, self.engine.config)) as pool:
                    collect(pool.map(_track_cpu_task, tasks(), chunksize=1))
            else:
                collect((start, _track_cpu_chunk(self.engine, part, seed, history))
                        for start, part, seed, history in tasks())
        return TransportResult(output, truncated, steps, tracks)
