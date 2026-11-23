"""JAX implementation of the stochastic ApproximateEngine ray tracer.

The class `JaxApproximateEngine` mirrors the single- and
batch-photon entry points of `ApproximateEngine` while keeping the numerical
state in JAX arrays.  The functional entry point `transport_photons` is
JIT compiled and differentiable almost everywhere with respect to photon
coordinates, directions, face positions/normals, and continuous material
parameters.

!TWO POSSIBLE OPTION AVAILABEL. BUT ONLY `soft_interactions=False` works well

- `soft_interactions=False` (the default) performs the same hard Monte Carlo
choices as `ApproximateEngine`.  This is the mode to use for statistically
faithful simulations faster then ApproximateEngine

- `soft_interactions=True` replaces the direction at a
dielectric interface by a temperature-controlled relaxation of the sampled
Fresnel branch.  It is useful when gradients through reflection probabilities
are required, at the cost of no longer describing an exact ray at the
interface.

The geometry adapter supports every existing `ConvexPolyhedronVolume`
subclass (boxes, frusta, prisms, and pyramids).  Geometry is represented by
padded arrays of face points and outward normals, so the compiled transport
loop has a fixed shape and contains no Python-side photon iteration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from Photon_Simulator.core.detectorSetup.scene import Scene
from Photon_Simulator.core.photonsEngines.approximateEngine import (
    PhotonBatch, PhotonEvent
)
from .types import (
    Array,
    JaxTrackingConfig,
    JaxPhotonBatchInput,
    JaxPhotonEvent,
    JaxPhotonBatchOutput,
    JaxPhotonTrackOutput,
)
from .adapters import scene_to_jax, as_jax_photon_batch
from .transport import (
    transport_photons_dynamic,
    transport_photons_with_tracks,
)


def _as_key(key: Array | int) -> Array:
    """Convert an integer seed or an existing key to a legacy JAX PRNG key."""
    if isinstance(key, (int, np.integer)):
        return jax.random.PRNGKey(int(key))
    return jnp.asarray(key, dtype=jnp.uint32)


class JaxApproximateEngine:
    """Simulate optical-photon transport through a detector scene using JAX.

    Convert a detector scene into JAX geometry and material arrays, then
    propagate photons through bulk absorption, boundary interactions,
    reflection, refraction, and collection at configured sensor planes.

    Parameters
    ----------
    scene : Scene
        Detector scene containing supported convex-polyhedron volumes and
        their optical materials. Must not be None.
    config : JaxTrackingConfig, optional
        Transport settings, including the maximum number of steps,
        numerical tolerances, and optional relaxed interface interactions.
        If omitted, the default configuration is used.
    dtype : jax.numpy.dtype, optional
        Floating-point precision used for geometry and photon inputs.
        Defaults to `jnp.float32`. Mixed-precision mode requires
        `jnp.float64` geometry and uses `jnp.float32` material arrays.

    Attributes
    ----------
    scene : Scene
        Original detector scene. Its JAX representation is built during
        initialization; later scene changes are not automatically applied.
    config : JaxTrackingConfig
        Configuration used by the transport methods.
    dtype : jax.numpy.dtype
        Floating-point precision used to convert geometry and photon inputs.
    geometry : JaxSceneGeometry
        Padded face-plane representation of the detector volumes, including
        material indices, overlap priorities, and collection-plane settings.
    materials : JaxMaterialTable
        Optical material properties used during photon propagation.

    Notes
    -----
    JAX-native counterpart of `ApproximateEngine`.

    `track_photon_batch` propagates a batch until all photons are inactive
    or `config.max_steps` is reached. This method uses a dynamic loop and
    is not intended for reverse-mode differentiation.
    `track_photon_batch_with_tracks` also retains the photon states for
    visualization, while `track_one` returns a single-photon result.

    For differentiation with explicitly parameterized geometry or materials,
    call `transport_photons` directly, using `engine.geometry` and
    `engine.materials` as starting pytrees. Its fixed-step loop supports
    reverse-mode differentiation, subject to the discrete choices and
    piecewise behavior of the transport model.

    Randomness is controlled explicitly through the `key` argument of
    each tracking method. Reusing the same key with identical inputs and
    configuration reproduces the same random sequence.

    Photons still active at the step limit are marked as truncated in the
    returned result. By default, interface interactions use hard Monte Carlo
    choices; `config.soft_interactions` enables a relaxed direction model
    at dielectric interfaces.
    """

    def __init__(
        self,
        scene: Scene,
        config: JaxTrackingConfig | None = None,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if scene is None:
            raise ValueError("JaxApproximateEngine: missing scene")
        self.scene = scene
        self.config = config or JaxTrackingConfig()
        self.dtype = dtype
        self.geometry, self.materials = scene_to_jax(scene, dtype=dtype)
        if self.config.mixed_precision:
            if dtype != jnp.float64:
                raise ValueError(
                    "mixed_precision requires dtype=jnp.float64 for geometry"
                )
            _, self.materials = scene_to_jax(scene, dtype=jnp.float32)

    def track_photon_batch(
        self,
        photons: JaxPhotonBatchInput | PhotonBatch,
        *,
        key: Array | int = 0,
    ) -> JaxPhotonBatchOutput:
        batch = as_jax_photon_batch(photons, dtype=self.dtype)
        return transport_photons_dynamic(
            batch,
            _as_key(key),
            self.geometry,
            self.materials,
            self.config,
        )

    def track_photon_batch_with_tracks(
        self,
        photons: JaxPhotonBatchInput | PhotonBatch,
        *,
        key: Array | int = 0,
    ) -> JaxPhotonTrackOutput:
        """Track a batch while retaining states for graphical inspection."""
        batch = as_jax_photon_batch(photons, dtype=self.dtype)
        return transport_photons_with_tracks(
            batch, _as_key(key), self.geometry, self.materials, self.config
        )

    def track_one(
        self,
        photon: PhotonEvent,
        *,
        key: Array | int = 0,
    ) -> JaxPhotonEvent:
        output = self.track_photon_batch(
            as_jax_photon_batch(photon, dtype=self.dtype), key=key
        )
        return JaxPhotonEvent(
            interaction_point=output.interaction_points[0],
            direction=output.directions[0],
            status=output.statuses[0],
            time=output.times[0],
            truncated=output.truncated[0],
            steps=output.steps[0],
        )
