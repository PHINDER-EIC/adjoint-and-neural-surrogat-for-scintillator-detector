"""
Architecture
------------
ApproximateEngine
    Contains only the physics/numerics needed to propagate one photon.

PhotonBatchSimulator
    Executes many photons either sequentially or in parallel.

Worker functions
    Minimal multiprocessing glue. Each worker owns its local ApproximateEngine
    and reads from one PhotonBatch stored in worker-local globals.
"""

from logging import Logger
import os
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable, NamedTuple, Optional
from numba import njit
import math

import numpy as np

from Photon_Simulator.config.defaults import (
    RUNNING_TRANSMITTED,
    RUNNING_INSIDE,
    ABSORBED,
    LOST,
    CAUGHT,
    FASTMATH
)

from Photon_Simulator.config.settings import Settings
from Photon_Simulator.core.detectorSetup.material import ReflectiveMaterial
from Photon_Simulator.core.detectorSetup.scene import Scene, BoundaryHit
from Photon_Simulator.core.detectorSetup.Shapes.box import BoxVolume
from Photon_Simulator.utils.logger import get_logger


# =============================================================================
#                           DOMAINS MODELS
# =============================================================================

# =========================    TRACKING    ====================================

class Interaction(Enum):
    ABSORBED = 0
    REFLECTED = 1
    TRANSMITTED = 2


class PhotonStatus(IntEnum):
    RUNNING_TRANSMITTED = RUNNING_TRANSMITTED
    RUNNING_INSIDE = RUNNING_INSIDE
    ABSORBED = ABSORBED
    LOST = LOST
    CAUGHT = CAUGHT
    ABSORBED_AT_EDGES = 3
    ABSORBED_AT_REFLECTOR = 4
    ABSORBED_REFL_AT_Scintillator = 5
    ABSORBED_REFL_AT_Guide = 6
    ABSORBED_LAMBDA_AT_Scintillator = 7
    ABSORBED_LAMBDA_AT_Guide = 8
    LOST_AT_Scintillator = 9
    LOST_AT_Guide = 10

@dataclass(slots=True)
class PhotonEvent:
    """
    Contains details about a photon interaction event.

    Parameters
    ----------
    interaction_point : np.ndarray
        (3,) array representing the spatial position of the interaction.

    direction : np.ndarray
        (3,) array representing the photon direction after the interaction.

    status : PhotonStatus
        State of the photon after the interaction.

    time : float
        Time at which the interaction occurred.
    
     Notes
    -----
    For the initial photon event, the status should typically be
    set to ``PhotonStatus.RUNNING_INSIDE``.
    """
    interaction_point: np.ndarray
    direction: np.ndarray
    status: PhotonStatus
    time: float

class FresnelCoefficients(NamedTuple):
    theta_t: float | None
    R_s: float
    R_p: float
    T_s: float
    T_p: float
    R: float
    T: float
    A: float


# =========================    PARALLEL WORK    ===============================

@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """
    Configuration parameters controlling parallel photon tracking.

    Parameters
    ----------
    n_workers : int | None, default=None
        Number of worker processes used for multiprocessing.
        If None, the number of available CPU cores is used.

    chunk_size : int | None, default=None
        Number of photons assigned to each task chunk.
        If None, the chunk size is determined automatically.

    min_parallel_photons_per_worker : int, default=10
        Minimum number of photons required per worker before
        parallel execution is enabled. Smaller batches are
        processed sequentially.

    min_chunk_size : int, default=1000
        Minimum allowed chunk size when automatic chunking
        is used.

    chunk_multiplier : int, default=6
        Factor used to compute the automatic chunk size.
        Larger values generate smaller chunks and increase
        load balancing at the cost of scheduling overhead.

    multiprocessing_context : str | None, default=None
        Multiprocessing start method passed to
        `multiprocessing.get_context()`.
        Common values are `"spawn"`, `"fork`,
        and ``"forkserver"``. If None, the platform
        default is used.

    Notes
    -----
    If `multiprocessing_context` is None, the default start
    method of the operating system is used:

    - Windows: `spawn`
    - Linux: `fork`
    - macOS: `spawn`

    The `spawn` method is generally the safest option when
    working with libraries such as Numba, as it starts a fresh
    Python interpreter for each worker process.
    """
       
    n_workers: int | None = None
    chunk_size: int | None = None
    min_parallel_photons_per_worker: int = 10
    min_chunk_size: int = 1000
    chunk_multiplier: int = 6
    multiprocessing_context: str | None = None

#ANCHOR - PHOTON BATCHES
@dataclass(slots=True)
class PhotonBatch:
    interaction_points: np.ndarray
    directions: np.ndarray
    times: np.ndarray
    statuses: np.ndarray

    @property
    def size(self) -> int:
        return int(self.times.shape[0])

    def empty_like(self) -> "PhotonBatch":
        return PhotonBatch(
            interaction_points=np.empty_like(self.interaction_points),
            directions=np.empty_like(self.directions),
            statuses=np.empty_like(self.statuses),
            times=np.empty_like(self.times),
        )

    def state_of(self, i: int) -> PhotonEvent:
        return PhotonEvent(
            interaction_point=self.interaction_points[i],
            direction=self.directions[i],
            status=PhotonStatus(self.statuses[i]),
            time=float(self.times[i]),
        )

@dataclass(frozen=True, slots=True)
class ChunkResult:
    start: int
    stop: int
    points: np.ndarray
    directions: np.ndarray
    statuses: np.ndarray
    times: np.ndarray

# =========================    SIMULATION EXECUTION    ===============================

@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """
    Configuration parameters for tracking method.

    Parameters
    ----------
    max_steps: int
        tracking is a repetition of method ApproximateEngine.track_one(). 
        This sets max step before stoping the tracking
    eps_shift: float
        !keep this zero. Old parameter deprecated
    catch_tolerance: float
        the current (very weak) condition to define if a photon is caugth
        is if it has positive y direction and reach the guide end with 
        a spatial tolerance given by this parameter
    c_cm_ns: float
        light speed in cm/ns
    """
    max_steps: int = 5000000
    eps_shift: float = 0.# 1e-7
    catch_tolerance: float = 1e-8
    c_cm_ns: float = 29.9792458




# =============================================================================
# Single-photon physics engine
# =============================================================================


class ApproximateEngine:
    """
    Propagate a single photon through the detector scene.

    The engine is responsible for handling photon interactions
    with materials and boundaries, including absorption,
    reflection, and transmission.

    The tracking results can be obtained as:

    - the final photon state using `track_one()`
    - the complete interaction history using
      `track_one_with_history()`

    Parameters
    ----------
    scene : Scene
        Detector geometry and material configuration.
        The scene is used to determine photon propagation
        through different volumes and interfaces.

        If None, a `ValueError`is raised.

    config : TrackingConfig | None, default=None
        Tracking configuration parameters.
        See `TrackingConfig` for details.

        If None, default configuration values are used.

    logger : logging.Logger | None, default=None
        Logger used by the engine.

        If None, the default project logger is used.

    Notes
    -----
    The core of simulator is _next_event()

    This class is responsible only for single-photon tracking.

    It intentionally does not handle:

    - photon batches
    - multiprocessing pools
    - chunk scheduling
    - worker processes
    - progress reporting

    These responsibilities are delegated to `PhotonBatchSimulator`.

    Examples
    --------
    Track a single photon until it reaches a terminal state.

    >>> scene = Scene(...)
    >>> initial_state = PhotonState(
    ...     interaction_point=np.array([0.0, 0.0, 0.0]),
    ...     direction=np.array([0.0, 1.0, 0.0]),
    ...     status=PhotonStatus.RUNNING_INSIDE, 
    ...     time=0.,
    ... )
    >>> engine = ApproximateEngine(scene)
    >>> final_state = engine.track_one(initial_state)
    """

    def __init__(
        self,
        scene: Scene = None,
        config: TrackingConfig | None = None,
        logger:Optional[Logger]=None,
    ):
        if scene is None:
            raise ValueError("ApproximateEngine: missing scene")
        self.scene = scene
        self.config = config or TrackingConfig()
        self.log = get_logger(Settings.logger_name) if logger is None else logger


    # -------------------------------------------------------------
    #                       Public API
    # -------------------------------------------------------------
    #ANCHOR - TRACKING METHODS
    def track_one(self, initial_state: PhotonEvent) -> PhotonEvent:
        state = initial_state
        # old_state = initial_state
        # old_old_state = initial_state
        for _ in range(self.config.max_steps):
            if not self._is_running(state.status):
                return state
            try:
                # old_old_state = old_state
                # old_state = state
                state = self._next_event(state)
            except Exception as e:
                raise Exception(f"at step {_}: \n starting state: {initial_state} \n from state: `n {state} \n Error: {e}")

        else:
            raise Exception("Max steps reached")
            
        # return state

    def track_one_with_history(self, initial_state: PhotonEvent) -> list[PhotonEvent]:
        states = [initial_state]
        state = initial_state

        for _ in range(self.config.max_steps):
            if not self._is_running(state.status):
                break
            state = self._next_event(state)
            states.append(state)
        else:
            raise Exception("Max steps reached")

        return states

    
    
    # -------------------------------------------------------------
    #                     Propagation Logic
    # -------------------------------------------------------------
    def _next_event(self, state: PhotonEvent) -> PhotonEvent:
        start = state.interaction_point
        # direction = _unit_vector(state.direction)
        direction = state.direction
        # current_material = self.scene.get_material(start)
        
        # if self.scene.locate_volume(start) is None:
        #     raise ValueError(f"Startmg point is outside: \n {state}")
        
        try:
            hit = self.scene.first_boundary_hit(start, direction)
        except Exception as e:
            raise Exception(f"Error from first_boundary_hit for state: \n {state} \n {e}")

        if hit is None: 
            raise Exception(f"No hit found for state: \n {state}")
        current_material = hit.from_material
        intercept = hit.point
        new_time = state.time + self._travel_time(hit.distance, current_material.refractive_index)

        # Bulk absorption must be tested before interpreting the boundary: a
        # photon cannot reach an edge/readout if it was absorbed earlier along
        # the travelled segment.
        try:
            distance_to_absorption = self._sample_absorption_distance(current_material.absorption_mu)
        except Exception as e:
            raise Exception(f"Photon Stuts error  \"{e}\" \n in material {current_material} \n for photon state {state}")

        if distance_to_absorption < hit.distance:
            next_state = self._absorbed_in_material(
                start=start,
                direction=direction,
                distance=distance_to_absorption,
                refractive_index=current_material.refractive_index,
                previous_time=state.time,
            )
            if hit.from_volume.name == "scintillator":
                next_state.status = PhotonStatus.ABSORBED_LAMBDA_AT_Scintillator
            else:
                next_state.status = PhotonStatus.ABSORBED_LAMBDA_AT_Guide
            return next_state

        # Explicit readout ports define an unambiguous terminal plane, even if
        # the crossing happens geometrically at an edge of the terminal box.
        if self._is_caught(hit, direction):
            return PhotonEvent(intercept, direction, PhotonStatus.CAUGHT, new_time)

        # A box edge is ambiguous only when it is a *real optical interface*.
        # If both sides are the same material (e.g. two pieces of one logical
        # guide), the surface normal is irrelevant: there is no Fresnel jump.
        # Continue straight through the artificial segmentation boundary.
        if hit.from_volume.point_on_edge(intercept):
            if hit.to_volume is not None and hit.from_material == hit.to_material:
                return PhotonEvent(
                    intercept,
                    direction,
                    PhotonStatus.RUNNING_TRANSMITTED,
                    new_time,
                )
            return PhotonEvent(
                intercept,
                direction,
                PhotonStatus.ABSORBED_AT_EDGES,
                new_time,
            )

        normal = self._oriented_normal(hit.normal, direction)

        if isinstance(hit.to_material, ReflectiveMaterial):
            new_direction = np.empty(3, dtype=np.float64)
            status  = self._sample_reflection_model(hit.to_material.specular_fraction, 
                                                    hit.to_material.reflectance,
                                                    direction,normal, new_direction)
            shifted = intercept  #+ self.config.eps_shift * new_direction
            if status == PhotonStatus.ABSORBED:
                raise Exception("error: status cannot be ABSORBED in _sample_reflection_model")
            if status == PhotonStatus.ABSORBED_AT_REFLECTOR:
                if hit.from_volume.name == "scintillator":
                    status = PhotonStatus.ABSORBED_REFL_AT_Scintillator
                else:
                    status = PhotonStatus.ABSORBED_REFL_AT_Guide
            return PhotonEvent(shifted, new_direction, status, new_time)

            # # raise Exception("dubbio")
            # new_direction = np.empty(3, dtype=np.float64)
            # status = self._lambertian_reflection(
            #     hit.to_material.reflectance,
            #     normal,
            #     direction,
            #     new_direction
            # )
            # return PhotonState(
            #     intercept,
            #     new_direction,
            #     PhotonStatus(status),
            #     new_time
            # )
            interaction = self._sample_reflection(hit.to_material.reflectance)
            if interaction == Interaction.REFLECTED:
                reflected = self._reflected_direction(direction, normal)
                shifted = intercept + self.config.eps_shift * reflected
                new_state = PhotonEvent(shifted, reflected, PhotonStatus.RUNNING_INSIDE, new_time)
            else:
                new_state = PhotonEvent(intercept, direction, PhotonStatus.ABSORBED, new_time)
                
            return new_state

        
        n_from = hit.from_material.refractive_index
        n_to = hit.to_material.refractive_index

        theta_in = np.arccos(np.clip(abs(np.dot(direction, normal)), -1.0, 1.0))
        coeffs = self._fresnel_coefficients(theta_in, n_from, n_to)
        interaction = self._sample_interaction(coeffs.R, coeffs.T)

        if interaction is Interaction.ABSORBED:
            return PhotonEvent(intercept, direction, PhotonStatus.ABSORBED, new_time)

        if interaction is Interaction.REFLECTED:
            reflected = self._reflected_direction(direction, normal)
            shifted = intercept  #+ self.config.eps_shift * reflected
            new_state = PhotonEvent(shifted, reflected, PhotonStatus.RUNNING_INSIDE, new_time)
            # if self.scene.locate_volume(shifted+reflected) is None:
            #     raise ValueError(f"point is REFLECTED outside: \n from {state}\n TO: {new_state}")
            return new_state

        # if the photon is not reflected or absorbed, then it is transmited
        if interaction is Interaction.TRANSMITTED:
            
            # if hit.to_volume is None:
            #     return PhotonState(intercept, direction, PhotonStatus.LOST, new_time)
            
            ok, transmitted = self._transmitted_direction(direction, normal, n_from, n_to)

            if not ok:
                raise ValueError("transmitted direction is None")
                # reflected = self._reflected_direction(direction, normal)
                # shifted = intercept + self.config.eps_shift * reflected
                # return PhotonState(shifted, reflected, PhotonStatus.RUNNING_INSIDE, new_time)


            shifted = intercept #+ self.config.eps_shift * transmitted
            new_state = PhotonEvent(shifted, transmitted, PhotonStatus.RUNNING_TRANSMITTED, new_time)
            if hit.to_volume is None:
                if hit.from_volume.name == "scintillator":
                    return PhotonEvent(shifted, transmitted, PhotonStatus.LOST_AT_Scintillator, new_time)
                else:
                    return PhotonEvent(shifted, transmitted, PhotonStatus.LOST_AT_Guide, new_time)
                    # return PhotonEvent(shifted, transmitted, PhotonStatus.LOST, new_time)
                    raise ValueError(f"point is transmitted outside: \n from {state}\n TO: {new_state} \n considering T = {coeffs.T} e R = {coeffs.R} \n -> R+T = {coeffs.R+coeffs.T}")
        else:
            raise Exception("Interaction Error")
        return new_state
    # -------------------------------------------------------------
    #                     Utility helpers
    # -------------------------------------------------------------

    # --------------> INTERACTION UTILITIES <--------------------------
    @staticmethod
    def _is_running(status: PhotonStatus | int) -> bool:
        # return status < 0
        return status in (PhotonStatus.RUNNING_INSIDE, PhotonStatus.RUNNING_TRANSMITTED)
    
    # def _is_caught(self, hit: BoundaryHit, direction: np.ndarray) -> bool:
    #     if hit.from_volume is None:
    #         return False
    #     if hit.from_volume.name != "fiber_core":
    #         return False
    #     if not isinstance(hit.from_volume, BoxVolume):
    #         return False

    #     end_y = hit.from_volume.box_max[1]
    #     hit_terminal_face = np.isclose(hit.point[1], end_y, atol=self.config.catch_tolerance)
    #     moving_to_readout = direction[1] > 0

    #     return bool(hit_terminal_face and moving_to_readout)
    def _is_caught(self, hit: BoundaryHit, direction: np.ndarray) -> bool:
        # New PHINDER-compatible interface: if the Scene defines explicit
        # readout ports, only those terminal faces can catch photons.  This is
        # axis-independent and therefore works for future rotated/segmented
        # guide geometries.
        if getattr(self.scene, "readout_ports", ()):
            return self.scene.readout_channel(
                hit, direction, tolerance=self.config.catch_tolerance
            ) is not None

        # Backward-compatible Lorenzo behaviour for old scenes/examples.
        volume = hit.from_volume

        if volume is None:
            return False

        if volume.name != "fiber_core":
            return False

        if not isinstance(volume, BoxVolume):
            return False

        return (
            direction[1] > 0.0
            and abs(hit.point[1] - volume.box_max[1]) <= self.config.catch_tolerance
        )
    
    @staticmethod
    def _fresnel_coefficients(theta_in: float, n_from: float, n_to: float) -> FresnelCoefficients:
        sin_theta_t = n_from * math.sin(theta_in) / n_to

        if abs(sin_theta_t) > 1.0:
            return FresnelCoefficients(
                None,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )

        theta_t = math.asin(sin_theta_t)

        cos_i = math.cos(theta_in)
        cos_t = math.cos(theta_t)

        n_from_cos_i = n_from * cos_i
        n_to_cos_t = n_to * cos_t

        den_s = n_from_cos_i + n_to_cos_t
        den_p = n_to * cos_i + n_from * cos_t

        r_s = (n_from_cos_i - n_to_cos_t) / den_s
        r_p = (n_to * cos_i - n_from * cos_t) / den_p

        t_s = (2.0 * n_from_cos_i) / den_s
        t_p = (2.0 * n_from_cos_i) / den_p

        R_s = r_s * r_s
        R_p = r_p * r_p

        prefactor = (n_to * cos_t) / n_from_cos_i

        T_s = prefactor * t_s * t_s
        T_p = prefactor * t_p * t_p

        R = 0.5 * (R_s + R_p)
        T = 0.5 * (T_s + T_p)
        A = 1.0 - R - T

        if abs(A)>=1e-5:
            raise Exception("In _fresnel_coefficients: absorption coefficient is not null")
        # if A < 0.0:
        #     A = 0.0

        return FresnelCoefficients(theta_t, R_s, R_p, T_s, T_p, R, T, A)
        
    
    # .......... ABSORPTION .....................
    def _absorbed_in_material(
        self,
        *,
        start: np.ndarray,
        direction: np.ndarray,
        distance: float,
        refractive_index: float,
        previous_time: float,
    ) -> PhotonEvent:
        end_point = start + distance * direction
        time = previous_time + self._travel_time(distance, refractive_index)
        return PhotonEvent(end_point, direction, PhotonStatus.ABSORBED, time)

    # .......... REFLECTION .....................
    @staticmethod
    @njit(fastmath=True)
    def _reflected_direction(direction, normal):

        dot = (
            direction[0] * normal[0] +
            direction[1] * normal[1] +
            direction[2] * normal[2]
        )

        if dot > 0.0:
            dot = -dot
            raise Exception("something wrong")
            nx = -normal[0]
            ny = -normal[1]
            nz = -normal[2]
        else:
            nx = normal[0]
            ny = normal[1]
            nz = normal[2]

        out = np.empty(3, dtype=np.float64)

        out[0] = direction[0] - 2.0 * dot * nx
        out[1] = direction[1] - 2.0 * dot * ny
        out[2] = direction[2] - 2.0 * dot * nz

        return _unit_vector(out)

    @staticmethod
    @njit(fastmath=FASTMATH)
    def _lambertian_reflection(
        reflectance: float,
        normal: np.ndarray,
        direction: np.ndarray,
        out_direction: np.ndarray,
    ) -> int:

        if np.random.random() > reflectance:
            out_direction[0] = direction[0]
            out_direction[1] = direction[1]
            out_direction[2] = direction[2]
            return ABSORBED

        nx = normal[0]
        ny = normal[1]
        nz = normal[2]

        u1 = np.random.random()
        u2 = np.random.random()

        cos_theta = math.sqrt(u1)
        sin_theta = math.sqrt(1.0 - u1)

        phi = 2.0 * math.pi * u2

        lx = sin_theta * math.cos(phi)
        ly = sin_theta * math.sin(phi)
        lz = cos_theta

        # Tangent
        if abs(nz) < 0.999:

            tx = -ny
            ty = nx
            tz = 0.0

            inv_norm = 1.0 / math.sqrt(tx * tx + ty * ty)

        else:

            tx = 0.0
            ty = -nz
            tz = ny

            inv_norm = 1.0 / math.sqrt(ty * ty + tz * tz)

        tx *= inv_norm
        ty *= inv_norm
        tz *= inv_norm

        # Bitangent = normal x tangent
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx

        out_direction[0] = lx * tx + ly * bx + lz * nx
        out_direction[1] = lx * ty + ly * by + lz * ny
        out_direction[2] = lx * tz + ly * bz + lz * nz

        return RUNNING_INSIDE
    
    # .......... TRANSMISSION .....................
    @staticmethod
    @njit(fastmath=True)
    def _transmitted_direction(
        direction: np.ndarray,
        normal: np.ndarray,
        n1: float,
        n2: float,
    ):
    
        cos_i = -(
            normal[0] * direction[0] +
            normal[1] * direction[1] +
            normal[2] * direction[2]
        )
    
        eta = n1 / n2
    
        sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    
        if sin_t2 > 1.0:
            return False, np.empty(3, dtype=np.float64)
    
        cos_t = math.sqrt(1.0 - sin_t2)
    
        out = np.empty(3, dtype=np.float64)
    
        out[0] = eta * direction[0] + (eta * cos_i - cos_t) * normal[0]
        out[1] = eta * direction[1] + (eta * cos_i - cos_t) * normal[1]
        out[2] = eta * direction[2] + (eta * cos_i - cos_t) * normal[2]
    
        return True, _unit_vector(out)


    # --------------> GEOMETRY UTILITIES <-------------------------
    @staticmethod
    @njit(fastmath=True)
    def _oriented_normal(normal, direction):
        """
        If both direction and normal go outside, it change normal
        """
        if (
            direction[0] * normal[0] +
            direction[1] * normal[1] +
            direction[2] * normal[2]
        ) > 0.0:

            out = np.empty(3, dtype=np.float64)

            out[0] = -normal[0]
            out[1] = -normal[1]
            out[2] = -normal[2]

            return out

        return normal.copy()
    
    
    
    # --------------> NUMERICAL UTILITIES <------------------------
    def _travel_time(self, distance: float, refractive_index: float) -> float:
        return self.__travel_time(
        distance,
        refractive_index,
        self.config.c_cm_ns
    )

    # --------------> SAMPLING UTILITIES <-------------------------
    def _sample_absorption_distance(self, mu: float) -> float:
        if mu is None:
            raise ValueError("Mu cannot be NoneType")
        if mu <= 0:
            return np.inf
        return -np.log(np.random.rand()) / mu

    def _sample_interaction(self, reflection_coef: float, transmission_coef: float) -> Interaction:
        xi = np.random.rand()
        if abs(reflection_coef + transmission_coef-1) > 1e-12:
            raise Exception(f"R+T = {reflection_coef} + {transmission_coef} = {reflection_coef + transmission_coef} must be 1")
        if xi < reflection_coef:
            return Interaction.REFLECTED
        if xi < reflection_coef + transmission_coef:
            return Interaction.TRANSMITTED
        return Interaction.ABSORBED
    def _sample_reflection(self, reflectance):
        xi = np.random.rand()
        if xi < reflectance:
            return Interaction.REFLECTED
        else:
            return Interaction.ABSORBED
    def _sample_reflection_model(
        self,
        specular_fraction,
        reflectance,
        direction,
        normal,
        new_direction
    ):
    
        interaction = self._sample_reflection(reflectance)

        if interaction == Interaction.REFLECTED:
            xi = np.random.rand()
            if xi <= specular_fraction:
        
            
                tmp = self._reflected_direction(direction, normal)
    
                new_direction[0] = tmp[0]
                new_direction[1] = tmp[1]
                new_direction[2] = tmp[2]
    
                return PhotonStatus.RUNNING_INSIDE
    
            else:
               raise NotImplementedError("Lambertian reflection is not implemented yet")
               return PhotonStatus(
                   self._lambertian_reflection(
                       reflectance,
                       normal,
                       direction,
                       new_direction
                   )
               )
        else:
            new_direction[0] = direction[0]
            new_direction[1] = direction[1]
            new_direction[2] = direction[2]

            return PhotonStatus.ABSORBED_AT_REFLECTOR
    
    
    # --------------> Low-level numerical kernels <-------------------------
    @staticmethod
    @njit(fastmath=True)
    def __travel_time(distance, refractive_index, c_cm_ns):
        return distance * refractive_index / c_cm_ns

    

@njit(fastmath=True)
def _unit_vector(v):

        norm = math.sqrt(
            v[0] * v[0] +
            v[1] * v[1] +
            v[2] * v[2]
        )

        if norm < 1e-15:
            raise ValueError("Zero vector")

        out = np.empty(3, dtype=np.float64)

        out[0] = v[0] / norm
        out[1] = v[1] / norm
        out[2] = v[2] / norm

        return out


# =============================================================================
# Batch simulator
# =============================================================================
class PhotonBatchSimulator:
    """
    Execute photon tracking over a PhotonBatch.

    This class is responsible for the execution strategy of the
    simulation and is independent of the underlying photon physics,
    which is handled by ``ApproximateEngine``.

    A batch can be processed either sequentially or in parallel:

    - Sequential execution tracks photons one at a time in the
      current process.
    - Parallel execution distributes photons across multiple
      worker processes using Python's multiprocessing module.

    The simulator automatically selects the requested execution
    strategy and collects the final photon states into a new
    `PhotonBatch`.

    Parameters
    ----------
    engine : ApproximateEngine
        Physics engine used to propagate individual photons.

    parallel_config : ParallelConfig | None, default=None
        Configuration parameters controlling multiprocessing
        execution. If None, default settings are used.

    Notes
    -----
    This class manages execution only and does not implement any
    photon interaction physics.

    Parallel execution is only enabled when explicitly requested
    and when the batch size is large enough to justify the
    multiprocessing overhead.
   
    """

    def __init__(
        self,
        engine: ApproximateEngine,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        self.engine = engine
        self.parallel_config = parallel_config or ParallelConfig()

    def run(
        self,
        batch: PhotonBatch,
        *,
        parallel: bool = False,
        return_tracks: bool = False,
        show_progress: bool = False,
    ):
        if parallel:
            if return_tracks:
                raise ValueError("Parallel execution with return_tracks=True is not supported.")
            return self.run_parallel(batch, show_progress=show_progress)

        return self.run_sequential(
            batch,
            return_tracks=return_tracks,
            show_progress=show_progress,
        )

    def run_sequential(
        self,
        batch: PhotonBatch,
        *,
        return_tracks: bool = False,
        show_progress: bool = False,
    ):
        if return_tracks:
            return self._run_sequential_with_tracks(batch, show_progress=show_progress)

        output = batch.empty_like()
        iterator = _progress_iter(range(batch.size), show_progress, "Tracking photons")

        for i in iterator:
            final_state = self.engine.track_one(batch.state_of(i))
            self._write_state(output, i, final_state)

        return output

    def run_parallel(
        self,
        batch: PhotonBatch,
        *,
        show_progress: bool = False,
    ) -> PhotonBatch:
        """
        Track a batch of photons using multiple worker processes.

        The input batch is divided into chunks that are distributed
        among worker processes. Each worker owns a local
        ``ApproximateEngine`` instance and independently propagates
        the photons assigned to its chunk.

        Results from all workers are collected and assembled into
        a new ``PhotonBatch`` containing the final photon states.

        Parameters
        ----------
        batch : PhotonBatch
            Batch of photons to track.

        show_progress : bool, default=False
            If True, display a progress bar while tracking photons.

        Returns
        -------
        PhotonBatch
            Batch containing the final state of every photon.

        Notes
        -----
        For small batches, the simulator may automatically fall back
        to sequential execution when multiprocessing overhead is
        expected to outweigh its benefits.

        Track histories are not supported during parallel execution.
        """
        #ANCHOR - PARALLEL EXECUTION LOGIC
        n_photons = batch.size
        workers = self._choose_worker_count(n_photons)

        if n_photons < self.parallel_config.min_parallel_photons_per_worker * workers:
            return self.run_sequential(batch, return_tracks=False, show_progress=show_progress)

        chunk_size = self._choose_chunk_size(n_photons, workers)
        chunks = make_chunks(n_photons, chunk_size)
        output = batch.empty_like()

        context = self._get_mp_context()

        with context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.engine.scene, self.engine.config, batch),
        ) as pool:
            results = pool.imap_unordered(_track_chunk, chunks, chunksize=max(1, len(chunks) // (workers * 4)))
            # results = _progress_iter(results, show_progress, "Tracking photon chunks", total=len(chunks))
            
            if show_progress:
                from tqdm import tqdm
                pbar = tqdm(
                    total=n_photons,
                    desc="Tracking photons",
                    unit="ph",
                    dynamic_ncols=True,
                    leave=False,
                )
            else:
                pbar = None

            for result in results:
                output.interaction_points[result.start:result.stop] = result.points
                output.directions[result.start:result.stop] = result.directions
                output.statuses[result.start:result.stop] = result.statuses
                output.times[result.start:result.stop] = result.times
                if pbar is not None:
                    pbar.update(result.stop - result.start)


        return output

    def _run_sequential_with_tracks(self, batch: PhotonBatch, *, show_progress: bool = False):
        output = batch.empty_like()
        tracks: list[list[PhotonEvent]] = []
        iterator = _progress_iter(range(batch.size), show_progress, "Tracking photons")

        for i in iterator:
            track = self.engine.track_one_with_history(batch.state_of(i))
            final_state = track[-1]
            self._write_state(output, i, final_state)
            tracks.append(track)

        return output, tracks

    @staticmethod
    def _write_state(output: PhotonBatch, i: int, state: PhotonEvent) -> None:
        output.interaction_points[i] = state.interaction_point
        output.directions[i] = state.direction
        output.statuses[i] = int(state.status)
        output.times[i] = state.time

    def _choose_worker_count(self, n_items: int) -> int:
        if n_items <= 0:
            return 1

        requested = self.parallel_config.n_workers
        available = os.cpu_count() or 1
        workers = available if requested is None else requested

        return max(1, min(int(workers), n_items))

    def _choose_chunk_size(self, n_items: int, n_workers: int) -> int:
        """
        Determine the chunk size used for parallel execution.

        When no explicit chunk size is provided in the configuration,
        a dynamic value is computed from the batch size and the
        number of worker processes.

        The goal is to balance:

        - workload distribution among workers;
        - multiprocessing scheduling overhead;
        - overall CPU utilization.

        Parameters
        ----------
        n_items : int
            Total number of photons in the batch.

        n_workers : int
            Number of worker processes used for execution.

        Returns
        -------
        int
            Number of photons assigned to each chunk.

        Notes
        -----
        The automatically computed chunk size is given by:
        `n_items // (chunk_multiplier * n_workers)`
        and is never smaller than `min_chunk_size`.

        ? SUGGESTION:
        Smaller chunks generally improve load balancing but increase
        scheduling overhead, while larger chunks reduce overhead at
        the cost of potentially poorer workload distribution.
        """
        if self.parallel_config.chunk_size is not None:
            return int(self.parallel_config.chunk_size)

        dynamic_size = n_items // max(1, self.parallel_config.chunk_multiplier * n_workers)
        return max(self.parallel_config.min_chunk_size, dynamic_size)

    def _get_mp_context(self):
        if self.parallel_config.multiprocessing_context is None:
            return mp
        return mp.get_context(self.parallel_config.multiprocessing_context)

# =============================================================================
# Multiprocessing worker glue
# =============================================================================
#ANCHOR - MULTIPROCESSING WORKER 
"""
Helper functions used to support multiprocessing execution.

These functions manage process-local worker initialization and
chunk execution. They act as a thin interface between
`PhotonBatchSimulator` and Python's multiprocessing framework.

The worker processes maintain local copies of the tracking engine
and photon batch in order to minimize serialization overhead and
avoid repeatedly transferring large objects between processes.
"""

_WORKER_ENGINE: ApproximateEngine | None = None
_WORKER_BATCH: PhotonBatch | None = None


def _init_worker(
    scene: Scene,
    tracking_config: TrackingConfig,
    batch: PhotonBatch,
    seed: int | None = None,
) -> None:
    """
    Initialize process-local worker state.

    This function is executed once when a worker process starts.
    It creates a local ``ApproximateEngine`` instance and stores
    the input ``PhotonBatch`` in worker-local globals.

    Keeping these objects local to each process avoids repeatedly
    serializing and transferring them for every chunk execution.

    Parameters
    ----------
    scene : Scene
        Detector geometry and material configuration.

    tracking_config : TrackingConfig
        Configuration parameters used to initialize the tracking
        engine.

    batch : PhotonBatch
        Batch of photons to be processed by the worker.

    seed : int | None, default=None
        Random seed used to initialize NumPy's random number
        generator. If None, the worker process identifier
        (PID) is used.

    Notes
    -----
    This function is intended to be used as the ``initializer``
    argument of ``multiprocessing.Pool`` and should not be
    called directly during normal operation.
    """
    global _WORKER_ENGINE, _WORKER_BATCH

    if seed is None:
        seed = os.getpid()

    np.random.seed(seed)
    _WORKER_ENGINE = ApproximateEngine(scene=scene, config=tracking_config)
    _WORKER_BATCH = batch


def _track_chunk(bounds: tuple[int, int]) -> ChunkResult:
    """
    Track all photons contained in a chunk.
    
    The function retrieves the worker-local tracking engine and
    photon batch, propagates every photon within the specified
    range, and returns the corresponding final states.
    
    Parameters
    ----------
    bounds : tuple[int, int]
        Pair ``(start, stop)`` defining the range of photon
        indices assigned to the chunk.
    
    Returns
    -------
    ChunkResult
        Object containing the final photon positions,
        directions, statuses, and times for all photons
        in the chunk.
    
    Raises
    ------
    RuntimeError
        If the worker has not been initialized correctly.
    
    Notes
    -----
    This function is executed inside worker processes and is
    designed to be dispatched by ``multiprocessing.Pool``.
    The returned ``ChunkResult`` is later assembled into the
    final output batch by ``PhotonBatchSimulator``.
    """
    if _WORKER_ENGINE is None or _WORKER_BATCH is None:
        raise RuntimeError("Worker was not initialized correctly.")

    start, stop = bounds
    batch = _WORKER_BATCH
    engine = _WORKER_ENGINE

    points = np.empty_like(batch.interaction_points[start:stop])
    directions = np.empty_like(batch.directions[start:stop])
    statuses = np.empty_like(batch.statuses[start:stop])
    times = np.empty_like(batch.times[start:stop])

    for local_i, photon_i in enumerate(range(start, stop)):
        final_state = engine.track_one(batch.state_of(photon_i))
        points[local_i] = final_state.interaction_point
        directions[local_i] = final_state.direction
        statuses[local_i] = int(final_state.status)
        times[local_i] = final_state.time

    return ChunkResult(start, stop, points, directions, statuses, times)


# =============================================================================
# Utility functions
# =============================================================================

def make_chunks(n_items: int, chunk_size: int) -> list[tuple[int, int]]:
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    return [
        (start, min(start + chunk_size, n_items))
        for start in range(0, n_items, chunk_size)
    ]


def _progress_iter(iterable: Iterable, enabled: bool, description: str, total: int | None = None):
    if not enabled:
        return iterable

    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=description)
    except ImportError:
        return iterable


# =============================================================================
# Backward-compatible adapter 
# =============================================================================
#! DEPRECATED
class ApproximateBatchEngine(ApproximateEngine):
    """Compatibility layer for the old ApproximateEngine batch API.

    Prefer using:
        engine = ApproximateEngine(scene)
        simulator = PhotonBatchSimulator(engine)
        simulator.run_parallel(batch)
    """

    def track_photon_batch(
        self,
        batch: PhotonBatch,
        return_tracks: bool = False,
        show_progress: bool = False,
    ):
        simulator = PhotonBatchSimulator(self)
        return simulator.run_sequential(
            batch,
            return_tracks=return_tracks,
            show_progress=show_progress,
        )

    def track_photon_batch_parallel(
        self,
        batch: PhotonBatch,
        n_workers: int | None = None,
        chunk_size: int | None = None,
        show_progress: bool = False,
    ) -> PhotonBatch:
        simulator = PhotonBatchSimulator(
            self,
            parallel_config=ParallelConfig(
                n_workers=n_workers,
                chunk_size=chunk_size,
            ),
        )
        return simulator.run_parallel(batch, show_progress=show_progress)
