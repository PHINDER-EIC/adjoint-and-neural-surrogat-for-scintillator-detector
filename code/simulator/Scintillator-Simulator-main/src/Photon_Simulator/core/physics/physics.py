""" 
Physics of a MIP particle in a scintillator

1) particle track
!x(s) = x0+s*u with u is the direction and s the path lenght

2) I divide the track in steps dx

?3) I can use bethe block or a mean dE/dx = 2MeV/cm for a MIP

4) for each step 
!dE = dEdx * dx (MeV)
? Use a Landau/Vavilov

5) Use light yield of the scintilletor (Y e.g. 10e4 photons/MeV)
! N_mean = Y *dE

6) I extract effective photons by a Poisson
!N_photons = poisson(N_mean)

7) For each photon:
- the starting position is a casual point in the step dx
! x_emit = x_start + random() * dx * u
- the direction is isotrope
!cos_theta = uniform(-1, 1)
!phi = uniform(0, 2π)

- time of the emission (with tau depending on decay time of scintillator)
!t_emit = t_particle + exponential(tau)

TODO
- wavelenght

"""
from Photon_Simulator.utils.logger import get_logger
from Photon_Simulator.config.settings import settings
from Photon_Simulator.core.physics.particles import MIP, Particle
from Photon_Simulator.core.detectorSetup.scene import Scene

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the normalized vector."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Direction vector cannot be zero.")
    return v / norm


class ParticleInteraction:
    """
    Handle the interaction of a charged particle in a scintillating medium.

    This class generates scintillation photon emission points along a particle
    track and assigns emission directions and times according to simple
    scintillation physics.

    NOTE
    ----
    The current model assumes:
    - straight particle track
    - constant mean energy loss dE/dx
    - Poisson fluctuations in the number of emitted photons
    - isotropic photon emission
    - exponential scintillation emission delay
    """

    def __init__(
        self,
        scene: Scene,
        particle: Particle = MIP(),
        logger=None,
    ):
        self.log = get_logger(settings.logger_name) if logger is None else logger
        self.scene = scene
        self.particle = particle

        # Speed of light in vacuum expressed in cm/ns.
        #! DA SPOSTARE
        self.c_cm_ns = 29.9792458

    def generate_photon_source(
        self,
        x0,          # cm
        direction,   
        length,      # cm
        dx,          # cm
        beta,        
        dEdx,        # MeV/cm
        light_yield, # photons/MeV
        rng=None,
    ):
        """
        Generate scintillation photon emission positions along a particle track.

        The particle track is modeled as a straight line:

            x(s) = x0 + s * u

        where `x0` is the starting point, `u` is the normalized direction, and
        `s` is the path length along the track.

        The track is divided into steps of length `dx`. For each step, the
        deposited energy is computed using a constant stopping power `dEdx`.
        The expected number of scintillation photons is then obtained from the
        scintillator light yield, and the actual number of photons is sampled
        from a Poisson distribution.

        Parameters
        ----------
        x0 : array-like, shape (3,)
            Initial particle position in cm.
        direction : array-like, shape (3,)
            Particle direction vector. It does not need to be normalized.
        length : float
            Total particle path length inside the scintillator, in cm.
        dx : float
            Step size used to discretize the particle track, in cm.
        beta : float
            Particle velocity in units of the speed of light, v/c.
        dEdx : float
            Mean energy loss per unit length, in MeV/cm.
            For a MIP in many plastic scintillators, a typical value is
            approximately 2 MeV/cm.
        light_yield : float
            Scintillator light yield, in photons/MeV.
        rng : np.random.Generator, optional
            NumPy random number generator. If None, a default generator is used.

        Returns
        -------
        positions : np.ndarray, shape (N, 3)
            Generated photon emission positions in cm.
        times : np.ndarray, shape (N,)
            Particle arrival time at each emission position, in ns.

        Notes
        -----
        This method only generates the photon source points and the particle
        arrival times. It does not yet assign photon directions, scintillation
        delays, or wavelengths.
        
        ! Not Birks law, delta-electrons and light diffusion is considered
        Hint: potrei considerare un cilindro attorno alla traject della particella
        e poi prendere un punto random li... 
        """
        if rng is None:
            rng = np.random.default_rng()

        x0 = np.asarray(x0, dtype=float)
        u = normalize(direction)

        if length <= 0:
            raise ValueError("Track length must be positive.")

        if dx <= 0:
            raise ValueError("Step size dx must be positive.")

        if beta <= 0:
            raise ValueError("Particle beta must be positive.")

        if dEdx < 0:
            raise ValueError("dEdx cannot be negative.")

        if light_yield < 0:
            raise ValueError("Light yield cannot be negative.")

        n_steps = int(np.ceil(length / dx))

        positions = []
        times = []

        for i in range(n_steps):
            # Current step boundaries along the particle path.
            s_start = i * dx
            s_end = min((i + 1) * dx, length)
            step_length = s_end - s_start

            # Energy deposited in this step.
            # Current model: deterministic mean energy loss
            # TODO: replace or extend with Landau/Vavilov fluctuations
            dE = dEdx * step_length

            # Mean number of scintillation photons produced in this step
            n_mean = light_yield * dE

            # Actual number of photons sampled from Poisson statistics
            n_photons = rng.poisson(n_mean)

            if n_photons == 0:
                continue

            # Random emission positions uniformly distributed inside the step
            r = rng.random(n_photons)
            s_emit = s_start + r * step_length

            # Convert curvilinear coordinate s into 3D positions
            x_emit = x0 + np.outer(s_emit, u)

            # Particle arrival time at each emission point
            # v = beta * c
            t_emit_particle = s_emit / (beta * self.c_cm_ns) # +np.clip(np.random.exponential(scale=0.3), 0.05, 2.0)  

            positions.append(x_emit)
            times.append(t_emit_particle)

        if len(positions) == 0:
            self.log.warning("No scintillation photons generated.")
            return np.empty((0, 3)), np.empty((0,))

        positions = np.vstack(positions)
        times = np.concatenate(times)

        return positions, times
    
    def photon_generator(
        self,
        source_points,
        times,
        tau_r_ns,
        tau_d_ns,
        rng=None
    ):
        """
        Generate isotropically emitted photons and their emission times
        including scintillation delays.

        This method simulates the emission of photons from a set of source
        points. Photon directions are sampled isotropically over the unit
        sphere, while emission times are generated according to a
        scintillation time profile with finite rise and decay times.
        The scintillation delay distribution is sampled using rejection
        sampling with an exponential proposal distribution based on the
        decay constant.

        Parameters
        ----------
        source_points : ndarray
            Array containing the photon source positions.
            Only the number of points is used by this method.

        times : ndarray
            Initial emission times associated with each source point
            in nanoseconds.

        tau_r_ns : float
            Scintillation rise time constant in nanoseconds.

        tau_d_ns : float
            Scintillation decay time constant in nanoseconds.
            Must be greater than `tau_r_ns`.

        rng : numpy.random.Generator, optional
            Random number generator used for sampling.
            If `None`, a new default generator is created.

        Returns
        -------
        directions : ndarray of shape (N, 3)
            Unit vectors representing isotropically sampled photon
            propagation directions.

        t_photon : ndarray of shape (N,)
            Photon emission times in nanoseconds, including the
            scintillation delay.

        Notes
        -----
        The angular distribution is sampled uniformly over the sphere using:

        - cos(theta) ~ U(-1, 1)
        - phi ~ U(0, 2*pi)

        The rejection sampling uses the analytical maximum of the
        scintillation PDF.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(source_points)

        # Isotropic emission
        cos_theta = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

        sin_theta = np.sqrt(1.0 - cos_theta**2)

        directions = np.column_stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ])

        # ==========================================================
        # Scintillation delay with rise time + decay time
        # REJECTION SAMPLING
        #
        # PDF:
        #
        # p(t) propto exp(-t/tau_d) - exp(-t/tau_r)
        #
        # ==========================================================

        scintillation_delay = np.empty(n)

        # Position of the maximum of the distribution
        t_peak = (
            (tau_r_ns * tau_d_ns) /
            (tau_d_ns - tau_r_ns)
        ) * np.log(tau_d_ns / tau_r_ns)

        # Maximum value for rejection sampling
        p_max = (
            np.exp(-t_peak / tau_d_ns)
            - np.exp(-t_peak / tau_r_ns)
        )

        accepted = 0

        while accepted < n:

            # Candidate times sampled from decay exponential
            t_candidate = rng.exponential(tau_d_ns, size=n - accepted)

            # Uniform random numbers
            y = rng.uniform(0.0, p_max, size=n - accepted)

            # Target distribution
            p = (
                np.exp(-t_candidate / tau_d_ns)
                - np.exp(-t_candidate / tau_r_ns)
            )

            mask = y < p

            n_accept = np.sum(mask)

            scintillation_delay[
                accepted:accepted + n_accept
            ] = t_candidate[mask]

            accepted += n_accept

        # Photon emission times
        t_photon = times + scintillation_delay

        return directions, t_photon


    def generate_photon_directions_and_times(
        self,
        photon_positions,
        t_particle,
        tau_ns: float,
        rng=None,
    ):
        """
        !DEPRECATED
        Generate isotropic photon directions and scintillation emission times.

        Each photon is emitted isotropically in 3D. The emission time is obtained
        by adding an exponential scintillation delay to the particle arrival time
        at the photon production point.

        Parameters
        ----------
        photon_positions : np.ndarray, shape (N, 3)
            Photon emission positions. Only the number of photons is used here.
        t_particle : np.ndarray, shape (N,)
            Particle arrival time at each photon emission position, in ns.
        tau_ns : float
            Scintillator decay time constant, in ns.
        rng : np.random.Generator, optional
            NumPy random number generator. If None, a default generator is used.

        Returns
        -------
        directions : np.ndarray, shape (N, 3)
            Isotropic photon direction unit vectors.
        t_photon : np.ndarray, shape (N,)
            Final photon emission times in ns, including scintillation delay.

        Raises
        ------
        ValueError
            If `tau_ns` is negative or if the number of positions and times
            is inconsistent.
        """
        if rng is None:
            rng = np.random.default_rng()

        photon_positions = np.asarray(photon_positions, dtype=float)
        t_particle = np.asarray(t_particle, dtype=float)

        if photon_positions.ndim != 2 or photon_positions.shape[1] != 3:
            raise ValueError("photon_positions must have shape (N, 3).")

        if t_particle.ndim != 1:
            raise ValueError("t_particle must be a 1D array.")

        if len(photon_positions) != len(t_particle):
            raise ValueError(
                "photon_positions and t_particle must contain the same number of photons."
            )

        if tau_ns < 0:
            raise ValueError("Scintillation decay time tau_ns cannot be negative.")

        n = len(photon_positions)

        #NOTE Sample directions uniformly on the unit sphere.
        #
        # For isotropic emission:
        #   cos(theta) must be uniform in [-1, 1]
        #   phi must be uniform in [0, 2*pi]
        cos_theta = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

        sin_theta = np.sqrt(1.0 - cos_theta**2)

        directions = np.column_stack(
            [
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta,
            ]
        )

        # Sample scintillation emission delay from an exponential distribution.
        scintillation_delay = rng.exponential(tau_ns, size=n)

        # Final photon emission time.
        t_photon = t_particle + scintillation_delay

        return directions, t_photon