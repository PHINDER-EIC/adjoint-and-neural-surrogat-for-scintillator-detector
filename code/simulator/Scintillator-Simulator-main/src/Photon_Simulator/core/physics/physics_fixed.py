"""ParticleInteraction with the corrected scintillation emission delay.

``physics.ParticleInteraction.photon_generator`` samples the delay with a
rejection step that accepts candidates drawn from Exp(tau_d) with probability
p(t)/p_max. The result follows Exp(tau_d)*p(t) instead of the intended

    p(t) ∝ exp(-t/tau_d) - exp(-t/tau_r)

(EJ-200: mean 1.68 ns instead of tau_r + tau_d = 3.0 ns).

This module applies the fix proposed by the simulator's author: the delay is
the sum of two independent exponentials, Exp(tau_r) + Exp(tau_d), whose
density is exactly (exp(-t/tau_d) - exp(-t/tau_r)) / (tau_d - tau_r).
It needs no loop, and it also works when tau_r == tau_d (the original code
divides by zero there).

The original file is unchanged. To use the fix, change the import:

    # from Photon_Simulator.core.physics.physics import ParticleInteraction
    from Photon_Simulator.core.physics.physics_fixed import ParticleInteraction

Everything else (photon source generation, isotropic directions, the order of
the random draws for the directions) is inherited unchanged, so for the same
``rng`` the directions are identical to the original class.

Note: ``core/event/event.py`` (``Event``, ``MultiEvents``) imports the original
class; those code paths keep the old behaviour until their import is changed.
"""

import numpy as np

from Photon_Simulator.core.physics.physics import ParticleInteraction as _OriginalParticleInteraction

__all__ = ["ParticleInteraction"]


class ParticleInteraction(_OriginalParticleInteraction):
    """Drop-in replacement for ``physics.ParticleInteraction`` with the corrected delay."""

    def photon_generator(self, source_points, times, tau_r_ns, tau_d_ns, rng=None):
        """
        Generate isotropically emitted photons and their emission times.

        Directions: uniform on the unit sphere (cos(theta) ~ U(-1, 1),
        phi ~ U(0, 2*pi)), drawn exactly as in the original method.

        Emission delay: Exp(tau_r_ns) + Exp(tau_d_ns), i.e. the scintillation
        pulse p(t) ∝ exp(-t/tau_d) - exp(-t/tau_r) with mean tau_r + tau_d.

        Parameters
        ----------
        source_points : ndarray, shape (N, 3)
            Photon source positions (only N is used).
        times : ndarray, shape (N,)
            Particle arrival time at each source point, in ns.
        tau_r_ns, tau_d_ns : float
            Rise and decay time constants in ns (non-negative; 0 disables that term).
        rng : numpy.random.Generator, optional
            Random generator; a new default generator if None.

        Returns
        -------
        directions : ndarray, shape (N, 3)
        t_photon : ndarray, shape (N,)
            ``times`` plus the scintillation delay, in ns.
        """
        if rng is None:
            rng = np.random.default_rng()
        if tau_r_ns is None or tau_d_ns is None:
            raise ValueError("tau_r_ns and tau_d_ns must be given (material rise_time_ns / decay_time_ns)")
        if tau_r_ns < 0 or tau_d_ns < 0:
            raise ValueError("tau_r_ns and tau_d_ns must be non-negative")

        n = len(source_points)

        # Isotropic emission (same draws, same order as the original method)
        cos_theta = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        directions = np.column_stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ])

        # Scintillation delay: rise + decay (author's fix)
        scintillation_delay = (
            rng.exponential(scale=tau_r_ns, size=n)
            + rng.exponential(scale=tau_d_ns, size=n)
        )

        t_photon = times + scintillation_delay
        return directions, t_photon
