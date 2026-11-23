"""Explicit optical readout-port definitions.

A readout port is a terminal face of one logical guide.  The photon engine
recognises a photon as caught only when it leaves the named terminal volume
through the specified plane while moving along the outward normal.

The representation is deliberately geometric and independent of global axes,
so the same interface can be used later for rotated / segmented guides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True, slots=True)
class ReadoutPort:
    """One terminal optical readout face.

    Parameters
    ----------
    channel:
        Integer channel written by the PHINDER bridge (0, 1, ...).
    volume_name:
        Unique name of the *terminal guide volume* from which the photon exits.
    point:
        Any point on the readout plane, in the detector geometry units.
    outward_normal:
        Outward-pointing normal of the readout plane.  It is normalized at
        construction time.
    tolerance:
        Plane-matching tolerance.  The engine uses the larger of this value and
        TrackingConfig.catch_tolerance.
    label:
        Optional human-readable name used only for diagnostics.
    """

    channel: int
    volume_name: str
    point: tuple[float, float, float]
    outward_normal: tuple[float, float, float]
    tolerance: float = 1.0e-6
    label: str = ""

    def __post_init__(self) -> None:
        if self.channel < 0:
            raise ValueError("ReadoutPort.channel must be >= 0")
        if not self.volume_name:
            raise ValueError("ReadoutPort.volume_name cannot be empty")
        if self.tolerance <= 0.0:
            raise ValueError("ReadoutPort.tolerance must be positive")

        p = np.asarray(self.point, dtype=float)
        n = np.asarray(self.outward_normal, dtype=float)
        if p.shape != (3,) or n.shape != (3,):
            raise ValueError("ReadoutPort point and outward_normal must have length 3")
        norm = float(np.linalg.norm(n))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("ReadoutPort.outward_normal must be non-zero and finite")
        if not np.all(np.isfinite(p)):
            raise ValueError("ReadoutPort.point must be finite")

        n = n / norm
        object.__setattr__(self, "point", tuple(float(x) for x in p))
        object.__setattr__(self, "outward_normal", tuple(float(x) for x in n))

    @property
    def plane_point(self) -> np.ndarray:
        return np.asarray(self.point, dtype=float)

    @property
    def normal(self) -> np.ndarray:
        return np.asarray(self.outward_normal, dtype=float)

    def point_on_plane(self, point: Sequence[float], tolerance: float | None = None) -> bool:
        tol = self.tolerance if tolerance is None else max(self.tolerance, float(tolerance))
        p = np.asarray(point, dtype=float)
        return bool(abs(float(np.dot(p - self.plane_point, self.normal))) <= tol)

    def moving_outward(self, direction: Sequence[float]) -> bool:
        d = np.asarray(direction, dtype=float)
        return bool(float(np.dot(d, self.normal)) > 0.0)

    def matches_hit(self, hit, direction: Sequence[float], tolerance: float | None = None) -> bool:
        """Return True when ``hit`` exits this port's terminal volume."""
        volume = getattr(hit, "from_volume", None)
        if volume is None or volume.name != self.volume_name:
            return False
        return self.point_on_plane(hit.point, tolerance) and self.moving_outward(direction)

    def matches_terminal_state(self, point, direction, volume, tolerance: float | None = None) -> bool:
        """Match a final CAUGHT state after tracking.

        This is used by the bridge to recover the channel from the final photon
        state without encoding channel IDs into PhotonStatus.
        """
        if volume is None or volume.name != self.volume_name:
            return False
        # ``contains`` includes the boundary with a small tolerance.
        if not volume.contains(np.asarray(point, dtype=float), tol=max(self.tolerance, 1.0e-8)):
            return False
        return self.point_on_plane(point, tolerance) and self.moving_outward(direction)
