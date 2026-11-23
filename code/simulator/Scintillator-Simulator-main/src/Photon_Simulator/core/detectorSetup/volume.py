
from abc import ABC, abstractmethod
import numpy as np
from .material import Material


class Volume(ABC):
    """
    With this class i want the definition of a volume with:
    - its geometry
    - its material
    - if a point is inside
    - where a ray exit from it
    - the normal to the surface
    """
    
    def __init__(self, name: str, material: Material, priority:int = 0):
        self.name = name
        self.material = material
        self.priority = priority

    @abstractmethod
    def contains(self, point: np.ndarray, tol: float = 1e-9) -> bool:
        pass
    @abstractmethod
    def ray_intersections(self, start: np.ndarray, direction: np.ndarray):
        """
        Return all intersections between a ray and the object.

        Parameters
        ----------
        start : array-like of shape (3,)
            Ray origin.
        direction : array-like of shape (3,)
            Ray direction.

        Returns
        -------
        list of tuple
            Collection of intersections represented as
            ``(t, point, normal)``, where:

            - ``t`` is the ray parameter,
            - ``point`` is the intersection position,
            - ``normal`` is the *outward* surface normal at the intersection .

            An empty list is returned when no intersections are found.
        """
        pass
    @abstractmethod
    def point_on_edge(self, point):
        """
        Return True if a point is on an edge
        """
        pass


    @abstractmethod
    def plot(self, ax, color="tab:blue", alpha=0.25):
        raise NotImplementedError("Each Volume subclass must implement plot().")