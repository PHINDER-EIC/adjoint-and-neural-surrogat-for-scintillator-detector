from Photon_Simulator.core.detectorSetup.Shapes.ConvexPolyhedronVolume import ConvexPolyhedronVolume
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from numba import njit

@njit
def box_ray_intersections(box_min, box_max, start, direction, eps=1e-9):
    norm = np.sqrt(
        direction[0] ** 2 +
        direction[1] ** 2 +
        direction[2] ** 2
    )

    direction = direction / norm

    ts = np.empty(6, dtype=np.float64)
    points = np.empty((6, 3), dtype=np.float64)
    normals = np.empty((6, 3), dtype=np.float64)

    count = 0

    for axis in range(3):
        for side_index in range(2):
            side = box_min[axis] if side_index == 0 else box_max[axis]

            if abs(direction[axis]) < 1e-12:
                continue

            t = (side - start[axis]) / direction[axis]

            if t <= eps:
                continue

            point = start + t * direction

            inside = True
            for i in range(3):
                if i == axis:
                    continue

                if point[i] < box_min[i] - eps or point[i] > box_max[i] + eps:
                    inside = False
                    break

            if inside:
                normal = np.zeros(3, dtype=np.float64)
                normal[axis] = 1.0 if side_index == 1 else -1.0

                ts[count] = t
                points[count, :] = point
                normals[count, :] = normal

                count += 1

    order = np.argsort(ts[:count])

    return (
        ts[:count][order],
        points[:count][order],
        normals[:count][order]
    )

class BoxVolume(ConvexPolyhedronVolume):
    """
    Axis-aligned rectangular box volume.

    ``BoxVolume`` represents a rectangular parallelepiped whose faces are
    aligned with the Cartesian axes. It is the simplest and most commonly used
    detector geometry.

    The box is defined by two opposite corners: ``box_min`` and ``box_max``.
    The constructor automatically computes the actual minimum and maximum
    coordinates along each axis, so the two points may be provided in any
    order.

    Parameters
    ----------
    name : str
        Unique name of the volume.

    material : Material or ReflectiveMaterial
        Material assigned to the volume.

    box_min : array-like of shape (3,)
        Coordinates of one corner of the box.

    box_max : array-like of shape (3,)
        Coordinates of the opposite corner of the box.

    priority : int, default=0
        Priority used to resolve overlapping or nested volumes.

        Higher priority volumes override lower priority volumes.

    Attributes
    ----------
    box_min : ndarray of shape (3,)
        Minimum coordinate of the box along each axis.

    box_max : ndarray of shape (3,)
        Maximum coordinate of the box along each axis.

    vertices : ndarray of shape (8, 3)
        Vertices of the box.

    faces : list of list of int
        Six quadrilateral faces of the box.

    Notes
    -----
    ``BoxVolume`` inherits from ``ConvexPolyhedronVolume`` but overrides some
    geometric operations with box-specific implementations.

    Since the box is axis-aligned, it is suitable for rectangular detector
    components such as scintillators, fibers, guides or wrapping layers aligned
    with the global coordinate axes.

    Examples
    --------
    Create a scintillator box:

    ```python
    scintillator = BoxVolume(
        name="scintillator",
        material=EJ200,
        box_min=[0.0, 0.0, 0.0],
        box_max=[2.0, 5.0, 2.0],
        priority=0,
    )
    ```

    Create a fiber core nested inside another volume:

    ```python
    fiber_core = BoxVolume(
        name="fiber_core",
        material=CORE,
        box_min=[0.0, 0.0, 2.001],
        box_max=[2.0, 5.2, 2.099],
        priority=20,
    )
    ```
    """
    def __init__(self, name, material, box_min, box_max, priority=0):
        mn = np.minimum(box_min, box_max).astype(float)
        mx = np.maximum(box_min, box_max).astype(float)

        x0, y0, z0 = mn
        x1, y1, z1 = mx

        vertices = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ])

        faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]

        self.box_min = mn
        self.box_max = mx

        super().__init__(name, material, vertices, faces, priority)
    def contains(self, point, tol=1e-9):
        p = np.asarray(point)
        return np.all(p >= self.box_min - tol) and np.all(p <= self.box_max + tol)

    def ray_intersections(self, start, direction):
        """
        Return all intersections between a ray and the box faces.

        Parameters
        ----------
        start : array-like
            Ray origin.
        direction : array-like
            Ray direction.

        Returns
        -------
        list of tuple
            List of (t, point, normal) for each valid intersection, where:
            - t is the distance along the ray,
            - point is the intersection point,
            - normal is the outward face normal.
        """
        ts, points, normals = box_ray_intersections(
            self.box_min,
            self.box_max,
            np.asarray(start, dtype=float),
            np.asarray(direction, dtype=float),
        )

        return [
            (ts[i], points[i], normals[i])
            for i in range(len(ts))
        ]
        
        # NOT JITTED METHOD
        #// start = np.asarray(start)
        #// direction = direction / np.linalg.norm(direction)

        #// intersections = []

        #// for axis in range(3):
        #//     for side in [self.box_min[axis], self.box_max[axis]]:
        #//         if abs(direction[axis]) < 1e-12:
        #//             continue

        #//         t = (side - start[axis]) / direction[axis]
        #//         if t <= 1e-9:
        #//             continue

        #//         point = start + t * direction

        #//         # Check if inside face bounds
        #//         other_axes = [i for i in range(3) if i != axis]
        #//         if all(
        #//             self.box_min[i] - 1e-9 <= point[i] <= self.box_max[i] + 1e-9
        #//             for i in other_axes
        #//         ):
        #//             normal = np.zeros(3)
        #//             normal[axis] = 1 if side == self.box_max[axis] else -1
        #//             intersections.append((t, point, normal))

        #// return intersections

    def point_on_edge(self, point, eps: float = 1e-9):
        """
        Return True if point lies on an edge of an axis-aligned BoxVolume.
        """
        p = np.asarray(point, dtype=float)

        mn = self.box_min
        mx = self.box_max

        # Check if point is inside or on the box
        inside = np.all(p >= mn - eps) and np.all(p <= mx + eps)
        if not inside:
            return False

        # Count how many coordinates are close to a boundary plane
        on_min = np.isclose(p, mn, atol=eps, rtol=0.0)
        on_max = np.isclose(p, mx, atol=eps, rtol=0.0)

        n_boundary_planes = np.count_nonzero(on_min | on_max)

        # Edge: exactly two boundary planes
        # Vertex: three boundary planes
        return n_boundary_planes >= 2
    
    

    
    def plot(self, ax, color="tab:blue", alpha=0.25):
        xmin, ymin, zmin = self.box_min
        xmax, ymax, zmax = self.box_max

        vertices = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ])

        faces = [
            [vertices[i] for i in [0, 1, 2, 3]],
            [vertices[i] for i in [4, 5, 6, 7]],
            [vertices[i] for i in [0, 1, 5, 4]],
            [vertices[i] for i in [2, 3, 7, 6]],
            [vertices[i] for i in [1, 2, 6, 5]],
            [vertices[i] for i in [0, 3, 7, 4]],
        ]

        poly = Poly3DCollection(
            faces,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=alpha,
        )

        ax.add_collection3d(poly)

        # Invisible points to help autoscaling
        ax.scatter(*self.box_min, alpha=0)
        ax.scatter(*self.box_max, alpha=0)