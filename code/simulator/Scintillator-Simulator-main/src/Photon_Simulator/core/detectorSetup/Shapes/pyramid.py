import numpy as np

from Photon_Simulator.core.detectorSetup.Shapes import ConvexPolyhedronVolume


class PyramidVolume(ConvexPolyhedronVolume):
    """
    Pyramid volume.

    ``PyramidVolume`` represents a convex pyramid defined by a polygonal base
    and a single apex point. The base may have any number of vertices, provided
    that the resulting solid is convex.

    The constructor automatically creates:

    - one polygonal base face;
    - one triangular lateral face for each edge of the base.

    Parameters
    ----------
    name : str
        Unique name of the volume.

    material : Material or ReflectiveMaterial
        Material assigned to the volume.

    base_vertices : array-like of shape (N, 3)
        Vertices of the pyramid base.

        The vertices must be ordered consistently around the base polygon.

    apex : array-like of shape (3,)
        Coordinates of the pyramid apex.

    priority : int, default=0
        Priority used to resolve overlapping or nested volumes.

    Notes
    -----
    ``PyramidVolume`` inherits all geometric methods from
    ``ConvexPolyhedronVolume``.

    The base face is internally reversed to obtain the expected outward normal
    orientation for the base. Lateral faces are then created by connecting each
    base edge to the apex.

    Examples
    --------
    Create a square pyramid:

    ```python
    base = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]

    pyramid = PyramidVolume(
        name="pyramid",
        material=my_material,
        base_vertices=base,
        apex=[1.0, 1.0, 3.0],
        priority=0,
    )
    ```

    Create a triangular pyramid:

    ```python
    base = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ]

    pyramid = PyramidVolume(
        name="triangular_pyramid",
        material=my_material,
        base_vertices=base,
        apex=[0.5, 0.5, 2.0],
        priority=0,
    )
    ```
    """
    def __init__(self, name, material, base_vertices, apex, priority=0):
        base = np.asarray(base_vertices, dtype=float)
        apex = np.asarray(apex, dtype=float)

        n = len(base)
        vertices = np.vstack([base, apex])
        apex_idx = n

        faces = [list(reversed(range(n)))]

        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, apex_idx])

        super().__init__(name, material, vertices, faces, priority)