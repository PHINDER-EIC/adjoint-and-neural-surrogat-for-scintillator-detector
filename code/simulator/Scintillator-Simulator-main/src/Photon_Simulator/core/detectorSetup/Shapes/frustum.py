import numpy as np

from Photon_Simulator.core.detectorSetup.Shapes import ConvexPolyhedronVolume


class FrustumVolume(ConvexPolyhedronVolume):
    """
    Truncated pyramid or frustum volume.

    ``FrustumVolume`` represents a convex solid with two polygonal bases:
    a bottom base and a top base. The two bases must have the same number of
    vertices and their vertices must be ordered consistently.

    The lateral faces are automatically built by connecting corresponding
    vertices of the bottom and top bases.

    Parameters
    ----------
    name : str
        Unique name of the volume.

    material : Material or ReflectiveMaterial
        Material assigned to the volume.

    bottom_vertices : array-like of shape (N, 3)
        Vertices of the bottom base.

        The vertices must be ordered consistently around the base polygon.

    top_vertices : array-like of shape (N, 3)
        Vertices of the top base.

        The number of top vertices must match the number of bottom vertices.
        Vertex ``i`` of the bottom base is connected to vertex ``i`` of the
        top base.

    priority : int, default=0
        Priority used to resolve overlapping or nested volumes.

    Raises
    ------
    ValueError
        If ``bottom_vertices`` and ``top_vertices`` do not contain the same
        number of vertices.

    Notes
    -----
    ``FrustumVolume`` inherits all geometric methods from
    ``ConvexPolyhedronVolume``.

    The orientation of the base vertices affects the direction of the computed
    face normals. Vertices should be ordered consistently to produce outward
    normals.

    Examples
    --------
    Create a square frustum:

    ```python
    bottom = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]

    top = [
        [0.5, 0.5, 2.0],
        [1.5, 0.5, 2.0],
        [1.5, 1.5, 2.0],
        [0.5, 1.5, 2.0],
    ]

    frustum = FrustumVolume(
        name="light_guide",
        material=CORE,
        bottom_vertices=bottom,
        top_vertices=top,
        priority=10,
    )
    ```
    """

    def __init__(self, name, material, bottom_vertices, top_vertices, priority=0):
        bottom = np.asarray(bottom_vertices, dtype=float)
        top = np.asarray(top_vertices, dtype=float)

        if len(bottom) != len(top):
            raise ValueError("Bottom and top bases must have the same number of vertices.")

        n = len(bottom)
        vertices = np.vstack([bottom, top])

        faces = []

        faces.append(list(reversed(range(n))))          # bottom
        faces.append(list(range(n, 2 * n)))             # top

        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j, n + i])          # lateral face

        super().__init__(name, material, vertices, faces, priority)