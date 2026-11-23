import numpy as np

from Photon_Simulator.core.detectorSetup.Shapes.frustum import FrustumVolume


class PrismVolume(FrustumVolume):
    """
    Generic prism volume.

    ``PrismVolume`` represents a prism obtained by translating a polygonal
    base by a fixed height vector. The bottom base is provided explicitly,
    while the top base is automatically computed as:

    ``top_vertices = base_vertices + height_vector``

    This class is a convenience wrapper around ``FrustumVolume`` for the
    common case where the two bases have identical shape and orientation.

    Parameters
    ----------
    name : str
        Unique name of the volume.

    material : Material or ReflectiveMaterial
        Material assigned to the volume.

    base_vertices : array-like of shape (N, 3)
        Vertices of the prism base.

        The vertices must be ordered consistently around the polygon.

    height_vector : array-like of shape (3,)
        Translation vector from the bottom base to the top base.

        This vector defines both the height and the direction of the prism.

    priority : int, default=0
        Priority used to resolve overlapping or nested volumes.

    Notes
    -----
    ``PrismVolume`` inherits its geometry handling from ``FrustumVolume`` and
    ``ConvexPolyhedronVolume``.

    The prism is not required to be aligned with the Cartesian axes. By
    choosing an arbitrary ``height_vector``, inclined prisms can be created.

    Examples
    --------
    Create a triangular prism:

    ```python
    base = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ]

    prism = PrismVolume(
        name="triangular_prism",
        material=my_material,
        base_vertices=base,
        height_vector=[0.0, 0.0, 2.0],
        priority=0,
    )
    ```

    Create an inclined prism:

    ```python
    inclined_prism = PrismVolume(
        name="inclined_prism",
        material=my_material,
        base_vertices=base,
        height_vector=[0.5, 0.0, 2.0],
        priority=0,
    )
    ```
    """
    def __init__(self, name, material, base_vertices, height_vector, priority=0):
        bottom = np.asarray(base_vertices, dtype=float)
        height_vector = np.asarray(height_vector, dtype=float)

        top = bottom + height_vector

        super().__init__(name, material, bottom, top, priority)