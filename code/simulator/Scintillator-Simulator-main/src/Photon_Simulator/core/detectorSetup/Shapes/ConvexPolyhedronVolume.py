import numpy as np

from Photon_Simulator.core.detectorSetup.volume import Volume
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ConvexPolyhedronVolume(Volume):
    """
    Generic convex polyhedron volume.

    This class represents a closed convex polyhedron defined by a set of
    vertices and polygonal faces. It can be used directly to describe
    arbitrary convex detector components, or as a base class for more
    specific geometries such as boxes, frustums, prisms and pyramids.

    Each face must be provided as a list of vertex indices. The order of the
    vertices in each face is important: vertices should be ordered
    counter-clockwise when the face is viewed from outside the volume. This
    convention is used to compute outward-facing surface normals.

    Parameters
    ----------
    name : str
        Unique name of the volume.

        Some names may have a special meaning in the simulation. For example,
        the default scene configuration expects volumes named ``"scintillator"``
        and ``"fiber_core"``.

    material : Material or ReflectiveMaterial
        Material assigned to the volume.

        Use a standard ``Material`` for transmissive/refractive media and a
        ``ReflectiveMaterial`` for reflective or absorbing surfaces.

    vertices : array-like of shape (N, 3)
        Coordinates of the polyhedron vertices.

        Each row represents one vertex in Cartesian coordinates:

        ``[x, y, z]``

    faces : list of list of int
        List of polygonal faces.

        Each face is described by the indices of the vertices that form it.
        The indices refer to rows of the ``vertices`` array.

        Example
        -------
        A quadrilateral face using vertices 0, 1, 2 and 3 is written as:

        ``[0, 1, 2, 3]``

    priority : int, default=0
        Priority used to resolve overlapping or nested volumes.

        Higher priority volumes are selected before lower priority volumes.
        Typically, outer volumes have lower priority and inner volumes have
        higher priority.

    Attributes
    ----------
    vertices : ndarray of shape (N, 3)
        Vertex coordinates converted to a NumPy array.

    faces : list of list of int
        Face definitions.

    face_normals : ndarray of shape (M, 3)
        Outward normal vector for each face.

    face_points : ndarray of shape (M, 3)
        One reference point for each face, used for geometric tests.

    Notes
    -----
    This class assumes that the input geometry is convex. Non-convex
    polyhedra are not supported and may produce incorrect containment or ray
    intersection results.

    The correctness of surface normals depends on the ordering of the face
    vertices. If faces are ordered incorrectly, methods such as ``contains()``
    and ``ray_intersections()`` may return wrong results.

    Examples
    --------
    Create a tetrahedron-like convex polyhedron:

    ```python
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    faces = [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ]

    volume = ConvexPolyhedronVolume(
        name="custom_polyhedron",
        material=my_material,
        vertices=vertices,
        faces=faces,
        priority=0,
    )
    ```
    """

    def __init__(self, name, material, vertices, faces, priority=0):
        super().__init__(name, material, priority)
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = faces

        self.face_normals = []
        self.face_points = []

        for face in faces:
            pts = self.vertices[face]
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            normal = normal / np.linalg.norm(normal)

            self.face_normals.append(normal)
            self.face_points.append(pts[0])

        self.face_normals = np.asarray(self.face_normals)
        self.face_points = np.asarray(self.face_points)

    def contains(self, point, tol=1e-9):
        """
        Check whether a point lies inside or on the boundary of the volume.

        The point is tested against all face planes. Since the volume is assumed
        to be convex, the point is inside the polyhedron if it lies on the inner
        side of every face.

        Parameters
        ----------
        point : array-like of shape (3,)
            Cartesian coordinates of the point to test.

        tol : float, default=1e-9
            Numerical tolerance used when comparing the point position with the
            face planes.

        Returns
        -------
        bool
            ``True`` if the point is inside the volume or on its boundary,
            ``False`` otherwise.

        Notes
        -----
        This method assumes that all face normals point outward.
        """
        p = np.asarray(point, dtype=float)

        for p0, n in zip(self.face_points, self.face_normals):
            if np.dot(p - p0, n) > tol:
                return False

        return True

    def ray_intersections(self, start, direction, eps=1e-9):
        """
        Return all intersections between a ray and the polyhedron faces.

        The ray is defined by an origin point and a direction vector. The method
        computes the intersection between the ray and each face plane, then keeps
        only the intersections that lie inside the corresponding polygonal face.

        Parameters
        ----------
        start : array-like of shape (3,)
            Ray origin.

        direction : array-like of shape (3,)
            Ray direction.

            The vector does not need to be normalized. It is normalized internally.

        eps : float, default=1e-9
            Numerical tolerance.

            Intersections with distance ``t <= eps`` are ignored to avoid
            self-intersections with the current surface.

        Returns
        -------
        list of tuple
            Sorted list of valid intersections.

            Each intersection is returned as:

            ``(t, point, normal)``

            where:

            - ``t`` is the distance along the normalized ray direction;
            - ``point`` is the intersection point;
            - ``normal`` is the outward normal of the intersected face.

        Notes
        -----
        The returned list is sorted by increasing ``t``.
        """
        start = np.asarray(start, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        intersections = []

        for face, p0, n in zip(self.faces, self.face_points, self.face_normals):
            denom = np.dot(direction, n)

            if abs(denom) < eps:
                continue

            t = np.dot(p0 - start, n) / denom

            if t <= eps:
                continue

            point = start + t * direction

            if self._point_in_face(point, face, n, eps):
                intersections.append((t, point, n))

        intersections.sort(key=lambda x: x[0])
        return intersections

    def _point_in_face(self, point, face, normal, eps=1e-9):
        pts = self.vertices[face]
        m = len(pts)

        for i in range(m):
            a = pts[i]
            b = pts[(i + 1) % m]

            edge = b - a
            vp = point - a

            if np.dot(np.cross(edge, vp), normal) < -eps:
                return False

        return True

    def point_on_edge(self, point, eps=1e-9):
        """
        Check whether a point lies on an edge of the polyhedron.
    
        The method tests the distance between the input point and each segment
        belonging to each face of the polyhedron.
    
        Parameters
        ----------
        point : array-like of shape (3,)
            Cartesian coordinates of the point to test.
    
        eps : float, default=1e-9
            Numerical tolerance used to decide whether the point lies on an edge.
    
        Returns
        -------
        bool
            ``True`` if the point lies on one of the polyhedron edges,
            ``False`` otherwise.
        """
        p = np.asarray(point, dtype=float)

        for face in self.faces:
            pts = self.vertices[face]

            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]

                ab = b - a
                ap = p - a

                t = np.dot(ap, ab) / np.dot(ab, ab)

                if -eps <= t <= 1 + eps:
                    closest = a + t * ab

                    if np.linalg.norm(p - closest) <= eps:
                        return True

        return False
    def plot(self, ax, color="tab:blue", alpha=0.25):
        face_vertices = []

        for face in self.faces:
            face_vertices.append([self.vertices[i] for i in face])

        poly = Poly3DCollection(
            face_vertices,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=alpha,
        )

        ax.add_collection3d(poly)

        # Invisible points to help autoscaling
        ax.scatter(
            self.vertices[:, 0],
            self.vertices[:, 1],
            self.vertices[:, 2],
            alpha=0,
        )
    
    def face_shells(self, material, thickness: float, name_prefix: str = "shell", priority: int = 1):
        """
        Return one thin ConvexPolyhedronVolume for each face.
    
        Each shell is built by extruding the face outward along its normal.
        This is better than BoxVolume for inclined faces, because BoxVolume is axis-aligned.
        """
        shells = []
    
        for face_index, face in enumerate(self.faces):
            face = list(face)
    
            inner_vertices = self.vertices[face]
            normal = self.face_normals[face_index]
    
            outer_vertices = inner_vertices + thickness * normal
    
            vertices = np.vstack([inner_vertices, outer_vertices])
    
            n = len(face)
    
            # Local indices:
            # inner: 0 ... n-1
            # outer: n ... 2n-1
            shell_faces = []
    
            # Inner face, facing inward relative to shell
            shell_faces.append(list(reversed(range(n))))
    
            # Outer face
            shell_faces.append(list(range(n, 2 * n)))
    
            # Side faces
            for i in range(n):
                j = (i + 1) % n
                shell_faces.append([i, j, n + j, n + i])
    
            shell = ConvexPolyhedronVolume(
                name=f"{name_prefix}_{self.name}_face_{face_index}",
                material=material,
                vertices=vertices,
                faces=shell_faces,
                priority=priority,
            )
    
            shells.append(shell)
    
        return shells