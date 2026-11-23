from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Sequence

from Photon_Simulator.config.defaults import EPS_NEW_VOLUME_DETECTION
from .material import Material, PlasticScintillator
from .volume import Volume
from .readout import ReadoutPort
from .Shapes.box import BoxVolume
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
from matplotlib.patches import Patch

#ANCHOR - BOUNDARY HIT STRUCTURE
@dataclass
class BoundaryHit:
    point: np.ndarray
    normal: np.ndarray
    distance: float
    from_volume: Optional[Volume]
    to_volume: Optional[Volume]
    from_material: Material
    to_material: Material


class Scene:
    def __init__(
        self,
        volumes: List["Volume"],
        outside_material: Material,
        scintillator_material: PlasticScintillator,
        readout_ports: Optional[Sequence[ReadoutPort]] = None,
    ):
        """
        Parameters
        ----------
        volumes : list of Volume
            All volumes in the scene (can be overlapping / nested).
        outside_material : Material
            Material outside all volumes (typically air).
        """
    
        self.volumes = list(volumes)
    
        volume_names = {v.name for v in self.volumes}

        # ``scintillator`` remains a required semantic volume.  The historical
        # ``fiber_core`` name is required only for legacy scenes that do not
        # declare explicit readout ports.
        if "scintillator" not in volume_names:
            raise ValueError("A Scene must contain a volume named 'scintillator'.")

        self.readout_ports = tuple(readout_ports or ())
        self._readout_volumes = {}

        if self.readout_ports:
            # Explicit ports require unique terminal-volume names.  Internal
            # guide segments may have any names; only terminal volumes appear
            # here.
            for port in self.readout_ports:
                matches = [v for v in self.volumes if v.name == port.volume_name]
                if len(matches) != 1:
                    raise ValueError(
                        f"Readout port channel {port.channel} references volume "
                        f"'{port.volume_name}', but {len(matches)} matching volumes exist. "
                        "Terminal readout volume names must be unique."
                    )
                self._readout_volumes[port.volume_name] = matches[0]
        elif "fiber_core" not in volume_names:
            raise ValueError(
                "Legacy Scene without explicit readout_ports must contain a "
                "volume named 'fiber_core'."
            )
    
        self._volumes_by_priority = sorted(
            self.volumes,
            key=lambda v: v.priority,
            reverse=True,
        )
    
        self.outside_material = outside_material
        self.scintillator = scintillator_material

    # ---->  Locate current volume <----
    # def locate_volume(self, point: np.ndarray) -> Optional["Volume"]:
        # candidates = [v for v in self.volumes if v.contains(point)]
# 
        # if not candidates:
            # return None
# 
        # return max(candidates, key=lambda v: v.priority)
    def locate_volume(self, point: np.ndarray) -> Optional["Volume"]:
        """
        Return the most 'inner' volume containing the point.
        Uses priority to resolve nested volumes.
        """
        for volume in self._volumes_by_priority:
            if volume.contains(point):
                return volume
        return None
    
    # ---->  First boundary hit <----
    def _top_volume(self, active: set["Volume"]) -> Optional["Volume"]:
        for volume in self._volumes_by_priority:
            if volume in active:
                return volume
        return None
    
    
    # def first_boundary_hit(
    #     self,
    #     start: np.ndarray,
    #     direction: np.ndarray,
    #     max_distance: float,
    #     t_min: float = 1e-9,
    #     group_tol: float = 1e-8,
    #     normal_tol: float = 1e-12,
    # ) -> Optional[BoundaryHit]:
    
    #     direction = direction / np.linalg.norm(direction)
    
    #     active = {v for v in self.volumes if v.contains(start)}
    #     current_volume = self._top_volume(active)
    
    #     events = []
    
    #     for volume in self.volumes:
    #         for dist, point, normal in volume.ray_intersections(start, direction):
    #             if t_min < dist <= max_distance:
    #                 events.append((dist, volume, normal))
    
    #     if not events:
    #         return None
    
    #     events.sort(key=lambda x: x[0])
    
    #     i = 0
    #     n = len(events)
    
    #     while i < n:
    #         t = events[i][0]
    #         group = []
    
    #         while i < n and abs(events[i][0] - t) <= group_tol:
    #             group.append(events[i])
    #             i += 1
    
    #         before_volume = self._top_volume(active)
    #         before_material = (
    #             self.outside_material if before_volume is None else before_volume.material
    #         )
    
    #         new_active = set(active)
    
    #         for _, volume, normal in group:
    #             s = float(np.dot(normal, direction))
    
    #             if s < -normal_tol:
    #                 # Ray crosses inward through outward normal
    #                 new_active.add(volume)
    #             elif s > normal_tol:
    #                 # Ray crosses outward
    #                 new_active.discard(volume)
    #             else:
    #                 # Tangential hit: no topological crossing
    #                 pass
                
    #         after_volume = self._top_volume(new_active)
    #         after_material = (
    #             self.outside_material if after_volume is None else after_volume.material
    #         )
    
    #         if after_volume is not before_volume:
    #             point = start + t * direction
    
    #             normal = None
    
    #             if before_volume is not None:
    #                 for _, volume, candidate_normal in group:
    #                     if volume is before_volume:
    #                         normal = candidate_normal
    #                         break
                        
    #             if normal is None and after_volume is not None:
    #                 for _, volume, candidate_normal in group:
    #                     if volume is after_volume:
    #                         normal = -candidate_normal
    #                         break
                        
    #             if normal is None:
    #                 normal = group[0][2]
    
    #             return BoundaryHit(
    #                 point=point,
    #                 normal=normal,
    #                 distance=t,
    #                 from_volume=before_volume,
    #                 to_volume=after_volume,
    #                 from_material=before_material,
    #                 to_material=after_material,
    #             )
    
    #         active = new_active
    
    #     return None
    def first_boundary_hit(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        # max_distance: float,
        # t_min: float = 1e-9,
        # group_tol: float = 1e-8,
        # eps: float = 1e-7,
    ) -> Optional[BoundaryHit]:
        #ANCHOR - FIRST BOUNDARY HIT 
        direction = direction / np.linalg.norm(direction)
    
        hits = []
    
        for volume in self.volumes:
            intersections = tuple(volume.ray_intersections(start, direction))
            if intersections:
                hits.append(intersections[0])
    
        if not hits:
            return None
    
        hits.sort(key=lambda x: x[0])
    
        distance = hits[0][0]
        normal = hits[0][2]
    
        # if d > max_distance:
            # return None
    
        # Point strictly before the first boundary
        from_point = start + 0.5 * distance * direction
        from_volume = self.locate_volume(from_point)
        from_material = (
            self.outside_material if from_volume is None else from_volume.material
        )
    
        # Boundary point
        hit_point = start + distance * direction
        if self.locate_volume(hit_point) is None:
            raise Exception(f"Location Error at point {start} vs {direction}")
        

        hits = []
        for volume in self.volumes:
            intersections = tuple(volume.ray_intersections(hit_point, direction))
            if intersections:
                hits.append(intersections[0])
    
        
        if not hits:
            to_volume = None
            to_material = self.outside_material
            d = None
        else:
            hits.sort(key=lambda x: x[0])
            d = hits[0][0]
            # if d > max_distance:
                # return None

            # Point strictly before the first boundary
            to_point = hit_point + 0.5 * d * direction
            to_volume = self.locate_volume(to_point)
            to_material = (
                self.outside_material if to_volume is None else to_volume.material
            )
    
        return BoundaryHit(
            point=hit_point,
            normal=normal,
            distance=distance,
            from_volume=from_volume,
            to_volume=to_volume,
            from_material=from_material,
            to_material=to_material,
        )
    # def first_boundary_hit(
    #     self,
    #     start: np.ndarray,
    #     direction: np.ndarray,
    #     max_distance: float
    # ) -> Optional[BoundaryHit]:
    #     """
    #     Find the first boundary crossing along the ray.
    #     """

    #     direction = direction / np.linalg.norm(direction)

    #     current_volume = self.locate_volume(start)
    #     current_material = (
    #         self.outside_material if current_volume is None else current_volume.material
    #     )

    #     hits = []

    #     # Collect all intersections with all volumes
    #     for volume in self.volumes:
    #         intersections = volume.ray_intersections(start, direction)

    #         for dist, point, normal in intersections:
    #             if dist <= 1e-9:
    #                 continue
    #             if dist > max_distance:
    #                 continue

    #             hits.append((dist, point, normal, volume))

    #     if not hits:
    #         return None

    #     # Sort by distance
    #     hits.sort(key=lambda x: x[0])

    #     # Find first *valid* crossing
    #     for dist, point, normal, volume in hits:

    #         # Move slightly forward to detect new region
    #         eps = EPS_NEW_VOLUME_DETECTION
    #         probe_point = point + eps * direction

    #         new_volume = self.locate_volume(probe_point)
    #         new_material = (
    #             self.outside_material if new_volume is None else new_volume.material
    #         )

    #         if new_material != current_material:
    #             return BoundaryHit(
    #                 point=point,
    #                 normal=normal,
    #                 distance=dist,
    #                 from_volume=current_volume,
    #                 to_volume=new_volume,
    #                 from_material=current_material,
    #                 to_material=new_material
    #             )

    #     return None

    # ---->  Convenience <----
    def get_material(self, point: np.ndarray) -> Material:
        volume = self.locate_volume(point)
        return self.outside_material if volume is None else volume.material
    
    def get_scintillator(self):
        return self.scintillator

    # ---->  Explicit optical readout ports <----
    def readout_channel(self, hit: BoundaryHit, direction: np.ndarray, tolerance: float | None = None) -> Optional[int]:
        """Return the matched readout channel for a boundary crossing.

        ``None`` means that the boundary is not a terminal readout port.  More
        than one match is treated as a geometry-definition error.
        """
        matches = [
            port.channel
            for port in self.readout_ports
            if port.matches_hit(hit, direction, tolerance)
        ]
        if len(matches) > 1:
            raise RuntimeError(f"Boundary hit matches multiple readout ports: {matches}")
        return matches[0] if matches else None

    def readout_channel_at(self, point: np.ndarray, direction: np.ndarray, tolerance: float | None = None) -> Optional[int]:
        """Recover the readout channel from a final CAUGHT photon state."""
        matches = []
        for port in self.readout_ports:
            volume = self._readout_volumes.get(port.volume_name)
            if port.matches_terminal_state(point, direction, volume, tolerance):
                matches.append(port.channel)
        if len(matches) > 1:
            raise RuntimeError(f"Final photon state matches multiple readout ports: {matches}")
        return matches[0] if matches else None
        
        # ---->  PLOT SCENE <----
    def plot(
        self,
        alpha=0.3,
        figsize=(10, 8),
        axis_length=1.0,
        ax=None,
        show=True,
        show_labels=True,
        show_legend=True,
        label_offset_factor=0.15,
    ):
        # --------------------------------------------------
        # Create figure if needed
        # --------------------------------------------------

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        cmap = plt.get_cmap("tab10")

        # --------------------------------------------------
        # Compute global bounds
        # --------------------------------------------------

        mins = []
        maxs = []

        for volume in self.volumes:
            if isinstance(volume, BoxVolume):
                mins.append(volume.box_min)
                maxs.append(volume.box_max)

        mins = np.asarray(mins)
        maxs = np.asarray(maxs)

        global_min = mins.min(axis=0)
        global_max = maxs.max(axis=0)

        scene_center = 0.5 * (global_min + global_max)

        # scene_size = np.linalg.norm(global_max - global_min)

        # label_offset = label_offset_factor * scene_size

        legend_handles = []

        # --------------------------------------------------
        # Draw volumes
        # --------------------------------------------------

        for i, volume in enumerate(self.volumes):

            color = cmap(i % 10)

            volume.plot(
                ax,
                color=color,
                alpha=alpha,
            )

            material_name = getattr(
                volume.material,
                "name",
                str(volume.material),
            )

            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    alpha=alpha,
                    label=f"{volume.name} [{material_name}]",
                )
            )

            if not show_labels:
                continue

            if not isinstance(volume, BoxVolume):
                continue

            center = 0.5 * (volume.box_min + volume.box_max)

            direction = center - scene_center

            norm = np.linalg.norm(direction)

            if norm < 1e-12:
                direction = np.array([1.0, 1.0, 1.0])
                norm = np.linalg.norm(direction)

            direction /= norm

            # label_position = center + label_offset * direction

            # connector line

            # ax.plot(
            #     [center[0], label_position[0]],
            #     [center[1], label_position[1]],
            #     [center[2], label_position[2]],
            #     linestyle="--",
            #     linewidth=1.2,
            #     color=color,
            # )

            # # label

            # ax.text(
            #     label_position[0],
            #     label_position[1],
            #     label_position[2],
            #     f"{volume.name}\n{material_name}",
            #     fontsize=8,
            #     color=color,
            #     ha="center",
            #     va="center",
            #     bbox=dict(
            #         boxstyle="round,pad=0.25",
            #         facecolor="white",
            #         edgecolor=color,
            #         alpha=0.85,
            #     ),
            # )
        outside_name = getattr(
        self.outside_material,
            "name",
            str(self.outside_material)
        )

        legend_handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                alpha=0.5,
                label=f"outside [{outside_name}]",
            )
        )
        # --------------------------------------------------
        # Global axes
        # --------------------------------------------------

        ax.quiver(
            0, 0, 0,
            axis_length, 0, 0,
            color="red",
            linewidth=2,
            arrow_length_ratio=0.1,
        )

        ax.quiver(
            0, 0, 0,
            0, axis_length, 0,
            color="green",
            linewidth=2,
            arrow_length_ratio=0.1,
        )

        ax.quiver(
            0, 0, 0,
            0, 0, axis_length,
            color="blue",
            linewidth=2,
            arrow_length_ratio=0.1,
        )

        ax.text(
            axis_length,
            0,
            0,
            "+X",
            color="red",
            fontsize=11,
            fontweight="bold",
        )

        ax.text(
            0,
            axis_length,
            0,
            "+Y",
            color="green",
            fontsize=11,
            fontweight="bold",
        )

        ax.text(
            0,
            0,
            axis_length,
            "+Z",
            color="blue",
            fontsize=11,
            fontweight="bold",
        )

        # origin

        ax.scatter(
            0,
            0,
            0,
            color="black",
            s=40,
        )

        ax.text(
            0,
            0,
            0,
            "O",
            color="black",
            fontweight="bold",
        )

        # --------------------------------------------------
        # Axes style
        # --------------------------------------------------

        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        ax.set_zlabel("Z [cm]")

        ax.set_title("Detector Geometry")

        ax.grid(True)

        self._set_axes_equal(ax)

        # --------------------------------------------------
        # Legend
        # --------------------------------------------------

        if show_legend:

            ax.legend(
                handles=legend_handles,
                title="Volumes",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0,
            )

        plt.tight_layout()

        if show:
            plt.show()
        return ax

    @staticmethod
    def _set_axes_equal(ax):
        """
        Make 3D axes have equal scale.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_plotly(
        self,
        opacity: float = 0.35,
        show_axes: bool = True,
        axis_length: float | None = None,
        show_legend: bool = True,
    ):
        import numpy as np
        import plotly.express as px

        fig = go.Figure()

        colors = px.colors.qualitative.Plotly

        box_volumes = [
            volume
            for volume in self.volumes
            if isinstance(volume, BoxVolume)
        ]

        if len(box_volumes) == 0:
            return fig

        mins = np.asarray(
            [v.box_min for v in box_volumes],
            dtype=float,
        )

        maxs = np.asarray(
            [v.box_max for v in box_volumes],
            dtype=float,
        )

        global_min = mins.min(axis=0)
        global_max = maxs.max(axis=0)

        scene_size = np.linalg.norm(global_max - global_min)

        if axis_length is None:
            axis_length = 0.2 * scene_size

        for idx, volume in enumerate(box_volumes):

            color = colors[idx % len(colors)]

            x0, y0, z0 = np.asarray(volume.box_min, dtype=float)
            x1, y1, z1 = np.asarray(volume.box_max, dtype=float)

            x = [x0, x1, x1, x0, x0, x1, x1, x0]
            y = [y0, y0, y1, y1, y0, y0, y1, y1]
            z = [z0, z0, z0, z0, z1, z1, z1, z1]

            i = [
                0, 0,
                4, 4,
                0, 0,
                3, 3,
                0, 0,
                1, 1,
            ]

            j = [
                1, 2,
                5, 6,
                1, 5,
                2, 6,
                3, 7,
                2, 6,
            ]

            k = [
                2, 3,
                6, 7,
                5, 4,
                6, 7,
                7, 4,
                6, 5,
            ]

            material_name = getattr(
                volume.material,
                "name",
                str(volume.material),
            )

            # actual mesh
            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    color=color,
                    opacity=opacity,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{volume.name}</b><br>"
                        f"material: {material_name}<br>"
                        f"x=[{x0:.6g},{x1:.6g}]<br>"
                        f"y=[{y0:.6g},{y1:.6g}]<br>"
                        f"z=[{z0:.6g},{z1:.6g}]"
                        "<extra></extra>"
                    ),
                )
            )

            # wireframe
            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]

            ex = []
            ey = []
            ez = []

            for a, b in edges:
                ex.extend([x[a], x[b], None])
                ey.extend([y[a], y[b], None])
                ez.extend([z[a], z[b], None])

            fig.add_trace(
                go.Scatter3d(
                    x=ex,
                    y=ey,
                    z=ez,
                    mode="lines",
                    line=dict(
                        color="black",
                        width=3,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # legend item
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color,
                    ),
                    name=f"{volume.name} [{material_name}]",
                    showlegend=show_legend,
                    hoverinfo="skip",
                )
            )

        # outside material in legend

        outside_name = getattr(
            self.outside_material,
            "name",
            str(self.outside_material),
        )

        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color="white",
                    line=dict(
                        color="black",
                        width=1,
                    ),
                ),
                name=f"outside [{outside_name}]",
                showlegend=show_legend,
                hoverinfo="skip",
            )
        )

        # coordinate axes

        if show_axes:

            fig.add_trace(
                go.Scatter3d(
                    x=[0, axis_length],
                    y=[0, 0],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="red", width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[0, 0],
                    y=[0, axis_length],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="green", width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[0, 0],
                    y=[0, 0],
                    z=[0, axis_length],
                    mode="lines",
                    line=dict(color="blue", width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            title="Detector Geometry",
            showlegend=show_legend,
            scene=dict(
                xaxis_title="X [cm]",
                yaxis_title="Y [cm]",
                zaxis_title="Z [cm]",
                aspectmode="data",
            ),
            legend=dict(
                title="Volumes",
                x=1.02,
                y=1,
            ),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=40,
            ),
        )

        return fig