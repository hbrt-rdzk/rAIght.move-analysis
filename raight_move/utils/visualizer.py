import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import yaml
from matplotlib.figure import Figure
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from processors.angles.angle import Angle
from processors.joints.joint import Joint

ELEV = -90
AZIM = -90
X_LIM = (-1, 1)
Y_LIM = (-1, 1)
Z_LIM = (-1, 1)
JOINTS_COLOR = (2, 82, 212)
CONNECTIONS_COLOR = (245, 255, 230)
CONFIG_PATH = "configs/config.yaml"
PROGRESS_BAR_VERTICES = ([1.2, 1, 0.2], [1.2, -1, 0.2], [2, -1, 0.2], [2, 1, 0.2])
INSTRUCTION_MAPPER = {"up": "Go down!", "down": "Go up!"}

mp_drawing = mp.solutions.drawing_utils


class Visualizer:
    def __init__(self, connections: dict) -> None:
        self.figure, self.axis = self.__initialize_3D_joints_figure()
        self.connections = connections

    @staticmethod
    def __initialize_3D_joints_figure(
        elev: int = ELEV, azim: int = AZIM
    ) -> tuple[Figure, Axes3D]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if hasattr(plt.get_current_fig_manager(), "window"):
            plt.get_current_fig_manager().window.wm_geometry("800x600+800-400")

        ax.view_init(elev=elev, azim=azim)
        return fig, ax

    def __plot_joints(self, joints: list[Joint]) -> None:
        self.axis.scatter3D(
            xs=[joint.x for joint in joints],
            ys=[joint.y for joint in joints],
            zs=[joint.y for joint in joints],
        )

    def update_figure(
        self,
        joints: list[Joint],
        angles: list[Angle],
        progress: float,
        repetitions: int,
        state: str,
    ) -> None:
        self.__restart_figure()
        visible_joints = self.__prepare_joints_for_plotting(joints)

        self.__plot_joints(visible_joints)
        self.__plot_connections(visible_joints)
        self.__plot_progress_bar(progress)
        self.__add_text(f"Reps: {repetitions}", x=1.2, y=-1.2, z=0)
        self.__add_text(INSTRUCTION_MAPPER[state], x=0, y=-1.2, z=0)

        for idx, angle in enumerate(angles):
            angle_text = f"{angle.name}: {angle.value:.2f}°"
            x_position = X_LIM[0] - 1
            y_position = idx * 0.2 * Y_LIM[1] - 1
            z_position = Z_LIM[0]
            self.__add_text(angle_text, x_position, y_position, z_position)

        plt.pause(0.001)

    @staticmethod
    def draw_landmarks(
        frame: np.ndarray, results: NormalizedLandmarkList, connections: frozenset
    ) -> None:
        mp_drawing.draw_landmarks(
            frame,
            results,
            connections,
            mp_drawing.DrawingSpec(color=JOINTS_COLOR, thickness=6, circle_radius=6),
            mp_drawing.DrawingSpec(
                color=CONNECTIONS_COLOR, thickness=4, circle_radius=6
            ),
        )

    def __plot_connections(self, joints: list[Joint]) -> None:
        for connection in self.connections:
            joint_start, joint_end = connection
            try:
                start_coords = next(
                    (joint.x, joint.y, joint.z)
                    for joint in joints
                    if joint.id == joint_start
                )
                end_coords = next(
                    (joint.x, joint.y, joint.z)
                    for joint in joints
                    if joint.id == joint_end
                )
            except StopIteration:
                continue

            self.axis.plot(
                xs=[start_coords[0], end_coords[0]],
                ys=[start_coords[1], end_coords[1]],
                zs=[start_coords[2], end_coords[2]],
            )

    def __plot_progress_bar(self, progress: float):
        poly_bar = Poly3DCollection([PROGRESS_BAR_VERTICES], alpha=0.5)
        self.axis.add_collection3d(poly_bar)

        line_vertices = np.array(
            [[1.2, -1 + 2 * progress, 0.2], [2, -1 + 2 * progress, 0.2]]
        )
        line = Line3DCollection([line_vertices], colors="k", linewidths=4)
        self.axis.add_collection3d(line)

    def __add_text(self, text: str, x: float, y: float, z: float, **kwargs) -> None:
        angles_text_kwargs = {
            "s": text,
            "x": x,
            "y": y,
            "z": z,
            "fontsize": 10,
            "style": "italic",
            "bbox": {"facecolor": "white", "alpha": 0.7, "pad": 5},
        }
        self.axis.text3D(**angles_text_kwargs, **kwargs)

    def __restart_figure(self) -> None:
        self.axis.clear()
        self.axis.set_xlim3d(*X_LIM)
        self.axis.set_ylim3d(*Y_LIM)
        self.axis.set_zlim3d(*Z_LIM)

    @staticmethod
    def __prepare_joints_for_plotting(joints: list[Joint]) -> list[Joint]:
        return [joint for joint in joints if joint.visibility > 0.1]
