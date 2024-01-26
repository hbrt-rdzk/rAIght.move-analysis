import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import yaml
from matplotlib.figure import Figure
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

ELEV = -90
AZIM = -90
X_LIM = (-1, 1)
Y_LIM = (-1, 1)
Z_LIM = (-1, 1)
JOINTS_COLOR = (2, 82, 212)
CONNECTIONS_COLOR = (245, 255, 230)
CONFIG_PATH = "configs/config.yaml"

mp_drawing = mp.solutions.drawing_utils


class Visualizer:
    def __init__(self, model: str) -> None:
        self.figure, self.axes = self.__initialize_3D_joints_figure()
        try:
            with open(CONFIG_PATH) as config:
                self.config = yaml.safe_load(config)[model]
        except FileNotFoundError:
            raise FileNotFoundError("Configuration file not found")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")

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

    def update_figure(
        self, joints: np.ndarray, angles: dict, repetitions: int, progress: float
    ) -> None:
        self.__restart_figure()
        important_joints = self.__prepare_joints_for_plotting(joints)

        self.__plot_joints(important_joints)
        self.__plot_connections(important_joints)
        self.__plot_progress_bar(progress)
        self.__add_text(f"Reps: {repetitions}", x=1.2, y=-1.2, z=0)
        for index, (angle_name, angle) in enumerate(angles.items()):
            angle_text = f"{angle_name}: {angle:.2f}°"
            x_position = X_LIM[0] - 1
            y_position = index * 0.2 * Y_LIM[1] - 1
            z_position = Z_LIM[0]
            self.__add_text(angle_text, x_position, y_position, z_position)

        plt.pause(0.001)

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

    def __plot_joints(self, joints: np.ndarray) -> None:
        self.axes.scatter3D(
            xs=joints[:, 0],
            ys=joints[:, 1],
            zs=joints[:, 2],
        )

    def __plot_connections(self, joints: np.ndarray) -> None:
        for connection in self.config["connections"]["torso"]:
            joint_start, joint_end = connection
            start_coords = joints[joints[:, 4] == joint_start]
            end_coords = joints[joints[:, 4] == joint_end]
            if len(start_coords) > 0 and len(end_coords) > 0:
                self.axes.plot(
                    xs=[start_coords[0, 0], end_coords[0, 0]],
                    ys=[start_coords[0, 1], end_coords[0, 1]],
                    zs=[start_coords[0, 2], end_coords[0, 2]],
                )

    def __plot_progress_bar(self, progress: float):
        bottom_limit = [1.2, 1, 0.2], [2, 1, 0.2]
        upper_limit = [1.2, -1 + 2 * progress, 0.2], [2, -1 + 2 * progress, 0.2]

        vertices = [bottom_limit[0], upper_limit[0], upper_limit[1], bottom_limit[1]]
        poly = Poly3DCollection([vertices], alpha=0.5)
        self.axes.add_collection3d(poly)

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
        self.axes.text3D(**angles_text_kwargs, **kwargs)

    def __restart_figure(self) -> None:
        self.axes.clear()
        self.axes.set_xlim3d(*X_LIM)
        self.axes.set_ylim3d(*Y_LIM)
        self.axes.set_zlim3d(*Z_LIM)

        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_zlabel("Z")

    @staticmethod
    def __prepare_joints_for_plotting(joints: np.ndarray) -> np.ndarray:
        mask = [joint[3] >= 0.1 for joint in joints]
        return joints[mask]
