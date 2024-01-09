import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

ELEV = -79
AZIM = -91
X_LIM = (-1, 1)
Y_LIM = (0, 1)
Z_LIM = (-1, 1)
CONFIG_PATH = "configs/config.yaml"


class Visualizer:
    def __init__(self, model: str) -> None:
        self.figure, self.axes = self.__initialize_3D_joints_figure()
        self.__exercise_phase = ""
        try:
            with open(CONFIG_PATH) as config:
                self.config = yaml.safe_load(config)[model]
        except FileNotFoundError:
            raise FileNotFoundError("Configuration file not found")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")

    @property
    def phase(self) -> str:
        return self.__exercise_phase

    @phase.setter
    def phase(self, phase: str) -> None:
        if phase:
            self.__exercise_phase = phase

    def update_figure(self, joints: np.ndarray, angles: dict) -> None:
        self.__restart_figure()
        important_joints = self.__prepare_joints_for_plotting(joints)

        self.__plot_joints(important_joints)
        self.__plot_connections(important_joints)
        for index, (angle_name, angle) in enumerate(angles.items()):
            angle_text = f"{angle_name}: {angle:.2f}°"
            self.__add_text(
                angle_text,
                (X_LIM[0] - 1),
                (index * 0.1 * Y_LIM[1]),
                Z_LIM[0],
            )

        self.__add_text(
            self.__exercise_phase,
            (X_LIM[0] + X_LIM[1]),
            Y_LIM[0],
            Z_LIM[0],
            color="blue",
        )

        plt.pause(0.001)

    @staticmethod
    def __initialize_3D_joints_figure(
        elev: int = ELEV, azim: int = AZIM
    ) -> tuple[Figure, Axes3D]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

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
