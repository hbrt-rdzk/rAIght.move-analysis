import matplotlib

matplotlib.use("TkAgg")

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
        try:
            with open(CONFIG_PATH) as config:
                self.config = yaml.safe_load(config)[model]
        except FileNotFoundError:
            raise FileNotFoundError("Configuration file not found")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")

    @staticmethod
    def init_3djoints_figure(
        elev: int = ELEV, azim: int = AZIM
    ) -> tuple[Figure, Axes3D]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

        if hasattr(plt.get_current_fig_manager(), "window"):
            plt.get_current_fig_manager().window.wm_geometry("800x600+800-400")

        ax.view_init(elev=elev, azim=azim)
        return fig, ax

    def update_figure(
        self, axis: Axes3D, joints: np.ndarray, angles: dict[str, float]
    ) -> None:
        important_joints = self.__prepare_joints_for_plotting(joints)
        axis.clear()
        axis.set_xlim3d(*X_LIM)
        axis.set_ylim3d(*Y_LIM)
        axis.set_zlim3d(*Z_LIM)
        axis.scatter3D(
            xs=important_joints[:, 0],
            ys=important_joints[:, 1],
            zs=important_joints[:, 2],
        )
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")

        for connection in self.config["connections"]["torso"]:
            joint_start, joint_end = connection
            start_coords = important_joints[important_joints[:, 4] == joint_start]
            end_coords = important_joints[important_joints[:, 4] == joint_end]
            if len(start_coords) > 0 and len(end_coords) > 0:
                axis.plot(
                    xs=[start_coords[0, 0], end_coords[0, 0]],
                    ys=[start_coords[0, 1], end_coords[0, 1]],
                    zs=[start_coords[0, 2], end_coords[0, 2]],
                )

        text = ""
        for joint_name, angle in angles.items():
            text += f"{joint_name}: {angle:.2f}°\n"
        text_kwargs = {
            "s": text,
            "x": X_LIM[0] - 1,
            "y": Y_LIM[1] - 0.5,
            "z": Z_LIM[0],
            "fontsize": 10,
            "color": "blue",
            "style": "italic",
            "bbox": {"facecolor": "white", "alpha": 0.7, "pad": 5},
        }
        axis.text3D(**text_kwargs)
        plt.pause(0.001)

    def __prepare_joints_for_plotting(self, joints: np.ndarray) -> np.ndarray:
        mask = [joint[3] >= 0.2 for joint in joints]
        return joints[mask]
