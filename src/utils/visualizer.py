import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure
from mediapipe.framework.formats.landmark_pb2 import (
    LandmarkList,
    NormalizedLandmarkList,
)
from mpl_toolkits.mplot3d.axes3d import Axes3D

ELEV = -79
AZIM = -91


class Visualizer:
    """
    A class to visualize 2D and 3D joint data using plotly.

    Attributes:
        image (np.ndarray): Image on which 2D joints are to be plotted.
        joints_3d (np.ndarray, NormalizedLandmarkList, LandmarkList): Array of 3D joint positions.
    """

    def __init__(
        self,
        model: str,
    ) -> None:
        with open("configs/config.yaml") as config:
            self.config = yaml.safe_load(config)[model]

    @staticmethod
    def init_3djoints_figure(
        elev: int = ELEV, azim: int = AZIM
    ) -> tuple[Figure, Axes3D]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
        ax.view_init(elev=elev, azim=azim)
        return fig, ax

    def update_3djoints(self, axis: Axes3D, joints: np.ndarray) -> None:
        important_joints = self._prepare_joints_for_plotting(joints)

        axis.clear()
        axis.set_xlim3d(-1, 1)
        axis.set_ylim3d(-1, 1)
        axis.set_zlim3d(-1, 1)
        axis.scatter3D(
            xs=important_joints[:, 0],
            ys=important_joints[:, 1],
            zs=important_joints[:, 2],
        )
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")

        for connection in self.config["connections"]["torso"]:
            try:
                joint_start, joint_end = connection
                start_coords = important_joints[important_joints[:, 4] == joint_start][
                    0
                ]
                end_coords = important_joints[important_joints[:, 4] == joint_end][0]

                axis.plot(
                    xs=[start_coords[0], end_coords[0]],
                    ys=[start_coords[1], end_coords[1]],
                    zs=[start_coords[2], end_coords[2]],
                )
            except IndexError:
                continue
        plt.pause(0.001)

    def _prepare_joints_for_plotting(self, joints: np.ndarray) -> np.ndarray:
        if isinstance(joints, (NormalizedLandmarkList, LandmarkList)):
            joints = self._load_joints_from_landmark(joints)
        mask = [
            (joint[3] >= 0.5) and (idx in self.config["joints"])
            for idx, joint in enumerate(joints)
        ]
        good_joints = joints[mask]
        return good_joints

    @staticmethod
    def _load_joints_from_landmark(landmark_joints: np.ndarray) -> np.ndarray:
        joints = [
            np.array([joint.x, joint.y, joint.z, joint.visibility, idx])
            for idx, joint in enumerate(landmark_joints.landmark)
        ]
        return np.array(joints)
