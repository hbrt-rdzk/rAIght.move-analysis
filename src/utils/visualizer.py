import matplotlib.pyplot as plt
import numpy as np
import yaml
from mediapipe.framework.formats.landmark_pb2 import (
    LandmarkList,
    NormalizedLandmarkList,
)


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

    def update_3djoints(self, axis, joints):
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
            except:
                continue
        plt.pause(.001)

    def _prepare_joints_for_plotting(self, joints) -> tuple:
        if isinstance(joints, (NormalizedLandmarkList, LandmarkList)):
            joints = self._load_joints_from_landmark(joints)
        mask = [
            (joint[3] >= 0.5) and (idx in self.config["joints"])
            for idx, joint in enumerate(joints)
        ]
        good_joints = joints[mask]
        return good_joints

    @staticmethod
    def _load_joints_from_landmark(landmark_joints) -> list:
        joints = [
            np.array([joint.x, joint.y, joint.z, joint.visibility, idx])
            for idx, joint in enumerate(landmark_joints.landmark)
        ]
        return np.array(joints)
