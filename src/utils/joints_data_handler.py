import os

import numpy as np
import pandas as pd
import yaml

OUTPUT_COLUMNS = ("frame", "x", "y", "z", "visibility", "joint_id")


class JointsDataHandler:
    def __init__(self, model: str) -> None:
        with open("configs/config.yaml") as config:
            data = yaml.safe_load(config)
            self.joint_names = data[model]["joints"]
        self.results_path = data["joints_output_path"]
        self.joints = []

    def add_joints(self, joints: np.ndarray, frame_index: int = 0) -> None:
        frame_column = np.full((joints.shape[0], 1), frame_index)
        self.joints.append(np.hstack((frame_column, joints)))

    def save_joints(self, output_filename: str) -> None:
        if not self.joints:
            raise ValueError("0 joints found!")
        if not output_filename:
            raise ValueError("Please provide proper output path (--output)!")
        os.makedirs(self.results_path, exist_ok=True)
        output_path = os.path.join(self.results_path, output_filename)

        joints_ = np.concatenate(self.joints)
        joints_ = joints_[np.isin(joints_[:, -1], list(self.joint_names.keys()))]

        joints_df = pd.DataFrame(joints_, columns=OUTPUT_COLUMNS)
        joints_df["frame"] = joints_df["frame"].astype(int)
        joints_df["joint_id"] = joints_df["joint_id"].astype(int)

        joints_df.to_csv(output_path, index=False)

    def calculate_angle_between_joints(
        self, joint_1_id: int, joint_2_id: int, joint_3_id: int, frame: int = 0
    ) -> float:
        desired_frame_joints = self.joints[frame]
        return self._calculate_3d_angle(
            desired_frame_joints[desired_frame_joints[:, 6] == joint_1_id][0, 1:4],
            desired_frame_joints[desired_frame_joints[:, 6] == joint_2_id][0, 1:4],
            desired_frame_joints[desired_frame_joints[:, 6] == joint_3_id][0, 1:4],
        )

    @staticmethod
    def load_joints_from_landmark(landmark_joints: np.ndarray) -> np.ndarray:
        joints = [
            np.array([joint.x, joint.y, joint.z, joint.visibility, idx])
            for idx, joint in enumerate(landmark_joints.landmark)
        ]
        return np.array(joints)

    @staticmethod
    def _calculate_3d_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        assert A.shape == B.shape == C.shape == (3,)
        assert A.dtype == B.dtype == C.dtype == np.float64

        AB = np.array(B) - np.array(A)
        BC = np.array(C) - np.array(B)

        dot_product = np.dot(AB, BC)

        norm_AB = np.linalg.norm(AB)
        norm_BC = np.linalg.norm(BC)

        angle_rad = np.arccos(dot_product / (norm_AB * norm_BC))
        angle_deg = np.degrees(angle_rad)

        return angle_deg
