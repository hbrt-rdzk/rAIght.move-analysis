import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

COLUMNS = ("frame", "x", "y", "z", "visibility", "joint_name")


class JointsDataHandler:
    def __init__(self, model: str) -> None:
        with open("configs/config.yaml") as config:
            data = yaml.safe_load(config)
            self.joints_output_path = data["joints_output_path"]
            self.joint_names = data[model]["joints"]

        self.joints = []

    def add_joints(self, joints: np.ndarray, frame_index: int) -> None:
        frame_column = np.full((joints.shape[0], 1), frame_index)
        self.joints.append(np.hstack((frame_column, joints)))

    def save_joints(self) -> None:
        if not os.path.exists(self.joints_output_path):
            os.makedirs(self.joints_output_path)

        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        output_path = os.path.join(self.joints_output_path, current_time)

        joints_ = np.concatenate(self.joints)
        joints_ = joints_[np.isin(joints_[:, -1], list(self.joint_names.keys()))]

        joints_df = pd.DataFrame(joints_, columns=COLUMNS)
        joints_df["joint_name"] = joints_df["joint_name"].apply(
            lambda x: self.joint_names[x]
        )
        joints_df["frame"] = joints_df["frame"].astype(int)

        joints_df.to_csv(output_path, index=False)

    @staticmethod
    def load_joints_from_landmark(landmark_joints: np.ndarray) -> np.ndarray:
        joints = [
            np.array([joint.x, joint.y, joint.z, joint.visibility, idx])
            for idx, joint in enumerate(landmark_joints.landmark)
        ]
        return np.array(joints)
