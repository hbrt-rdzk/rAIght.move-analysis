import os

import numpy as np
import pandas as pd

from src.processors.abstract_handler import DataHandler

OUTPUT_COLUMNS = ("x", "y", "z", "visibility", "joint_id")


class JointsHandler(DataHandler):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.__joint_names = self._config_data[model]["joints"]

        """Joints as list of numpy arrays"""
        self.joints = []

    def load_data(self, data) -> np.ndarray:
        joint_ids = list(map(int, self.__joint_names.keys()))
        joints = np.array(
            [
                np.array([joint.x, joint.y, joint.z, joint.visibility, idx])
                for idx, joint in enumerate(data.landmark)
            ]
        )
        return joints[np.isin(joints[:, -1].astype(int), joint_ids)]

    def update(self, data: np.ndarray) -> None:
        self.joints.append(data)
        self._frames_counter += 1

    def save(self, output: str) -> None:
        if not self.joints:
            raise ValueError("No joints data to save.")

        os.makedirs(self._results_path, exist_ok=True)
        output_path = os.path.join(self._results_path, "joints_" + output)
        frames_num = self.joints[0].shape[0]
        joints_ = np.concatenate(self.joints)

        joints_df = pd.DataFrame(joints_, columns=OUTPUT_COLUMNS)
        frames_series = pd.Series(
            [i // frames_num for i in range(len(joints_df))], name="frame"
        )

        joints_df = pd.concat([frames_series, joints_df], axis=1)
        joints_df["joint_id"] = joints_df["joint_id"].astype(int)

        joints_df.to_csv(output_path, index=False)
