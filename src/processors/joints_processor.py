import os

import numpy as np
import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from src.processors.abstract_processor import DataProcessor

OUTPUT_COLUMNS = ("x", "y", "z", "visibility", "joint_id")


class JointsProcessor(DataProcessor):
    """Joints as list of numpy arrays"""

    def __init__(self, model: str) -> None:
        super().__init__()
        self.__joint_names = self._config_data[model]["joints"]

    def __str__(self) -> str:
        frames_num = len(self.data)
        features_per_frame = len(self.data[0])
        return f"{frames_num * features_per_frame * 3} joint features"

    def load_data(self, data: NormalizedLandmarkList) -> np.ndarray:
        return np.array(
            [
                np.array([joint.x, joint.y, joint.z, joint.visibility, idx])
                for idx, joint in enumerate(data.landmark)
            ]
        )

    def update(self, data: np.ndarray) -> None:
        self.data.append(data)

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        frames_num = self.data[0].shape[0]
        joints_ = np.concatenate(self.data)

        joints_df = pd.DataFrame(joints_, columns=OUTPUT_COLUMNS)
        frames_series = pd.Series(
            [i // frames_num for i in range(len(joints_df))], name="frame"
        )

        joints_df = pd.concat([frames_series, joints_df], axis=1)
        joints_df["joint_id"] = joints_df["joint_id"].astype(int)

        results_path = os.path.join(output, "joints.csv")
        joints_df.to_csv(results_path, index=False)

    def filter(self, joints: np.ndarray) -> np.ndarray:
        joint_ids = list(map(int, self.__joint_names.keys()))
        return joints[np.isin(joints[:, -1].astype(int), joint_ids)]
