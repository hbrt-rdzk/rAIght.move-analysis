import os

import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from src.processors.base import Processor
from src.processors.joints.joint import Joint

OUTPUT_COLUMNS = ("x", "y", "z", "visibility", "joint_id")


class JointsProcessor(Processor):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.__joint_names = self._config_data[model]["joints"]

    def __len__(self) -> int:
        return len(self.data) * len(self.data[0]) * 6

    def process(self, data: NormalizedLandmarkList) -> list[Joint]:
        return [
            Joint(
                idx,
                self.__joint_names[idx],
                joint.x,
                joint.y,
                joint.z,
                joint.visibility,
            )
            for idx, joint in enumerate(data.landmark)
            if idx in self.__joint_names.keys()
        ]

    def update(self, data: list[Joint]) -> None:
        self.data.append(data)

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        joints_per_frame_num = len(self.data[0])

        joints_df = self.__to_df()
        frames_series = pd.Series(
            [i // joints_per_frame_num for i in range(len(joints_df))], name="frame"
        )

        joints_df = pd.concat([frames_series, joints_df], axis=1)

        results_path = os.path.join(output, "joints.csv")
        joints_df.to_csv(results_path, index=False)

    def __to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [joint.__dict__ for frame_joints in self.data for joint in frame_joints]
        )
