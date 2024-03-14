import os

import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from src.processors.base import Processor
from src.processors.joints.joint import Joint

OUTPUT_COLUMNS = ("x", "y", "z", "visibility", "joint_id")


class JointsProcessor(Processor):
    def __init__(self, joint_names: dict) -> None:
        super().__init__()
        self.joint_names = joint_names
        self.__current_frame = 0

    def __len__(self) -> int:
        return len(self.data) * len(self.data[0]) * 6

    def process(self, data: NormalizedLandmarkList) -> list[Joint]:
        return [
            Joint(
                idx,
                self.joint_names[idx],
                joint.x,
                joint.y,
                joint.z,
                joint.visibility,
                self.__current_frame,
            )
            for idx, joint in enumerate(data.landmark)
            if idx in self.joint_names.keys()
        ]

    def update(self, data: list[Joint]) -> None:
        self.data.append(data)
        self.__current_frame += 1

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        joints_df = self.to_df(self.data)

        results_path = os.path.join(output, "joints.csv")
        joints_df.to_csv(results_path, index=False)

    @staticmethod
    def to_df(data: list[Joint]) -> pd.DataFrame:
        return pd.DataFrame(
            [joint.__dict__ for frame_joints in data for joint in frame_joints]
        )

    @staticmethod
    def from_df(data: pd.DataFrame) -> list[Joint]:
        ...
