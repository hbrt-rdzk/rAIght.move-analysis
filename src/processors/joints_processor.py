import os
from typing import Any

import pandas as pd

from models.joint import JOINT_PARAMETERS_NUM, Joint
from processors.base import Processor


class JointsProcessor(Processor):
    """
    Procssor of the joints in 3D space
    """

    current_processing_frame = 1

    def __init__(self, joint_names: dict) -> None:
        super().__init__()
        self.joint_names = joint_names

    def __len__(self) -> int:
        return len(self.data) * JOINT_PARAMETERS_NUM

    def process(self, data: Any) -> list[Joint]:
        return [
            Joint(
                frame=self.current_processing_frame,
                id=idx,
                name=self.joint_names[idx],
                x=joint.x,
                y=joint.y,
                z=joint.z,
                visibility=joint.visibility,
            )
            for idx, joint in enumerate(data.landmark)
            if idx in self.joint_names.keys()
        ]

    def update(self, data: list[Joint]) -> None:
        self.data.extend(data)

    @staticmethod
    def to_df(data: list[list[Joint]]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df = df.set_index("frame")
        return df

    @staticmethod
    def from_df(df: pd.DataFrame) -> list[list[Joint]]:
        joints = []
        for _, joint in df.iterrows():
            joints.append(
                Joint(
                    joint["frame"],
                    joint["id"],
                    joint["name"],
                    joint["x"],
                    joint["y"],
                    joint["z"],
                    joint["visibility"],
                )
            )
        return joints

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        joints_df = self.to_df(self.data)

        results_path = os.path.join(output, "joints.csv")
        joints_df.to_csv(results_path, index=False)
