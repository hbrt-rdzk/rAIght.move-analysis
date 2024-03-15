import os
from typing import Any

import pandas as pd
from processors.base import Processor
from processors.joints.joint import JOINT_PARAMETERS_NUM, Joint

OUTPUT_COLUMNS = ("x", "y", "z", "visibility", "joint_id")


class JointsProcessor(Processor):
    def __init__(self, joint_names: dict) -> None:
        super().__init__()
        self.joint_names = joint_names
        self.__current_frame = 1

    def __len__(self) -> int:
        return len(self.data) * JOINT_PARAMETERS_NUM

    def process(self, data: Any) -> list[Joint]:
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
        self.data.extend(data)
        self.__current_frame += 1

    @staticmethod
    def to_df(data: list[list[Joint]]) -> pd.DataFrame:
        return pd.DataFrame(data)

    @staticmethod
    def from_df(data: pd.DataFrame) -> list[list[Joint]]:
        joints = []
        for _, joint in data.iterrows():
            joints.append(
                Joint(
                    joint["id"],
                    joint["name"],
                    joint["x"],
                    joint["y"],
                    joint["z"],
                    joint["visibility"],
                    joint["frame"],
                )
            )
        return joints

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        joints_df = self.to_df(self.data)

        results_path = os.path.join(output, "joints.csv")
        joints_df.to_csv(results_path, index=False)
