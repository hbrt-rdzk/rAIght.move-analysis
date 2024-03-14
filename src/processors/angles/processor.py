import os

import numpy as np
import pandas as pd

from src.processors.angles.angle import Angle
from src.processors.base import Processor
from src.processors.joints.joint import Joint


class AnglesProcessor(Processor):
    def __init__(self, angle_names: dict) -> None:
        super().__init__()
        self.angle_names = angle_names
        self.__current_frame = 0

    def __len__(self) -> int:
        return len(self.data) * len(self.data[0]) * 2

    def process(self, data: list[Joint]) -> list[Angle]:
        joint_dict = {joint.id: [joint.x, joint.y, joint.z] for joint in data}

        angles = []
        for angle_name, joint_ids in self.angle_names.items():
            coords = np.array([joint_dict[joint_id] for joint_id in joint_ids])
            angle = self.calculate_3D_angle(*coords)
            angles.append(Angle(angle_name, angle, self.__current_frame))

        return angles

    def update(self, data: list[Angle]) -> None:
        self.data.append(data)
        self.__current_frame += 1

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)
        angles_df = self.to_df(self.data)

        results_path = os.path.join(output, "angles.csv")
        angles_df.to_csv(results_path, index_label="frame")

    @staticmethod
    def to_df(data: list[list[Angle]]) -> pd.DataFrame:
        frames = [frame_angles[0].frame for frame_angles in data]
        return pd.DataFrame(
            [
                {angle.name: angle.value for angle in frame_angles}
                for frame_angles in data
            ],
            index=frames,
        )

    @staticmethod
    def from_df(data: pd.DataFrame) -> list[Angle]:
        ...

    @staticmethod
    def calculate_3D_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        if not (A.shape == B.shape == C.shape == (3,)):
            raise ValueError("Input arrays must all be of shape (3,).")

        ba = A - B
        bc = C - B

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
