import os

import numpy as np
import pandas as pd

from src.processors.angles.angle import Angle
from src.processors.base import Processor
from src.processors.joints.joint import Joint


class AnglesProcessor(Processor):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.__angle_names = self._config_data[model]["angles"]

    def process(self, data: list[Joint]) -> list[Angle]:
        joint_dict = {joint.id: [joint.x, joint.y, joint.z] for joint in data}

        angles = []
        for angle_name, joint_ids in self.__angle_names.items():
            coords = np.array([joint_dict[joint_id] for joint_id in joint_ids])
            angle = self.calculate_3D_angle(*coords)
            angles.append(Angle(angle_name, angle))

        return angles

    def update(self, data: list[Angle]) -> None:
        self.data.append(data)

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)
        angles = [
            {angle.name: angle.value for angle in frame_angles}
            for frame_angles in self.data
        ]

        angles_df = pd.DataFrame.from_dict(angles)

        results_path = os.path.join(output, "angles.csv")
        angles_df.to_csv(results_path, index=True, index_label="frame")

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
