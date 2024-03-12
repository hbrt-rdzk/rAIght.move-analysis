import os

import numpy as np
import pandas as pd

from processors.base import Processor


class AnglesProcessor(Processor):
    """Angles as list of dicts"""

    def __init__(self, model: str) -> None:
        super().__init__()
        self.__angle_names = self._config_data[model]["angles"]

    def __str__(self) -> str:
        frames_num = len(self.data)
        features_per_frame = len(self.data[0])
        return f"{frames_num * features_per_frame} angle features"

    def load_data(self, data: np.ndarray) -> dict[str, float]:
        return {
            angle_name: self.calculate_3D_angle(
                data[data[:, 4] == joint_ids[0]][0, 0:3],
                data[data[:, 4] == joint_ids[1]][0, 0:3],
                data[data[:, 4] == joint_ids[2]][0, 0:3],
            )
            for angle_name, joint_ids in self.__angle_names.items()
        }

    def update(self, data: dict[str, float]) -> None:
        self.data.append(data)

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        angles_df = pd.DataFrame.from_dict(self.data)

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