import os

import numpy as np
import pandas as pd

from src.processors.abstract_handler import DataHandler


class AnglesHandler(DataHandler):
    def __init__(self, model: str, exercise: str) -> None:
        super().__init__()
        self.__angle_names = self._config_data[model]["angles"]
        self.__reference_table = self._reference_table[exercise]

        """Angles as list of dicts"""
        self.angles = []

    def load_data(self, data) -> dict[str, float]:
        return {
            angle_name: self.calculate_3D_angle(
                data[data[:, 4] == joint_ids[0]][0, 1:4],
                data[data[:, 4] == joint_ids[1]][0, 1:4],
                data[data[:, 4] == joint_ids[2]][0, 1:4],
            )
            for angle_name, joint_ids in self.__angle_names.items()
        }

    def update(self, data) -> None:
        self.angles.append(data)
        self._frames_counter += 1

    def save(self, output: str) -> None:
        if not self.angles:
            raise ValueError("No angles data to save.")

        os.makedirs(self._results_path, exist_ok=True)
        output_path = os.path.join(self._results_path, "angles_" + output)
        angles_df = pd.DataFrame.from_dict(self.angles)
        angles_df.to_csv(output_path, index=True, index_label="frame")

    def get_exercise_phase(self) -> str:
        phases = self.__reference_table["phases"]
        for phase, positions in phases.items():
            if any(
                [
                    scope[0] > self.angles[-1][angle_name] > scope[1]
                    for angle_name, scope, in positions.items()
                ]
            ):
                return phase

    @staticmethod
    def calculate_3D_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        if not (A.shape == B.shape == C.shape == (3,)):
            raise ValueError("Input arrays must all be of shape (3,).")
        if not (A.dtype == B.dtype == C.dtype == np.float64):
            raise TypeError("Input arrays must all have dtype np.float64.")

        ba = A - B
        bc = C - B
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
