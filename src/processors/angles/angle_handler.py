import os

import numpy as np
import pandas as pd

from src.processors.abstract_handler import DataHandler


class AnglesHandler(DataHandler):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.__angle_names = self._config_data[model]["angles"]

        """Angles as list of dicts"""
        self.angles = []

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
        self.angles.append(data)
        self._frames_counter += 1

    def save(self, output: str) -> None:
        if not self.angles:
            raise ValueError("No angles data to save.")

        os.makedirs(self._results_path, exist_ok=True)
        output_path = os.path.join(self._results_path, "angles_" + output)
        angles_df = pd.DataFrame.from_dict(self.angles)
        angles_df.to_csv(output_path, index=True, index_label="frame")

    @staticmethod
    def calculate_3D_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        if not (A.shape == B.shape == C.shape == (3,)):
            raise ValueError("Input arrays must all be of shape (3,).")

        # Obliczanie wektorów
        ba = A - B
        bc = C - B

        # Obliczanie kąta
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Ograniczenie wartości do zakresu [-1, 1], aby uniknąć błędów numerycznych
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
