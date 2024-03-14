import numpy as np

from src.processors.angles.angle import Angle


class RepetitionsCounter:
    def __init__(self, exercise_phases: str) -> None:
        self.start_angles = exercise_phases["start"]
        self.finish_angles = exercise_phases["finish"]

        self.repetitions_count = 0
        self.state = "up"

    def process(self, data: list[Angle]) -> list:
        reference_angle_names = [angle_name for angle_name in self.start_angles.keys()]
        start_reference_angles = np.array(
            [angle_name for angle_name in self.start_angles.values()]
        )
        finish_reference_angles = np.array(
            [angle_name for angle_name in self.finish_angles.values()]
        )

        data_angles = np.array(
            [angle.value for angle in data if angle.name in reference_angle_names]
        )

        reference_progress = start_reference_angles[0] - finish_reference_angles[0]

        progress = np.mean(start_reference_angles - data_angles)
        progress_normalized = progress / reference_progress
        return np.clip(progress_normalized, 0, 1)

    def update(self, progress) -> None:
        if progress >= 1.0 and self.state == "up":
            self.state = "down"
        elif progress <= 0.0 and self.state == "down":
            self.repetitions_count += 1
            self.state = "up"
