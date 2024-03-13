import logging
import sys
from abc import ABC, abstractmethod

import mediapipe as mp

POSE_ESTIMATION_MODEL_NAME = "pose_landmarker"


class App(ABC):
    def __init__(self, exercise: str):
        super().__init__()
        self.__set_logging()

        self.exercise = exercise

        self.mp_pose = mp.solutions.pose
        self._pose_estimation_model = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self._pose_estimation_model_name = POSE_ESTIMATION_MODEL_NAME

    @abstractmethod
    def run(self, input: str, output: str, save_results: bool, loop: bool) -> None:
        """
        Run app's flow
        """

    def __set_logging(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)-8s: %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
