import logging
import sys
from abc import ABC, abstractmethod

import mediapipe as mp

from processors.angles.processor import AnglesProcessor
from processors.joints.processor import JointsProcessor
from src.utils.visualizer import Visualizer

POSE_ESTIMATION_MODEL = "pose_landmarker"


class App(ABC):
    def __init__(self, exercise: str):
        super().__init__()
        self.__set_logging()

        self._mp_pose = mp.solutions.pose
        self._pose_estimation_model = self._mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self._joints_processor = JointsProcessor(POSE_ESTIMATION_MODEL)
        self._angles_processor = AnglesProcessor(POSE_ESTIMATION_MODEL)
        self._visualizer = Visualizer(POSE_ESTIMATION_MODEL)

        self._exercise = exercise

    @abstractmethod
    def run(self, input: str, output: str, save_results: bool, loop: bool) -> None:
        ...

    def __set_logging(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)-8s: %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
