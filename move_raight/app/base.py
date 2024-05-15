import logging
from abc import ABC, abstractmethod

import mediapipe as mp
import yaml

CONFIG_PATH = "configs/config.yaml"
PHASES_TABLE = "configs/exercises_table.yaml"
POSE_ESTIMATION_MODEL_NAME = "mediapipe"
OUTPUT_PATH_FIELD = "output_path"


class App(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self._mp_pose = mp.solutions.pose
        self._pose_estimation_model = self._mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
        )

        self._config_data = self.__load_yaml_file(CONFIG_PATH)
        self._exercise_table = self.__load_yaml_file(PHASES_TABLE)

    @abstractmethod
    def run(self, input: str, output: str, save_results: bool) -> None:
        """
        Run app's flow
        """

    def __load_yaml_file(self, file_path: str) -> dict:
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"File not found: {error.filename}") from None
        except yaml.YAMLError as error:
            raise ValueError(f"Error parsing YAML file: {error}") from None
