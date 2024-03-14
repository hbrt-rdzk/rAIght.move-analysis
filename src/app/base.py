import logging
import sys
from abc import ABC, abstractmethod

import mediapipe as mp
import yaml

CONFIG_PATH = "configs/config.yaml"
REFERENCE_TABLE_PATH = "configs/reference_tables.yaml"
POSE_ESTIMATION_MODEL_NAME = "pose_landmarker"
OUTPUT_PATH_FIELD = "output_path"


class App(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.__set_logging()

        self._mp_pose = mp.solutions.pose
        self._pose_estimation_model = self._mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self._config_data = self.__load_yaml_file(CONFIG_PATH)
        self._reference_table = self.__load_yaml_file(REFERENCE_TABLE_PATH)

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

    def __load_yaml_file(self, file_path: str) -> dict:
        try:
            with open(file_path) as file:
                return yaml.safe_load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"File not found: {error.filename}") from None
        except yaml.YAMLError as error:
            raise ValueError(f"Error parsing YAML file: {error}") from None
