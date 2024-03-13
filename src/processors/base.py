import datetime
import os
from abc import ABC, abstractmethod
from typing import Any

import yaml

CONFIG_PATH = "configs/config.yaml"
REFERENCE_TABLE_PATH = "configs/reference_tables.yaml"


class Processor(ABC):
    def __init__(self) -> None:
        self._config_data = self.__load_yaml_file(CONFIG_PATH)
        self._reference_table = self.__load_yaml_file(REFERENCE_TABLE_PATH)
        self._results_path = self._config_data["output_path"]

        self.data = []

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process data flow
        """

    @abstractmethod
    def update(self, data: Any) -> None:
        """
        Update internals state of the object
        """

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """
        Save state to static file
        """

    def __load_yaml_file(self, file_path: str) -> dict:
        try:
            with open(file_path) as file:
                return yaml.safe_load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"File not found: {error.filename}") from None
        except yaml.YAMLError as error:
            raise ValueError(f"Error parsing YAML file: {error}") from None

    def _validate_output(self, output_dir: str) -> str:
        current_time = datetime.datetime.now()
        output_subdir = datetime.datetime.strftime(current_time, "%Y-%m-%d_%H:%M:%S")

        if not self.data:
            raise ValueError("No data to save.")

        output_dir = os.path.join(output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        return output_dir
