from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import yaml

CONFIG_PATH = "configs/config.yaml"
REFERENCE_TABLE_PATH = "configs/reference_tables.yaml"


class DataHandler(ABC):
    def __init__(self) -> None:
        self._config_data = self.__load_yaml_file(CONFIG_PATH)
        self._reference_table = self.__load_yaml_file(REFERENCE_TABLE_PATH)
        self._results_path = self._config_data["output_path"]
        self._frames_counter = 0

    @abstractmethod
    def load_data(self, data) -> Any:
        ...

    @abstractmethod
    def update(self, data) -> None:
        ...

    @abstractmethod
    def save(self, output_path: str) -> None:
        ...

    def __load_yaml_file(self, file_path):
        try:
            with open(file_path) as file:
                return yaml.safe_load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"File not found: {error.filename}") from None
        except yaml.YAMLError as error:
            raise ValueError(f"Error parsing YAML file: {error}") from None
