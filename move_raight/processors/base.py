import datetime
import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Processor(ABC):
    """
    Classes that inherits from Processor are responsible for storing and processing data
    """

    def __init__(self) -> None:
        self.data = []
        self.current_time = datetime.datetime.now()

    @abstractmethod
    def process(self, data: list[Any]) -> list[Any]:
        """
        Process data flow
        """

    @abstractmethod
    def update(self, data: list[Any]) -> None:
        """
        Update internal state of the object
        """

    @staticmethod
    @abstractmethod
    def to_df(data: Any) -> pd.DataFrame:
        """
        Convert DataFrame object from data
        """

    @staticmethod
    @abstractmethod
    def from_df(data: pd.DataFrame) -> Any:
        """
        Convert data from DataFrame object
        """

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """
        Save state to static file
        """

    def _validate_output(self, output_dir: str) -> str:
        output_subdir = datetime.datetime.strftime(
            self.current_time, "%Y-%m-%d_%H:%M:%S"
        )

        if not self.data:
            raise ValueError("No data to save.")

        output_dir = os.path.join(output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        return output_dir
