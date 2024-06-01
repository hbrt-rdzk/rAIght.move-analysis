import pandas as pd

from models.mistake import Mistake
from models.result import Result
from processors.base import Processor

FIX_INFO_KEY = "fix_info"
ANGLE_NAME_KEY = "angle_name"
THRESHOLD_KEY = "threshold"
ERRORS_KEY = "errors"


class MistakesProcessor(Processor):
    def __init__(self, mistakes_table: dict, exercise: str) -> None:
        super().__init__()
        self.mistake_templates = self.__get_mistake_templates(mistakes_table, exercise)

    def process(self, data: list[Result]) -> list[Mistake]:
        return 1

    def update(self, data: list[Mistake]) -> None:
        self.data = data

    @staticmethod
    def to_df(self, data: list[Mistake]) -> pd.DataFrame:
        pass

    @staticmethod
    def from_df(self, df: pd.DataFrame) -> list[Mistake]:
        pass

    def save(self, output_dir: str) -> None:
        pass

    @staticmethod
    def __get_mistake_templates(mistakes_table: dict, exercise: str) -> list[Mistake]:
        mistake_templates = []
        for mistake_name, mistake_data in mistakes_table.items():
            fix_info = mistake_data[FIX_INFO_KEY]
            for error in mistake_data[ERRORS_KEY]:
                mistake_templates.append(
                    Mistake(
                        exercise=exercise,
                        mistake_name=mistake_name,
                        fix_info=fix_info,
                        angle_name=error[ANGLE_NAME_KEY],
                        threshold=error[THRESHOLD_KEY],
                    )
                )
        return mistake_templates
