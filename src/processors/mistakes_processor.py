import os

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
        # TODO: improve this
        mistakes = [
            mistake_template
            for mistake_template in self.mistake_templates
            for result in data
            if result.angle_name == mistake_template.angle_name
            and abs(result.diff) > mistake_template.threshold
        ]
        return list(set(mistakes))

    def update(self, data: list[Mistake]) -> None:
        self.data.append(data)

    @staticmethod
    def to_df(data: list[Mistake]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def from_df(df: pd.DataFrame) -> list[Mistake]:
        mistakes = []
        for _, mistake in df.iterrows():
            mistakes.append(
                Mistake(
                    mistake["exercise"],
                    mistake["mistake_name"],
                    mistake["fix_info"],
                    mistake["angle_name"],
                    mistake["threshold"],
                )
            )
        return mistakes

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        for rep, segment_mistakes in enumerate(self.data):
            results_path = os.path.join(output, f"rep_{rep + 1}")
            os.makedirs(results_path, exist_ok=True)
            results_df = self.to_df(segment_mistakes)
            results_df.to_csv(os.path.join(results_path, "mistakes.csv"), index=False)

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
