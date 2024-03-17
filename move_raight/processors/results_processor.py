import os

import pandas as pd
from models.result import Result
from processors.base import Processor


class ResultsProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()

    def process(self, data: list[list]) -> list[Result]:
        return [Result(*frame_data) for frame_data in data]

    def update(self, data: list[Result]) -> None:
        self.data.extend(data)

    @staticmethod
    def to_df(data: list[Result]) -> pd.DataFrame:
        return pd.DataFrame(data)

    @staticmethod
    def from_df(data: list[Result]) -> pd.DataFrame:
        results = []
        for _, result in data.iterrows():
            results.append(
                Result(result["frame"], result["angle_name"], result["diff"])
            )
        return results

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)
        angles_df = self.to_df(self.data)

        results_path = os.path.join(output, "resutls.csv")
        angles_df.to_csv(results_path, index=False)
