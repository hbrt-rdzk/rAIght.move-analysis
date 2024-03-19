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
    def from_df(data: pd.DataFrame) -> list[Result]:
        results = []
        for _, result in data.iterrows():
            results.append(
                Result(result["frame"], result["angle_name"], result["diff"])
            )
        return results

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)
        results_df = self.to_df(self.data)
        results_segments = results_df.groupby("rep")

        for rep, results in results_segments:
            results_path = os.path.join(output, f"rep_{rep}")
            os.makedirs(results_path, exist_ok=True)

            results.to_csv(os.path.join(results_path, "results.csv"), index=False)
