import os

import pandas as pd

from models.mistake import Mistake
from models.result import Result
from models.segment import Segment
from processors.angles_processor import ANGLE_TYPES
from processors.base import Processor
from utils.dtw import (filter_repetable_reference_indexes,
                       get_warped_frame_indexes)


class ResultsProcessor(Processor):
    """
    Processor of differences between query and reference values
    """

    def __init__(
        self, reference_segement: Segment, compariston_features: list[str]
    ) -> None:
        super().__init__()
        self.reference_segment = reference_segement
        self.comparison_features = compariston_features

    def process(self, data: Segment) -> Result:
        query = data
        results = []
        for feature in self.comparison_features:
            for angle_type in ANGLE_TYPES.keys():
                angle_name = feature + "_" + angle_type
                query = [
                    angle.value for angle in data.angles if angle.name == angle_name
                ]
                reference = [
                    angle.value
                    for angle in self.reference_segment.angles
                    if angle.name == angle_name
                ]
                path = get_warped_frame_indexes(query, reference)
                query_to_reference_warping = filter_repetable_reference_indexes(
                    path[:, 1], path[:, 0]
                )
                for reference_idx, query_idx in enumerate(query_to_reference_warping):
                    diff = reference[reference_idx] - query[query_idx]
                    results.append(Result(reference_idx + 1, angle_name, diff))

        return results

    def update(self, data: list[Result]) -> None:
        self.data.append(data)

    @staticmethod
    def to_df(data: list[Result]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df_reshaped = df.pivot(index="frame", columns="angle_name", values="diff")
        return df_reshaped

    @staticmethod
    def from_df(df: pd.DataFrame) -> list[Result]:
        results = []
        for _, result in df.iterrows():
            results.append(
                Result(
                    result["frame"],
                    results["rep"],
                    result["angle_name"],
                    result["diff"],
                )
            )
        return results

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)

        for rep, segment_results in enumerate(self.data):
            results_path = os.path.join(output, f"rep_{rep + 1}")
            os.makedirs(results_path, exist_ok=True)
            results_df = self.to_df(segment_results)
            results_df.to_csv(os.path.join(results_path, "angles_diffs.csv"))
