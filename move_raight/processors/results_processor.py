import os

import pandas as pd
from models.angle import ANGLES_PER_FRAME
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
                    results.append(Result(reference_idx, data.rep, angle_name, diff))

        return results

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
        results_df = self.to_df(self.data)
        results_segments = results_df.groupby("rep")

        for rep, results in results_segments:
            results_path = os.path.join(output, f"rep_{rep}")
            os.makedirs(results_path, exist_ok=True)

            results.to_csv(os.path.join(results_path, "angles_diffs.csv"), index=False)
