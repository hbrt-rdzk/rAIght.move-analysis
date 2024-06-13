from collections import defaultdict
import os

import numpy as np
import pandas as pd

from models.angle import Angle
from models.joint import Joint
from models.segment import Segment
from processors.angles_processor import AnglesProcessor
from processors.base import Processor
from processors.joints_processor import JointsProcessor
from scipy.signal import find_peaks


class SegmentsProcessor(Processor):
    def __init__(self, fps: int, segmentation_features: dict) -> None:
        super().__init__()
        self.fps = fps
        self.segmentation_features = segmentation_features

    def process(self, data: tuple[list[Joint], list[Angle]]) -> list[Segment]:
        joints, angles = data
        peaks, valleys = self._get_peaks_and_valleys(angles)
        segments_frames = self._get_segments_indexes(peaks, valleys)

        segments = []
        for rep, (start_frame, finish_frame) in enumerate(segments_frames, 1):
            segment_angles = [
                angle for angle in angles if start_frame <= angle.frame <= finish_frame
            ]
            segment_joints = [
                joint for joint in joints if start_frame <= joint.frame <= finish_frame
            ]
            segments.append(
                Segment(
                    start_frame=start_frame,
                    finish_frame=finish_frame,
                    rep=rep,
                    joints=segment_joints,
                    angles=segment_angles,
                )
            )
        return segments

    def update(self, data: list[Segment]) -> None:
        self.data = data

    @staticmethod
    def to_df(data: Segment) -> tuple[pd.DataFrame, pd.DataFrame]:
        return JointsProcessor.to_df(data.joints), AnglesProcessor.to_df(data.angles)

    @staticmethod
    def from_df(df: tuple[pd.DataFrame, pd.DataFrame]) -> list[Segment]:
        joints_df, angles_df = df
        joints = JointsProcessor.from_df(joints_df)
        angles = AnglesProcessor.from_df(angles_df)
        start_frame = joints[0].frame
        finish_frame = joints[-1].frame

        return Segment(
            rep=0,
            start_frame=start_frame,
            finish_frame=finish_frame,
            joints=joints,
            angles=angles,
        )

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)
        for segment in self.data:
            segment_file = f"rep_{segment.rep}"
            results_path = os.path.join(output, segment_file)
            os.makedirs(results_path, exist_ok=True)

            joints_df, angles_df = self.to_df(segment)
            joints_df.to_csv(os.path.join(results_path, "joints.csv"))
            angles_df.to_csv(os.path.join(results_path, "angles.csv"))

    def _get_peaks_and_valleys(self, angles: list[Angle]) -> np.ndarray:
        data = defaultdict(list)
        for angle in angles:
            if angle.name in self.segmentation_features:
                data[angle.frame].append(angle.value)

        exercise_signal = np.array([np.mean(values) for values in data.values()])
        zero_point = np.mean(exercise_signal)

        peaks, _ = find_peaks(exercise_signal, zero_point)
        valleys, _ = find_peaks(-exercise_signal, -zero_point)
        return peaks, valleys

    def _get_segments_indexes(
        self, peaks: np.ndarray, valleys: np.ndarray
    ) -> list[list[int, int]]:
        segments = []
        valley_idx = 0
        peaks_idx = 0
        while peaks_idx < len(peaks) - 1:
            if peaks[peaks_idx] < valleys[valley_idx]:
                if peaks[peaks_idx + 1] < valleys[valley_idx]:
                    peaks[peaks_idx + 1] = (
                        peaks[peaks_idx] + peaks[peaks_idx + 1]
                    ) // 2
                else:
                    segments.append([peaks[peaks_idx], peaks[peaks_idx + 1]])
                peaks_idx += 1

            else:
                if valley_idx >= len(valleys) - 1:
                    break
                if valleys[valley_idx + 1] < peaks[peaks_idx]:
                    valleys[valley_idx + 1] = (
                        valleys[valley_idx] + valleys[valley_idx + 1]
                    ) // 2
                valley_idx += 1
        return segments
