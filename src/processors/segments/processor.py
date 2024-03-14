import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.processors.angles.angle import Angle
from src.processors.angles.processor import AnglesProcessor
from src.processors.base import Processor
from src.processors.joints.joint import Joint
from src.processors.joints.processor import JointsProcessor
from src.processors.segments.segment import Segment

SEGMENTATION_PARAMETERS_NUM = 4


class SegmentsProcessor(Processor):
    def __init__(self, fps) -> None:
        super().__init__()
        self.fps = fps

    def process(self, data: tuple[list[Joint], list[Angle]]) -> list[Segment]:
        joints, angles = data
        scaler = MinMaxScaler()
        angles_df = AnglesProcessor.to_df(angles)
        important_features = (
            angles_df.std()
            .sort_values(ascending=False)[:SEGMENTATION_PARAMETERS_NUM]
            .keys()
        )
        important_angles_df = angles_df[important_features]
        important_angles_normalized = scaler.fit_transform(important_angles_df)

        exercise_signal = important_angles_normalized.mean(axis=1)
        cutoff_freq = self.fps // 5
        filtered_exercise_signal = self.__filter_signal(exercise_signal, cutoff_freq)

        threshold = filtered_exercise_signal.mean()
        breakpoints = self.__get_breakpoints(filtered_exercise_signal, threshold)

        segments = []
        for rep, (start_frame, finish_frame) in enumerate(breakpoints):
            segment_joints = joints[start_frame:finish_frame]
            segment_angles = angles[start_frame:finish_frame]

            segments.append(
                Segment(rep, start_frame, finish_frame, segment_joints, segment_angles)
            )
        return self.__filter_segments(segments, filtered_exercise_signal)

    def update(self, data: list[Segment]) -> None:
        self.data = data

    def save(self, output_dir: str) -> None:
        output = self._validate_output(output_dir)
        for segment in self.data:
            segment_file = f"rep_{segment.rep}.csv"
            segment_df = self.to_df(segment)

            results_path = os.path.join(output, segment_file)
            segment_df.to_csv(results_path, index=False)

    @staticmethod
    def to_df(data: Segment) -> pd.DataFrame:
        joints_df = JointsProcessor.to_df(data.joints)
        angles_df = AnglesProcessor.to_df(data.angles)
        return pd.concat([joints_df, angles_df], axis=0, join="outer")

    @staticmethod
    def from_df(data: pd.DataFrame) -> Segment:
        ...

    def compare_with_reference(self, reference: Segment) -> list:
        ...

    @staticmethod
    def __get_breakpoints(
        signal: np.ndarray,
        threshold: float,
        sliding_window_size: int = 5,
        stride: int = 1,
        tolerance: float = 0.05,
    ) -> list:
        state = "down"
        breakpoints = []
        for idx in range(0, len(signal) - sliding_window_size + 1, stride):
            window = signal[idx : idx + sliding_window_size]

            if state == "up":
                if abs(np.std(window)) < tolerance and np.mean(window) < threshold:
                    state = "down"
                    breakpoints.append(idx + sliding_window_size // 2)

            elif state == "down":
                if abs(np.std(window)) < tolerance and np.mean(window) > threshold:
                    state = "up"
                    breakpoints.append(idx + sliding_window_size // 2)
        return [
            [start, end] for start, end in zip(breakpoints[0::2], breakpoints[2::2])
        ]

    @staticmethod
    def __filter_signal(signal: np.ndarray, cutoff_freq: int) -> np.ndarray:
        kernel = np.ones(cutoff_freq) / cutoff_freq
        filtered_signal = np.convolve(signal, kernel, mode="same")
        return filtered_signal

    def __filter_segments(
        self, segments: list[Segment], signal: np.ndarray
    ) -> list[Segment]:
        reps_length = [
            segment.finish_frame - segment.start_frame for segment in segments
        ]
        reps_height_diffs = [
            np.max(signal[segment.start_frame : segment.finish_frame])
            - np.min(signal[segment.start_frame : segment.finish_frame])
            for segment in segments
        ]

        median_rep_length = np.median(reps_length)
        median_rep_height = np.median(reps_height_diffs)

        def is_valid_segment(
            segment: Segment, fps: int, threshold_length: int, threshold_height: int
        ) -> bool:
            segment_length = segment.finish_frame - segment.start_frame
            signal_height_diff = np.max(
                signal[segment.start_frame : segment.finish_frame]
            ) - np.min(signal[segment.start_frame : segment.finish_frame])

            if (
                threshold_length + fps <= segment_length
                or segment_length <= threshold_length - fps
            ):
                return False

            if (
                threshold_height * 1.5 < signal_height_diff
                or signal_height_diff < threshold_height / 1.5
            ):
                return False
            return True

        valid_segments = list(
            filter(
                lambda segment: is_valid_segment(
                    segment, self.fps, median_rep_length, median_rep_height
                ),
                segments,
            )
        )
        self.__reset_segments_indexes(valid_segments)
        return valid_segments

    @staticmethod
    def __reset_segments_indexes(segments: list[Segment]) -> None:
        for idx, segment in enumerate(segments, start=1):
            segment.repetition_index = idx
