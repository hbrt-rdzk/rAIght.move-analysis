import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.angle import Angle
from models.joint import Joint
from models.segment import Segment
from processors.angles_processor import AnglesProcessor
from processors.base import Processor
from processors.joints_processor import JointsProcessor


class SegmentsProcessor(Processor):
    def __init__(self, fps: int, segmentation_parameters: dict) -> None:
        super().__init__()
        self.fps = fps
        self.segmentaion_parameters = segmentation_parameters

    def process(self, data: tuple[list[Joint], list[Angle]]) -> list[Segment]:
        joints, angles = data

        exercise_signal = self.__get_exercise_signal(angles)
        filtered_exercise_signal = self.__filter_signal(exercise_signal)
        threshold = filtered_exercise_signal.mean()
        breakpoints = self.__get_breakpoints(filtered_exercise_signal, threshold)

        segments = []
        for rep, (start_frame, finish_frame) in enumerate(breakpoints):
            segment_joints = [
                joint for joint in joints if start_frame <= joint.frame <= finish_frame
            ]
            segment_angles = [
                angle for angle in angles if start_frame <= angle.frame <= finish_frame
            ]
            segments.append(
                Segment(start_frame, finish_frame, rep, segment_joints, segment_angles)
            )
        return self.__filter_segments(segments, filtered_exercise_signal)

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

    def __get_exercise_signal(self, angles: list[Angle]) -> np.ndarray:
        angles_df = AnglesProcessor.to_df(angles)
        scaler = MinMaxScaler()
        important_features = (
            angles_df.std()
            .sort_values(ascending=False)[
                : self.segmentaion_parameters["signal_features"]
            ]
            .keys()
        )

        important_angles_df = angles_df[important_features]
        important_angles_normalized = scaler.fit_transform(important_angles_df)

        return self.__filter_signal(important_angles_normalized.mean(axis=1))

    def __get_breakpoints(self, signal: np.ndarray, threshold: float) -> list:
        sliding_window_size = (
            self.fps // self.segmentaion_parameters["sliding_window_scaler"]
        )
        tolerance = self.segmentaion_parameters["tolerance"]
        stride = self.segmentaion_parameters["stride"]

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
            [start, end + self.fps]
            for start, end in zip(breakpoints[0::2], breakpoints[2::2])
        ]

    def __filter_signal(self, signal: np.ndarray) -> np.ndarray:
        cutoff_freq = self.fps
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

        def is_valid_segment(segment: Segment) -> bool:
            threshold_height_scaler = self.segmentaion_parameters[
                "threshold_height_scaler"
            ]
            segment_length = segment.finish_frame - segment.start_frame
            signal_height_diff = np.max(
                signal[segment.start_frame : segment.finish_frame]
            ) - np.min(signal[segment.start_frame : segment.finish_frame])

            if (
                median_rep_length + self.fps <= segment_length
                or segment_length <= median_rep_length - self.fps
            ):
                return False

            if (
                median_rep_height * threshold_height_scaler < signal_height_diff
                or signal_height_diff < median_rep_height / threshold_height_scaler
            ):
                return False
            return True

        valid_segments = list(
            filter(
                is_valid_segment,
                segments,
            )
        )
        self.__reset_segments_indexes(valid_segments)
        return valid_segments

    @staticmethod
    def __reset_segments_indexes(segments: list[Segment]) -> None:
        for idx, segment in enumerate(segments, start=1):
            segment.rep = idx
