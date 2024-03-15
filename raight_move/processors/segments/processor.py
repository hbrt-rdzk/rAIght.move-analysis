import os

import numpy as np
import pandas as pd
from processors.angles.angle import Angle
from processors.angles.processor import AnglesProcessor
from processors.base import Processor
from processors.joints.joint import Joint
from processors.joints.processor import JointsProcessor
from processors.segments.segment import Segment
from sklearn.preprocessing import MinMaxScaler
from utils.dtw import get_warped_frame_indexes


class SegmentsProcessor(Processor):
    def __init__(self, fps: int, segmentation_parameters: dict) -> None:
        super().__init__()
        self.fps = fps
        self.segmentaion_parameters = segmentation_parameters

    def process(self, data: tuple[list[Joint], list[Angle]]) -> list[Segment]:
        joints, angles = data
        joints_per_frame = len(np.unique([joint.name for joint in joints]))
        angles_per_frame = len(np.unique([angle.name for angle in angles]))

        exercise_signal = self.__get_exercise_signal(angles)
        filtered_exercise_signal = self.__filter_signal(exercise_signal)

        threshold = filtered_exercise_signal.mean()
        breakpoints = self.__get_breakpoints(filtered_exercise_signal, threshold)
        segments = []
        for rep, (start_frame, finish_frame) in enumerate(breakpoints):
            segment_joints = joints[
                start_frame * joints_per_frame : finish_frame * joints_per_frame
            ]
            segment_angles = angles[
                start_frame * angles_per_frame : finish_frame * angles_per_frame
            ]

            segments.append(
                Segment(rep, start_frame, finish_frame, segment_joints, segment_angles)
            )
        return self.__filter_segments(segments, filtered_exercise_signal)

    def update(self, data: list[Segment]) -> None:
        self.data = data

    @staticmethod
    def to_df(data: Segment) -> tuple[pd.DataFrame, pd.DataFrame]:
        return JointsProcessor.to_df(data.joints), AnglesProcessor.to_df(data.angles)

    @staticmethod
    def from_df(data: tuple[pd.DataFrame, pd.DataFrame]) -> list[Segment]:
        joints, angles = data
        joints = JointsProcessor.from_df(joints)
        angles = AnglesProcessor.from_df(angles)
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
            os.mkdir(results_path)

            joints_df, angles_df = self.to_df(segment)

            joints_df.to_csv(os.path.join(results_path, "joints.csv"), index=False)
            angles_df.to_csv(os.path.join(results_path, "angles.csv"), index=False)

    def compare_segments(self, query: Segment, reference: Segment) -> list:
        query_signal = self.__get_exercise_signal(query.angles)
        reference_signal = self.__get_exercise_signal(reference.angles)
        query_to_reference_indexes = get_warped_frame_indexes(
            query_signal, reference_signal
        )

    def __get_exercise_signal(self, angles: list[Angle]) -> np.ndarray:
        angles_df = AnglesProcessor.to_df(angles).pivot(
            index="frame", columns="name", values="value"
        )
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

        return important_angles_normalized.mean(axis=1)

    def __get_breakpoints(self, signal: np.ndarray, threshold: float) -> list:
        sliding_window_size = self.segmentaion_parameters["sliding_window_size"]
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
            [start, end] for start, end in zip(breakpoints[0::2], breakpoints[2::2])
        ]

    def __filter_signal(self, signal: np.ndarray) -> np.ndarray:
        cutoff_freq = self.segmentaion_parameters["cutoff_freq"]
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
                lambda segment: is_valid_segment(segment),
                segments,
            )
        )
        self.__reset_segments_indexes(valid_segments)
        return valid_segments

    @staticmethod
    def __reset_segments_indexes(segments: list[Segment]) -> None:
        for idx, segment in enumerate(segments, start=1):
            segment.rep = idx
