import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.app.abstract_app import App


class VideoAnalysisApp(App):
    """
    More advanced whole video analysis.
    """

    def run(
        self, input: str, exercise: str, output: str, save_results: bool, loop: bool
    ) -> None:
        cap = cv2.VideoCapture(input)

        if not cap.isOpened():
            self.logger.critical("Error on opening video stream or file!")
            return

        self.logger.info("Starting features extraction from video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self._pose_estimation_model.process(frame)
            world_landmards = results.pose_world_landmarks

            if world_landmards:
                # Joints processing
                joints = self._joints_processor.load_data(world_landmards)
                filtered_joints = self._joints_processor.filter(joints)
                self._joints_processor.update(filtered_joints)

                # Angle processing
                angles = self._angles_processor.load_data(filtered_joints)
                self._angles_processor.update(angles)

        cap.release()
        self.logger.info("Finished features extraction.")

        self.logger.info("Starting video segmentation...")
        reps_finish_frames = self.segment_video(self._angles_processor.data)
        self.logger.info("Finished video segmentation.")

        start_frame = 0
        for rep, finish_frame in enumerate(reps_finish_frames):
            self.logger.info(
                f"Starting comparison with reference video for rep: {rep + 1}..."
            )
            rep_joints = self._joints_processor.data[start_frame:finish_frame]

            start_frame = finish_frame

    def segment_video(self, angles: list[dict]) -> list:
        angles_df = pd.DataFrame(angles)
        important_features = angles_df.std().sort_values(ascending=False)[:4].keys()
        important_angles_df = angles_df[important_features]

        scaler = MinMaxScaler()
        important_angles_normalized_df = scaler.fit_transform(important_angles_df)
        exercise_signal = important_angles_normalized_df.sum(axis=1)
        zero_point = exercise_signal.mean()
        return self.__get_breakpoints(exercise_signal, zero_point)[1::2]

    @staticmethod
    def __get_breakpoints(
        signal: np.ndarray,
        zero_point: float,
        sliding_window_size: int = 5,
        stride: int = 1,
        tolerance: float = 0.1,
    ) -> list:
        state = "up"
        breakpoints = []
        for idx in range(0, len(signal) - sliding_window_size + 1, stride):
            window = signal[idx : idx + sliding_window_size]

            if state == "up":
                if (
                    abs(np.std(window) - 0.1) < tolerance
                    and np.mean(window) < zero_point
                ):
                    state = "down"
                    breakpoints.append(idx + sliding_window_size // 2)

            elif state == "down":
                if (
                    abs(np.std(window) - 0.1) < tolerance
                    and np.mean(window) > zero_point
                ):
                    state = "up"
                    breakpoints.append(idx + sliding_window_size // 2)

        return breakpoints
