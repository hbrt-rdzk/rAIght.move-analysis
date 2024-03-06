import cv2
import numpy as np

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
        self.__segment_video(self._angles_processor.data)

    def __segment_video(angles: np.ndarray) -> list:
        ...
