import os

import cv2
import pandas as pd

from app.base import POSE_ESTIMATION_MODEL_NAME, App
from models.angle import Angle
from models.joint import Joint
from processors.angles_processor import AnglesProcessor
from processors.joints_processor import JointsProcessor
from processors.mistakes_processor import MistakesProcessor
from processors.results_processor import ResultsProcessor
from processors.segments_processor import SegmentsProcessor

PATH_TO_REFERENCE = "data/{exercise}/features/reference"


class VideoAnalysisApp(App):
    """
    More advanced whole video analysis.
    """

    def __init__(self, exercise: str) -> None:
        super().__init__()
        self.exercise = exercise
        self.segmentation_params = self._segmentation_config["segmentation_params"]
        self.comparison_features = self._exercise_table[exercise]["comparison_features"]
        self.mistakes_table = self._exercise_table[exercise]["mistakes_table"]

        model_config_data = self._pose_estimation_config[POSE_ESTIMATION_MODEL_NAME]
        self.angle_names = model_config_data["angles"]
        self.joint_names = model_config_data["joints"]
        self.connections = model_config_data["connections"]["torso"]

        path_to_reference = os.path.join(PATH_TO_REFERENCE.format(exercise=exercise))
        reference_joints = pd.read_csv(os.path.join(path_to_reference, "joints.csv"))
        reference_angles = pd.read_csv(os.path.join(path_to_reference, "angles.csv"))
        self.reference_segment = SegmentsProcessor.from_df(
            (reference_joints, reference_angles)
        )

    def run(self, input_source: str, output: str, save_results: bool) -> None:
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            self.logger.critical("❌ Error on opening video stream or file! ❌")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        joint_data, angles_data = self.extract_features(cap)

        segments_processor = SegmentsProcessor(fps, self.segmentation_params)
        mistakes_proecssor = MistakesProcessor(self.mistakes_table, self.exercise)
        results_processor = ResultsProcessor(
            self.reference_segment, self.comparison_features
        )

        self.logger.info(
            "Analyzed %s frames 🖼️, extracted %s joint features 💪 and %s angle features 📐",
            video_length,
            len(joint_data),
            len(angles_data),
        )
        segments = segments_processor.process(data=(joint_data, angles_data))
        segments_processor.update(segments)

        segments_info = ", ".join(
            [
                f"repetition {segment.rep}: frames: [{segment.start_frame} - {segment.finish_frame}]"
                for segment in segments
            ]
        )
        print(segments_info)
        self.logger.info("Segmented video frames: %s", segments_info)

        for segment in segments:
            self.logger.info(
                "Starting comparison with reference video for rep: %s... 🔥",
                segment.rep,
            )
            results = results_processor.process(segment)
            results_processor.update(results)

            feedback = mistakes_proecssor.process(results)
            mistakes_proecssor.update(feedback)

        self.logger.info("Analysis complete! ✅")
        if save_results:
            try:
                segments_processor.save(output)
                results_processor.save(output)
                self.logger.info("Results here: %s 💽", output)
            except ValueError as error:
                self.logger.critical("Error on trying to save results:\n %s", error)

    def extract_features(
        self, cap: cv2.VideoCapture
    ) -> tuple[list[Joint], list[Angle]]:
        joints_processor = JointsProcessor(self.joint_names)
        angles_processor = AnglesProcessor(self.angle_names)

        self.logger.info("Starting features extraction from video... 🎬")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self._pose_estimation_model.process(frame)
            world_landmards = results.pose_world_landmarks

            if world_landmards:
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                JointsProcessor.current_processing_frame = frame_number

                joints = joints_processor.process(world_landmards)
                joints_processor.update(joints)

                angles = angles_processor.process(joints)
                angles_processor.update(angles)
        cap.release()

        return joints_processor.data, angles_processor.data
