import os

import cv2
import pandas as pd

from src.app.base import OUTPUT_PATH_FIELD, POSE_ESTIMATION_MODEL_NAME, App
from src.processors.angles.processor import AnglesProcessor
from src.processors.joints.processor import JointsProcessor
from src.processors.segments.processor import SegmentsProcessor
from src.processors.segments.segment import Segment

PATH_TO_REFERENCE = "data/{exercise}/features/reference"


class VideoAnalysisApp(App):
    """
    More advanced whole video analysis.
    """

    def __init__(self, exercise: str) -> None:
        super().__init__()
        model_config_data = self._config_data[POSE_ESTIMATION_MODEL_NAME]
        self.angle_names = model_config_data["angles"]
        self.joint_names = model_config_data["joints"]
        self.connections = model_config_data["connections"]["torso"]
        reference_joints = pd.read_csv(
            os.path.join(PATH_TO_REFERENCE.format(exercise=exercise), "joints.csv")
        )
        reference_angles = pd.read_csv(
            os.path.join(PATH_TO_REFERENCE.format(exercise=exercise), "angles.csv")
        )
        self.reference_segment = Segment(
            rep=0,
            start_frame=0,
            finish_frame=len(reference_angles),
            joints=AnglesProcessor.from_df(reference_joints),
            angles=JointsProcessor.from_df(reference_angles),
        )

    def run(self, input: str, output: str, save_results: bool, loop: bool) -> None:
        joints_processor = JointsProcessor(self.joint_names)
        angles_processor = AnglesProcessor(self.angle_names)

        cap = cv2.VideoCapture(input)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        segments_processor = SegmentsProcessor(fps)

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
                joints = joints_processor.process(world_landmards)
                joints_processor.update(joints)

                # Angle processing
                angles = angles_processor.process(joints)
                angles_processor.update(angles)
        cap.release()

        frames_num = len(angles_processor.data)
        analysis_info = (
            f"Analyzed {frames_num} frames"
            f", extracted {len(joints_processor)} joint feature and"
            f" {len(angles_processor)} angle features."
        )
        self.logger.info(analysis_info)

        segments = segments_processor.process(
            data=(joints_processor.data, angles_processor.data)
        )
        segments_processor.update(segments)

        segments_info = {
            f"repetition {segment.repetition_index}": f"frames: [{segment.start_frame}; {segment.finish_frame}]"
            for segment in segments
        }
        self.logger.info(f"Segmented video frames: {segments_info}")

        for segment in segments:
            self.logger.info(
                f"Starting comparison with reference video for rep: {segment.repetition_index}..."
            )
            # TODO: comparison between reference and query video
            # results = segments_processor.compare_with_reference(self.reference_segment)
