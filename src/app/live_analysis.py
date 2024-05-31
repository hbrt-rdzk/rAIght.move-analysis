import cv2

from app.base import POSE_ESTIMATION_MODEL_NAME, App
from processors.angles_processor import AnglesProcessor
from processors.joints_processor import JointsProcessor
from utils.repetitions_counter import RepetitionsCounter
from utils.visualizer import Visualizer


class LiveAnalysisApp(App):
    """
    Real-time angles visualization and reps countning.
    """

    def __init__(self, exercise: str) -> None:
        super().__init__()
        self.exercise_phases = self._exercise_table[exercise]

        model_config_data = self._pose_estimation_config[POSE_ESTIMATION_MODEL_NAME]
        self.angle_names = model_config_data["angles"]
        self.joint_names = model_config_data["joints"]
        self.connections = model_config_data["connections"]["torso"]

    def run(self, input_source: str, output: str, save_results: bool) -> None:
        joints_processor = JointsProcessor(self.joint_names)
        angles_processor = AnglesProcessor(self.angle_names)

        visualizer = Visualizer(self.connections)
        repetitions_counter = RepetitionsCounter(self.exercise_phases)

        cap = cv2.VideoCapture(input_source)
        cv2.namedWindow("Mediapipe", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Mediapipe", 0, -100)

        if not cap.isOpened():
            self.logger.critical("Error on opening video stream or file!")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self._pose_estimation_model.process(frame)
            landmarks = results.pose_landmarks
            world_landmards = results.pose_world_landmarks

            if world_landmards:
                # Joints processing
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                JointsProcessor.current_processing_frame = frame_number

                joints = joints_processor.process(world_landmards)
                joints_processor.update(joints)

                # Angle processing
                angles = angles_processor.process(joints)
                angles_processor.update(angles)

                # Exercise state
                progress = repetitions_counter.process(angles)
                repetitions_counter.update(progress)

                # Updating window
                visualizer.update_figure(
                    joints,
                    angles,
                    progress,
                    repetitions_counter.repetitions_count,
                    repetitions_counter.state,
                )
                visualizer.draw_landmarks(
                    frame, landmarks, self._mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("Mediapipe", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if save_results:
            try:
                joints_processor.save(output)
                angles_processor.save(output)
            except ValueError as error:
                self.logger.critical("Error on trying to save results: %s", error)
