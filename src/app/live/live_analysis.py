import cv2

from src.app.base import OUTPUT_PATH_FIELD, POSE_ESTIMATION_MODEL_NAME, App
from src.processors.angles.processor import AnglesProcessor
from src.processors.joints.processor import JointsProcessor
from src.utils.repetitions_counter import RepetitionsCounter
from src.utils.visualizer import Visualizer


class LiveAnalysisApp(App):
    """
    Real-time angles visualization and reps countning.
    """

    def __init__(self, exercise: str) -> None:
        super().__init__()
        self.results_path = self._config_data[OUTPUT_PATH_FIELD]
        self.exercise_phases = self._reference_table[exercise]["phases"]

        model_config_data = self._config_data[POSE_ESTIMATION_MODEL_NAME]
        self.angle_names = model_config_data["angles"]
        self.joint_names = model_config_data["joints"]
        self.connections = model_config_data["connections"]["torso"]

    def run(self, input: str, output: str, save_results: bool, loop: bool) -> None:
        joints_processor = JointsProcessor(self.joint_names)
        angles_processor = AnglesProcessor(self.angle_names)

        visualizer = Visualizer(self.connections)
        repetitions_counter = RepetitionsCounter(self.exercise_phases)

        cap = cv2.VideoCapture(input)
        cv2.namedWindow("Mediapipe", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Mediapipe", 0, -100)

        if not cap.isOpened():
            self.logger.critical("Error on opening video stream or file!")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap = cv2.VideoCapture(input)
                    continue
                else:
                    break

            results = self._pose_estimation_model.process(frame)
            landmarks = results.pose_landmarks
            world_landmards = results.pose_world_landmarks

            if world_landmards:
                # Joints processing
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
                self.logger.critical(f"Error on trying to save results:\n{error}")
