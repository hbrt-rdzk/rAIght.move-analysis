import cv2

from src.app.base import App
from src.processors.angles.processor import AnglesProcessor
from src.processors.joints.processor import JointsProcessor
from src.processors.repetitions.processor import RepetitionsProcessor
from src.utils.visualizer import Visualizer


class LiveAnalysisApp(App):
    """
    Real-time angles visualization and reps countning.
    """

    def __init__(self, exercise: str) -> None:
        super().__init__(exercise)

    def run(self, input: str, output: str, save_results: bool, loop: bool) -> None:
        joints_processor = JointsProcessor(self._pose_estimation_model_name)
        angles_processor = AnglesProcessor(self._pose_estimation_model_name)
        repetitions_processor = RepetitionsProcessor(self.exercise)

        visualizer = Visualizer(self._pose_estimation_model_name)

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
                progress = repetitions_processor.process(angles)
                repetitions_processor.update(progress)

                # Updating window
                visualizer.update_figure(
                    joints,
                    angles,
                    progress,
                    repetitions_processor.repetitions_count,
                    repetitions_processor.state,
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
                repetitions_processor.save(output)
            except ValueError as error:
                self.logger.critical(f"Error on trying to save results:\n{error}")
