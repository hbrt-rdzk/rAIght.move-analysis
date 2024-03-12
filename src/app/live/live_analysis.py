import cv2
import mediapipe as mp

from app.base import App
from processors.repetitions.processor import RepetitionsProcessor


class LiveAnalysisApp(App):
    """
    Real-time angles visualization and reps countning.
    """

    def __init__(self, exercise: str):
        super().__init__(exercise)
        self.repetitions_processor = RepetitionsProcessor(self._exercise)

    def run(self, input: str, output: str, save_results: bool, loop: bool) -> None:
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
                joints = self._joints_processor.load_data(world_landmards)
                filtered_joints = self._joints_processor.filter(joints)
                self._joints_processor.update(filtered_joints)

                # Angle processing
                angles = self._angles_processor.load_data(filtered_joints)
                self._angles_processor.update(angles)

                # Exercise state
                progress = self.repetitions_processor.load_data(angles)
                self.repetitions_processor.update(progress)

                # Updating window
                self._visualizer.update_figure(
                    filtered_joints,
                    angles,
                    progress,
                    self.repetitions_processor.repetitions_count,
                    self.repetitions_processor.state,
                )
                self._visualizer.draw_landmarks(
                    frame, landmarks, self._mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("Mediapipe", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if save_results:
            try:
                self._joints_processor.save(output)
                self._angles_processor.save(output)
                self.repetitions_processor.save(output)
            except ValueError as error:
                self.logger.critical(f"Error on trying to save results:\n{error}")
