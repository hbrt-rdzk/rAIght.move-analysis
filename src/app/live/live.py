import cv2
import mediapipe as mp

from src.app.abstract_app import App
from src.processors.angles.angles_processor import AnglesProcessor
from src.processors.joints.joints_processor import JointsProcessor
from src.processors.repetitions.repetitions_processor import \
    RepetitionsProcessor
from src.utils.visualizer import Visualizer


class LiveAnalysisApp(App):
    def run(self, args):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        repetitions_handler = RepetitionsProcessor(args.exercise)
        joints_handler = JointsProcessor("pose_landmarker")
        angles_handler = AnglesProcessor("pose_landmarker")
        visualizer = Visualizer("pose_landmarker")

        cap = cv2.VideoCapture(args.input)
        cv2.namedWindow("Mediapipe", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Mediapipe", 0, -100)

        if not cap.isOpened():
            print("Error on opening video stream or file!")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    cap = cv2.VideoCapture(args.input)
                    continue
                else:
                    break

            results = pose.process(frame)
            landmarks = results.pose_landmarks
            world_landmards = results.pose_world_landmarks

            if world_landmards:
                # Joints processing
                joints = joints_handler.load_data(world_landmards)
                filtered_joints = joints_handler.filter(joints)
                joints_handler.update(filtered_joints)

                # Angle processing
                angles = angles_handler.load_data(filtered_joints)
                angles_handler.update(angles)

                # Exercise state
                progress = repetitions_handler.load_data(angles)
                repetitions_handler.update(progress)

                # Updating window
                visualizer.update_figure(
                    filtered_joints,
                    angles,
                    progress,
                    repetitions_handler.repetitions_count,
                    repetitions_handler.state,
                )

            visualizer.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Mediapipe", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if args.save_results:
            try:
                joints_handler.save(args.output)
                angles_handler.save(args.output)
                repetitions_handler.save(args.output)
            except ValueError as error:
                print(f"Error on trying to save results:\n{error}")
