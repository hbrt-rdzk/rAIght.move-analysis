import argparse

import cv2
import mediapipe as mp

from src.utils.joints_data_handler import JointsDataHandler
from src.utils.visualizer import Visualizer


def run(args) -> None:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    visualizer = Visualizer("pose_landmarker")
    handler = JointsDataHandler("pose_landmarker", args.exercise)

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

        results = pose.process(frame).pose_landmarks
        if results:
            joints = handler.load_joints_from_landmark(results)
            handler.update_joints(joints)
            angles = handler.calculate_angles()

            visualizer.phase = handler.get_exercise_phase(angles)
            visualizer.update_figure(joints, angles)
            mp_drawing.draw_landmarks(frame, results, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Mediapipe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.save_results:
        try:
            handler.save_joints(args.output)
        except ValueError as error:
            print(f"Error on trying to save results:\n{error}")


def main():
    parser = argparse.ArgumentParser(description="Visualization parameters")

    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-i",
        "--input",
        help="Camera numer or path to video",
        default=0,
        type=lambda x: int(x) if x.isdigit() else x,
        required=True,
    )
    required.add_argument(
        "--exercise",
        help="Type of exercise",
        default="squat",
        type=str,
        required=True,
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-o",
        "--output",
        help="Output path for joints results",
    )
    optional.add_argument(
        "--save_results",
        help="If joints should be saved or not",
        action="store_true",
    )
    optional.add_argument(
        "--loop",
        help="If video should be looped or not",
        action="store_true",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
