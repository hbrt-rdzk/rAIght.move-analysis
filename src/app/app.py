import argparse

import cv2
import mediapipe as mp

from src.processors.angle_handler import AnglesHandler
from src.processors.joint_handler import JointsHandler
from src.utils.visualizer import Visualizer


def parse_arguments() -> argparse.Namespace:
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
        "--save_joints",
        help="If joints should be saved or not",
        action="store_true",
    )
    optional.add_argument(
        "--save_angles",
        help="If angles should be saved or not",
        action="store_true",
    )
    optional.add_argument(
        "--loop",
        help="If video should be looped or not",
        action="store_true",
    )
    return parser.parse_args()


def run(args) -> None:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2
    )

    visualizer = Visualizer("pose_landmarker")
    joints_handler = JointsHandler("pose_landmarker")
    angles_handler = AnglesHandler("pose_landmarker", args.exercise)

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
            joints = joints_handler.load_data(results)
            joints_handler.update(joints)

            angles = angles_handler.load_data(joints)

            angles_handler.update(angles)
            visualizer.phase = angles_handler.get_exercise_phase()
            visualizer.update_figure(joints, angles)

        mp_drawing.draw_landmarks(
            frame,
            results,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(2, 82, 212), thickness=6, circle_radius=6),
            mp_drawing.DrawingSpec(color=(245, 255, 230), thickness=4, circle_radius=6),
        )
        cv2.imshow("Mediapipe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if args.save_joints:
        try:
            joints_handler.save(args.output)
        except ValueError as error:
            print(f"Error on trying to save results:\n{error}")
    if args.save_angles:
        try:
            angles_handler.save(args.output)
        except ValueError as error:
            print(f"Error on trying to save results:\n{error}")


def main():
    args = parse_arguments()
    run(args)


if __name__ == "__main__":
    main()
