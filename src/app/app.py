import argparse

import cv2
import mediapipe as mp

from src.utils.joints_data_handler import JointsDataHandler
from src.utils.visualizer import Visualizer


def run(args: dict):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    visualizer = Visualizer("pose_landmarker")
    _, ax = visualizer.init_3djoints_figure()

    handler = JointsDataHandler("pose_landmarker")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(args.input)
    frame_idx = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if args.loop:
                cap = cv2.VideoCapture(args.input)
                frame_idx = 0
                continue
            else:
                break

        results = pose.process(frame).pose_landmarks
        if results:
            mp_drawing.draw_landmarks(frame, results, mp_pose.POSE_CONNECTIONS)

            joints = handler.load_joints_from_landmark(results)
            visualizer.update_3djoints(ax, joints)
            handler.add_joints(joints, frame_idx)

        cv2.imshow("Mediapipe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if args.save_joints:
        handler.save_joints()


def main():
    parser = argparse.ArgumentParser(description="Visualization parameters")
    parser.add_argument(
        "-i",
        "--input",
        help="Camera numer or path to video",
        default=0,
        type=lambda x: int(x) if x.isdigit() else x,
    )
    parser.add_argument(
        "--save_joints",
        help="If joints should be saved or not",
        action="store_true",
    )
    parser.add_argument(
        "--loop",
        help="If video should be looped or not",
        action="store_true",
    )
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
