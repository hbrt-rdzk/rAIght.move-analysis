import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import plotly.graph_objects as go
from mediapipe.framework.formats.landmark_pb2 import (
    LandmarkList,
    NormalizedLandmarkList,
)

from src.utils.visualizer import Visualizer


def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    visualizer = Visualizer("pose_landmarker")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(frame)
        if results.pose_landmarks:
            visualizer.update_3djoints(ax, results.pose_landmarks)
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Mediapipe", frame)
    cap.release()


if __name__ == "__main__":
    main()
