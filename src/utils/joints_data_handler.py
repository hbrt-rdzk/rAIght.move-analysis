import os

import numpy as np
import pandas as pd
import yaml

OUTPUT_COLUMNS = ("frame", "x", "y", "z", "visibility", "joint_id")
CONFIG_PATH = "configs/config.yaml"


class JointsDataHandler:
    def __init__(self, model: str) -> None:
        try:
            with open(CONFIG_PATH) as config:
                data = yaml.safe_load(config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")

        self.joint_names = data[model]["joints"]
        self.angles = data[model]["angles"]
        self.results_path = data["joints_output_path"]
        self.joints = []
        self._frames_num = 0

    def update_joints(self, joints: np.ndarray) -> None:
        frame_column = np.full((joints.shape[0], 1), self._frames_num)
        self.joints.append(np.hstack((frame_column, joints)))
        self._frames_num += 1

    def load_joints_from_landmark(self, landmark_joints: np.ndarray) -> np.ndarray:
        joint_ids = list(map(int, self.joint_names.keys()))
        joints = np.array(
            [
                np.array([joint.x, joint.y, joint.z, joint.visibility, idx])
                for idx, joint in enumerate(landmark_joints.landmark)
            ]
        )
        return joints[np.isin(joints[:, -1].astype(int), joint_ids)]

    def calculate_angles(self) -> dict[str, float]:
        angles = {}
        for name, joint_nums in self.angles.items():
            angles[name] = self.__calculate_angle_between_joints(*joint_nums)
        return angles

    def save_joints(self, output_filename: str) -> None:
        if not self.joints:
            raise ValueError("No joints data to save.")
        if not output_filename:
            raise ValueError("Output filename not provided.")

        os.makedirs(self.results_path, exist_ok=True)
        output_path = os.path.join(self.results_path, output_filename)

        joints_ = np.concatenate(self.joints)
        joints_df = pd.DataFrame(joints_, columns=OUTPUT_COLUMNS)
        joints_df["frame"] = joints_df["frame"].astype(int)
        joints_df["joint_id"] = joints_df["joint_id"].astype(int)

        joints_df.to_csv(output_path, index=False)

    def __calculate_angle_between_joints(
        self, joint_1_id: int, joint_2_id: int, joint_3_id: int, frame: int = -1
    ) -> float:
        if frame >= len(self.joints) or frame < -len(self.joints):
            raise IndexError("Frame index out of range.")

        desired_frame_joints = self.joints[frame]
        return self.calculate_3d_angle(
            desired_frame_joints[desired_frame_joints[:, 5] == joint_1_id][0, 1:4],
            desired_frame_joints[desired_frame_joints[:, 5] == joint_2_id][0, 1:4],
            desired_frame_joints[desired_frame_joints[:, 5] == joint_3_id][0, 1:4],
        )

    @staticmethod
    def calculate_3d_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        if not (A.shape == B.shape == C.shape == (3,)):
            raise ValueError("Input arrays must all be of shape (3,).")
        if not (A.dtype == B.dtype == C.dtype == np.float64):
            raise TypeError("Input arrays must all have dtype np.float64.")

        ba = A - B
        bc = C - B
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
