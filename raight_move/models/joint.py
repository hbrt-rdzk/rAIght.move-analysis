from dataclasses import dataclass

JOINT_PARAMETERS_NUM = 7
JOINTS_PER_FRAME = 16


@dataclass
class Joint:
    """
    Joint extracted by DNN model from video
    """

    frame: int
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float
