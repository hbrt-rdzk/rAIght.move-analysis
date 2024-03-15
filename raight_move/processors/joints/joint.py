from dataclasses import dataclass

JOINT_PARAMETERS_NUM = 7


@dataclass
class Joint:
    """
    Joint extracted by DNN model from video
    """

    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float
    frame: int
