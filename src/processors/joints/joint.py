from dataclasses import dataclass


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
