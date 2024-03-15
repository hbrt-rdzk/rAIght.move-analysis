from dataclasses import dataclass

ANGLE_PARAMETERS_NUM = 3


@dataclass
class Angle:
    """
    Angle calculated from joint positions
    """

    name: str
    value: float
    frame: int
