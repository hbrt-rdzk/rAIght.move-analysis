from dataclasses import dataclass

ANGLE_PARAMETERS_NUM = 3
ANGLES_PER_FRAME = 8


@dataclass
class Angle:
    """
    Angle calculated from joint positions
    """

    frame: int
    name: str
    value: float
