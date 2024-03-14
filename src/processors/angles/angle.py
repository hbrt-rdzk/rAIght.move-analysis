from dataclasses import dataclass


@dataclass
class Angle:
    """
    Angle calculated from joint positions
    """

    name: str
    value: float
    frame: int
