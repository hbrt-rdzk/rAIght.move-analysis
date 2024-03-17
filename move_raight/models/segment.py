from dataclasses import dataclass

from models.angle import Angle
from models.joint import Joint


@dataclass
class Segment:
    """
    One exercise repetition features
    """

    start_frame: int
    finish_frame: int
    rep: int
    joints: list[Joint]
    angles: list[Angle]
