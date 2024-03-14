from dataclasses import dataclass

from src.processors.angles.angle import Angle
from src.processors.joints.joint import Joint


@dataclass
class Segment:
    rep: int
    start_frame: int
    finish_frame: int
    joints: list[Joint]
    angles: list[Angle]
