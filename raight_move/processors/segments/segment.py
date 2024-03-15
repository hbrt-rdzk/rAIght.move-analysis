from dataclasses import dataclass

from processors.angles.angle import Angle
from processors.joints.joint import Joint


@dataclass
class Segment:
    rep: int
    start_frame: int
    finish_frame: int
    joints: list[Joint]
    angles: list[Angle]
