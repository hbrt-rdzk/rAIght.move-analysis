from src.processors.angles.angle import Angle
from src.processors.joints.joint import Joint
from src.utils.dtw import DTW


class Segment:
    """
    One repetition timeframe with it's data
    """

    def __init__(
        self,
        rep: int,
        start_frame: int,
        finish_frame: int,
        joints: list[Joint],
        angles: list[Angle],
    ) -> None:
        self.repetition_index = rep
        self.start_frame = start_frame
        self.finish_frame = finish_frame
        self.joints = joints
        self.angles = angles

    def compare_with_reference(self, reference):
        # dtw()
        pass
