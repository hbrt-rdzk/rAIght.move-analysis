from dataclasses import dataclass


@dataclass
class Result:
    """
    Results from frame comparison
    """

    frame: int
    rep: int
    angle_name: str
    diff: float
