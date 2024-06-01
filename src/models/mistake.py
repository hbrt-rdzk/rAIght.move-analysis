from dataclasses import dataclass


@dataclass(frozen=True)
class Mistake:
    """
    Mistake object that handles feedback
    """

    exercise: str
    mistake_name: str
    fix_info: str
    angle_name: str
    threshold: float
