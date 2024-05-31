from dataclasses import dataclass


@dataclass
class Mistake:
    """
    Mistake object that handles feedback
    """

    exercise: str
    mistake_name: str
    angle_name: str
    threshold: float
    fix_info: str
