from dataclasses import dataclass


@dataclass
class ExerciseState:
    exercise_name: str
    step: int
    angles: list[str]
