from abc import ABC, abstractmethod

from models.state import ExerciseState


class Exercise(ABC):
    def __init__(
        self, query_states: list[ExerciseState], reference_states: list[ExerciseState]
    ) -> None:
        self.query_states = query_states
        self.reference_states = reference_states

    @abstractmethod
    def compare_states(self) -> dict: ...
