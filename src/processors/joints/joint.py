from dataclasses import dataclass


@dataclass
class Joint:
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float
