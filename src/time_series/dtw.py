from typing import Any

import numpy as np
from dtw import dtw


class DTW:
    """Dynamic time warping implementation
    for comparing two time serieses"""

    def __init__(self) -> None:
        pass

    def __call__(
        self, x: np.ndarray, y: np.ndarray, dist: function, *args: Any, **kwds: Any
    ) -> Any:
        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=dist)
