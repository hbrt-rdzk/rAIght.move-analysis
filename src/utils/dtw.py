from typing import Any

import numpy as np
from dtw import dtw


class DTW:
    """Dynamic time warping implementation
    for comparing two time serieses"""

    def __init__(self) -> None:
        pass

    def __call__(
        self, query: np.ndarray, reference: np.ndarray, *args: Any, **kwds: Any
    ) -> Any:
        alignment = dtw(query, reference, keep_internals=True)
        return alignment

    def __filter_repetable_reference_indexes(
        self, referene_to_query: np.ndarray, query_to_refernce: np.ndarray
    ) -> np.ndarray:
        query_to_refernce_cp = query_to_refernce.copy()
        for idx in range(len(referene_to_query) - 2, 1, -1):
            if referene_to_query[idx] == referene_to_query[idx + 1]:
                query_to_refernce_cp = np.delete(query_to_refernce_cp, idx)

        return query_to_refernce_cp
