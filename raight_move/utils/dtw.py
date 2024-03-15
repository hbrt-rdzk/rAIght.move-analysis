from typing import Any

import numpy as np
import pandas as pd
from dtw import dtw


def get_warped_frame_indexes(query: np.ndarray, reference: np.ndarray) -> Any:
    alignment = dtw(query, reference, keep_internals=True)
    reference_to_query_warped = alignment.index1
    query_to_refernce_warped = alignment.index2
    return filter_repetable_reference_indexes(
        reference_to_query_warped, query_to_refernce_warped
    )


def filter_repetable_reference_indexes(
    referene_to_query: np.ndarray, query_to_refernce: np.ndarray
) -> np.ndarray:
    query_to_refernce_cp = query_to_refernce.copy()
    for idx in range(len(referene_to_query) - 2, 1, -1):
        if referene_to_query[idx] == referene_to_query[idx + 1]:
            query_to_refernce_cp = np.delete(query_to_refernce_cp, idx)

    return query_to_refernce_cp
