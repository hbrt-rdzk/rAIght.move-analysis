import numpy as np
from tslearn.metrics import dtw_path


def get_warped_frame_indexes(query: np.ndarray, reference: np.ndarray) -> list:
    path, _ = dtw_path(query, reference)
    path = np.array(path)
    return path


def filter_repetable_reference_indexes(
    referene_to_query: np.ndarray, query_to_refernce: np.ndarray
) -> np.ndarray:
    query_to_refernce_cp = query_to_refernce.copy()

    for idx in range(len(referene_to_query) - 1, -1, -1):
        if idx > 0 and referene_to_query[idx] == referene_to_query[idx - 1]:
            query_to_refernce_cp = np.delete(query_to_refernce_cp, idx)

    return query_to_refernce_cp
