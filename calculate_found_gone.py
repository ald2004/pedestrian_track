import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_found_gone(current_frame: np.ndarray, last_frame: np.ndarray, THRES=.7):
    assert current_frame.ndim == 2 & last_frame.ndim == 2
    current_frame, last_frame = current_frame.reshape(-1, 128), last_frame.reshape(-1, 128)
    cdistance = cosine_similarity(current_frame, last_frame)
    indexes = np.where(cdistance > THRES)
    found_current_index, found_last_index = np.unique(indexes[0]), np.unique(indexes[1])
    # np.take(a,found_c_i) np.delete(a,found_c_id
    return (found_current_index, found_last_index)
