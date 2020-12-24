import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_found_gone(current_frame: np.ndarray, last_frame: np.ndarray, THRES=.7):
    assert current_frame.ndim == 2 & last_frame.ndim == 2
    current_frame, last_frame = current_frame.reshape(-1, 128), last_frame.reshape(-1, 128)
    cdistance = cosine_similarity(current_frame, last_frame)
    maxmat_0 = (np.max(cdistance, axis=0)[None, ...] == cdistance) & (cdistance > THRES)
    maxmat_1 = (np.max(cdistance, axis=1)[..., None] == cdistance) & (cdistance > THRES)
    indexes = np.where(maxmat_0 & maxmat_1)
    # indexes = np.where(cdistance > THRES)
    # found_current_index, found_last_index = np.unique(indexes[0]).reshape(-1), np.unique(indexes[1]).reshape(-1)
    found_current_index, found_last_index = indexes[0].reshape(-1), indexes[1].reshape(-1)
    assert found_current_index.size == found_last_index.size
    # np.take(a,found_c_i) np.delete(a,found_c_id
    # npfname=f'/dev/shm/{uuid.uuid4().hex}'
    # np.save(npfname+'_c', current_frame)
    # np.save(npfname+'_l',last_frame)
    # np.save(npfname+'_d',cdistance)
    # print(current_frame, '\n', last_frame, '\n', cdistance)
    return (found_current_index, found_last_index)
