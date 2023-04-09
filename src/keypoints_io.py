import numpy as np
from collections import abc

def load(path):
    data = np.load(path,allow_pickle=True)
    nb_kpts = len([el for el in data.keys() if "kpt" in el ])
    nb_scores = len([el for el in data.keys() if "score" in el ])
    assert nb_kpts == nb_scores,(nb_scores,nb_kpts)

    keypoints = [data["kpt_"+str(idx)] for idx in range(nb_kpts)]
    scores = [data["score_"+str(idx)] for idx in range(nb_scores)]

    return keypoints,scores

def save(path,data):
    keypoints,scores = data 
    assert isinstance(keypoints,abc.Iterable)
    assert all([isinstance(el,np.ndarray) for el in keypoints])
    np.savez(path,
             **{"kpt_"+str(idx):kpt for (idx,kpt) in enumerate(keypoints)},
             **{"score_"+str(idx):score for (idx,score) in enumerate(scores)},
    )
