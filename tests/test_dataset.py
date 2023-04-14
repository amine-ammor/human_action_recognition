import pytest,torch
from pathlib import Path
import numpy as np




def test_dataset_samples(video_kpt_dataset):
    assert isinstance(len(video_kpt_dataset),int), "test can be performed only on finite dataset"
    for sample in video_kpt_dataset:
        nb_frames,kpts,is_detection_present = sample.values()
        # import pdb;pdb.set_trace()
        assert kpts.shape[0] == is_detection_present.shape[0]
        assert kpts.shape[0]>=nb_frames

        assert (~is_detection_present[nb_frames:]).all(),"""frames corresponding to padding must
          contain no detection"""
        
        assert (kpts[~is_detection_present]==0).all(),"""
        kpts values are equal to 0 on frames with no detection"""


