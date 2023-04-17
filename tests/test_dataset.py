import pytest,torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader



def test_dataset_samples(video_kpt_dataset):
    assert isinstance(len(video_kpt_dataset),int), "test can be performed only on finite dataset"
    for sample in video_kpt_dataset:
        nb_frames,kpts,is_detection_present,label = sample.values()
        import pdb;pdb.set_trace()
        assert kpts.shape[0] == is_detection_present.shape[0]
        assert kpts.shape[0]>=nb_frames

        assert (~is_detection_present[nb_frames:]).all(),"""frames corresponding to padding must
          contain no detection"""
        
        assert (kpts[~is_detection_present]==0).all(),"""
        kpts values are equal to 0 on frames with no detection"""

@pytest.fixture
# @pytest.mark.parametrize("batch_size",[10])
def unshuffled_data_loader(video_kpt_dataset):
    batch_size = 10
    d_loader = DataLoader(video_kpt_dataset,batch_size=batch_size,shuffle=False)
    return d_loader


def test_if_dataloader_batches_are_correctly_defined(unshuffled_data_loader,video_kpt_dataset):
    from itertools import islice
    batch_size = unshuffled_data_loader.batch_size
    for idx_batch,batch in enumerate(unshuffled_data_loader):
        for key in batch.keys():
            assert torch.all(batch[key]==torch.tensor([el[key] for el in 
            islice(video_kpt_dataset,batch_size*idx_batch,batch_size*(idx_batch+1))]))