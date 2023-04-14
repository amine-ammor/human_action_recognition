import pytest,torch



@pytest.fixture
def random_input_sample(video_dataloader):
    """for the model"""
    assert type(video_dataloader.sampler) == torch.utils.data.sampler.RandomSampler
    res = next(iter(video_dataloader))
    return res

def test_model_output(model,random_input_sample):
    model = model.eval()

    batch,batch_frames_valid = random_input_sample

    output = model(batch,batch_frames_valid)

    batch[~batch_frames_valid] = 10.0
    output1 = model(batch,batch_frames_valid)

    assert torch.all(output == output1),"masked frames should not interfere in the result"

    assert batch.shape[2] == model.d_keypoints,"""tralling dimension of input, should be equal
      to the number keypoints"""
    assert output.shape == torch.Size([batch.shape[0],model.nb_actions])