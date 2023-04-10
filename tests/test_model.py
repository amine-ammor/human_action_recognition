import pytest

import torch

@pytest.mark.parametrize("input",[torch.rand(10,15,34),])
def test_model_output(model,input):
    output = model(input)
    assert output.shape == torch.Size([input.shape[0],model.nb_actions])
