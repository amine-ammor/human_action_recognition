import pytest,torch
from pathlib import Path
from ploomber_engine.ipython import PloomberClient

dir_file = Path(__file__).parent

@pytest.fixture
def path_training_setup():
    path = dir_file.joinpath("../notebooks/training_setup.ipynb")
    assert path.exists()
    return path

@pytest.fixture
def training_setup(path_training_setup):
    client = PloomberClient.from_path(path_training_setup,cwd=Path("./"))
    train_setup = client.get_namespace()
    return train_setup

@pytest.mark.temp
def test_dummy(training_setup):
    a = type(training_setup)
    import pdb;pdb.set_trace()


@pytest.mark.parametrize("input,batch_frames_valid",
                         [(torch.rand(10,15,34),torch.ones((10,15),dtype=torch.bool))])
def test_model_output(model,input,batch_frames_valid):
    output = model(input,batch_frames_valid)
    assert input.shape[2] == model.d_keypoints
    assert output.shape == torch.Size([input.shape[0],model.nb_actions])
