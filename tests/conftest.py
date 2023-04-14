import pytest
from pathlib import Path
from ploomber_engine.ipython import PloomberClient

from models import action_transfomer_mod 

dir_file = Path(__file__).parent


@pytest.fixture(scope="session")
def path_training_setup():
    path = dir_file.joinpath("../notebooks/training_setup.ipynb")
    assert path.exists()
    return path

@pytest.fixture(scope="session")
def full_training_setup(path_training_setup):
    """with dataset of reduced size"""
    client = PloomberClient.from_path(path_training_setup,cwd=path_training_setup.parent,
                                      remove_tagged_cells=["parameters"])
    train_setup = client.get_namespace(dict(truncate = 10,clear_cache = True,location = '/root/.cache/keypoints'))
    return train_setup


@pytest.fixture(scope="session")
def video_kpt_dataset(full_training_setup):
    video_kpt_dataset = full_training_setup["video_kpt_dataset"]
    return video_kpt_dataset

@pytest.fixture(scope="session")
def video_dataloader(full_training_setup):
    video_dataloader = full_training_setup["video_dataloader"]
    return video_dataloader


@pytest.fixture(scope="session")
def model():
    model = action_transfomer_mod.ActionTransformer(nhead=4,num_layers=4)
    return model
