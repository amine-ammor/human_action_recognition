import pytest

from models import action_transfomer_mod 
@pytest.fixture()
def model():
    model = action_transfomer_mod.ActionTransformer(nhead=4,num_layers=4)
    return model

