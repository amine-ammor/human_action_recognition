batch_size = 32
lr = 10**(-4)
n_epochs = 30
fraction_splits = [0.6,0.4 ]

# +
# fill the cache and print informations about caching in other notebook
from pathlib import Path
from ploomber_engine.ipython import PloomberClient

path_training_setup = "./training_setup.py"

import utils_notebook
import psutil
from tqdm import tqdm
with utils_notebook.NotebookPyExecutor(path_training_setup,remove_tagged_cells="parameters",
                                      truncate = None,location = '/root/.cache/keypoints',
                                       clear_cache = False,batch_size=batch_size,
                                       fraction_splits = fraction_splits,lr=lr) as nb_exec:
    trainer = nb_exec.namespace["trainer"]
    validator = nb_exec.namespace["validator"]

    train_video_dataloader = nb_exec.namespace["train_video_dataloader"]
    val_video_dataloader = nb_exec.namespace["val_video_dataloader"]
    print(nb_exec.path_training_setup_as_notebook)
    assert Path(nb_exec.path_training_setup_as_notebook).exists()


# +
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
print(writer.log_dir)
# -

import torch
trainer.model.load_state_dict(torch.load("tmp_model.pkl"))

# +
from tqdm import tqdm

from utils_notebook import print_memory

for i,epoch in tqdm(enumerate(range(n_epochs))):
    

    print_memory() # to check if there are no memory leak

    print("training")
    train_metrics_vals = trainer.iterate_on_epoch()
    print("validation")
    val_metrics_vals = validator.iterate_on_epoch()
    

    writer.add_scalar("Loss/train",float(train_metrics_vals["loss"]),i)
    writer.add_scalar("Accuracy/train",float(train_metrics_vals["accuracy"]),i)
    
    writer.add_scalar("Loss/val",float(val_metrics_vals["loss"]),i)
    writer.add_scalar("Accuracy/val",float(val_metrics_vals["accuracy"]),i)
# -


