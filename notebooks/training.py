# +
# fill the cache and print informations about caching in other notebook
from pathlib import Path
from ploomber_engine.ipython import PloomberClient

path_cache_setup = "./training_setup.ipynb"
path_cache_setup = str(Path(path_cache_setup).resolve())

client = PloomberClient.from_path(path_cache_setup,remove_tagged_cells="parameters")
namespace = client.get_namespace(
    dict(truncate = None,
         location = '/root/.cache/keypoints',
         clear_cache = False))
# -
video_dataloader = namespace["video_dataloader"]
label_to_idx = namespace["label_to_idx"]

nb_actions=len(label_to_idx)

from action_transformer.models import action_transfomer_mod
model = action_transfomer_mod.ActionTransformer(1,2,nb_actions=nb_actions)

# +
from torch import nn

from torch import optim
from torch.optim import lr_scheduler
optimizer = optim.AdamW(model.parameters())
scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=10**(-3),epochs=10,steps_per_epoch=len(video_dataloader))
# -

for batch in video_dataloader:
    out_model = model(*batch[:2])
    break

loss_fn = nn.CrossEntropyLoss()
def loss_batch_fn(out_model,batch):
    res = loss_fn(out_model,batch[-1])
    return res


out_model.shape,[el.shape for el in batch]

import torchmetrics
accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=model.nb_actions)
metrics = {"accuracy":accuracy_metric}

import torch
batch[-1],out_model.argmax(1)


def extract_predictions(out_model,batch):
    


from action_transformer.model_handling import model_handler_mod
model_handler = model_handler_mod.ModelHandler(model,optimizer,scheduler,
                                              video_dataloader,loss_batch_fn=loss_batch_fn,
                                              metrics=metrics,
                                              extract_predictions=extract_predictions)


for el in ["video_dataloader"]:
    globals()[el] = namespace[el]

batch,batch_frames_valide = next(iter(video_dataloader))
