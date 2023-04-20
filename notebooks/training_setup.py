# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=["parameters"]
# parameters

truncate = None
location = '/root/.cache/keypoints'
clear_cache = False

fraction_splits = [0.6,0.4 ]

# + tags=["dataset"]
import joblib

from joblib import Memory
memory = Memory(location,verbose=0)
if clear_cache:
    memory.clear()

# + tags=["dataset"]
from pathlib import Path
from action_transformer import dataset_mod


folder_video_keypoints = "../data/keypoints/as_arrays" 
folder_video_keypoints = str(Path(folder_video_keypoints).resolve())
# with respect to the notebook's directory, all the notebooks are launched with respect to their folder's location
# using papermill or ploomber engine

dataset_mod.VideoKeyPointDataset.memory = memory

video_kpt_dataset = dataset_mod.VideoKeyPointDataset(folder_video_keypoints,with_frame_padding=False,truncate=truncate)


# + tags=["dataset"]
if truncate is not None:
    assert len(video_kpt_dataset) == min(truncate,len(video_kpt_dataset.folder_video_keypoints))
print(len(video_kpt_dataset))

# + tags=["notebook_call"]
# fill the cache and print informations about caching in other notebook
from ploomber_engine.ipython import PloomberClient
path_cache_setup = "./performance_tests/caching_tests_and_metrics.py"
path_cache_setup = str(Path(path_cache_setup).resolve())


import jupytext,time,os

path_cache_setup_as_notebook = path_cache_setup.replace(".py","temp"+str(time.time())+".ipynb")
jupytext.write(nb=jupytext.read(path_cache_setup),
               fp =path_cache_setup_as_notebook)


client = PloomberClient.from_path(path_cache_setup_as_notebook,remove_tagged_cells="notebook_call")
namespace = client.get_namespace(dict(video_kpt_dataset=video_kpt_dataset,clear_cache=clear_cache,location=location))

if os.path.exists(path_cache_setup_as_notebook):
    os.remove(path_cache_setup_as_notebook)

# + tags=["dataset"]
import numpy as np
all_labels = list(np.unique([el["label"] for el in video_kpt_dataset]))

label_to_idx = {label:idx for (idx,label) in enumerate(all_labels)}

transform_labels = lambda label : label_to_idx[label]
video_kpt_dataset.with_frame_padding = True
video_kpt_dataset.transform_labels = transform_labels


# + tags=["dataset"]
num_to_sample =  {i:video_kpt_dataset[i]["label"] for i in range(len(video_kpt_dataset))}
all_labels = np.unique(list(num_to_sample.values()))
label_to_samples_idx = {el:[] for el in all_labels}
for idx,label  in num_to_sample.items():
    label_to_samples_idx[label].append(idx)
    
import random,torch
num_splits = len(fraction_splits)


splits = [[] for _ in range(num_splits) ]
for idx_split,split in enumerate(splits[:-1]):
    for label in all_labels:
        tmp = label_to_samples_idx[label]
        random.shuffle(tmp)
        num_samples = int(fraction_splits[idx_split]*len(tmp))
        split += tmp[:num_samples] 

splits[-1] = list(set(num_to_sample.keys()).difference(
    set([el for split in splits[:-1] for el in split])))


for idx_split in range(num_splits-1):
    assert np.abs(len(splits[idx_split])- fraction_splits[idx_split]*len(num_to_sample)) <1.0
    res = np.unique([num_to_sample[idx] for idx in splits[idx_split]],return_counts=True)[1]
    assert  np.max(np.abs(np.mean(res) - res)) <1.0


split_dataset = True
if split_dataset:
    train_dataset,val_dataset = [torch.utils.data.Subset(video_kpt_dataset,split) for split in splits]

# + tags=["dataloader", "parameters"]
batch_size = 32


# + tags=["dataloader"]
import torch
from collections import OrderedDict

def collate_fn(batch):
    assert all([type(el) == OrderedDict for el in  batch])

    batch = {k: [dic[k] for dic in batch] for k in batch[0]}
    nb_frames,kpts,is_detection_present,labels = batch.values()
    assert list(batch.keys()) == ['number_of_frames', 'kpts', 'is_detection_present', 'label']
    
    max_nb_frames = max(batch.pop("number_of_frames")) 
    # we compute the maximum over the batch to remove the padded frames from the batch
    # for some extra computation savings
    kpts = torch.tensor(kpts)
    kpts = kpts.reshape(kpts.shape[0],kpts.shape[1],-1)    
    kpts = kpts[:,:max_nb_frames]
    
    is_detection_present = torch.tensor(is_detection_present)
    is_detection_present = is_detection_present[:,:max_nb_frames]
    #nb_frames = np.array(nb_frames)
    labels = torch.tensor(labels,dtype=torch.long)
    
    batch["kpts"] = kpts
    batch["is_detection_present"] = is_detection_present
    batch["label"] = labels
    return batch

from torch.utils.data import DataLoader

train_video_dataloader,val_video_dataloader = [ DataLoader(dset,batch_size=batch_size,collate_fn=collate_fn,shuffle=True,
                             pin_memory=True) for dset in [train_dataset,val_dataset]
                         ]

# + tags=["dataloader"]
for el in train_video_dataloader:
    break
el["kpts"].shape,el["is_detection_present"].shape,el['label'].shape

# + tags=["dataloader"]
import torch
assert type(train_video_dataloader.sampler) == torch.utils.data.sampler.RandomSampler

# + tags=["dataloader"]
#sanity check test
#res[0][0].numpy()
#video_kpt_dataset[0][0]#res[0][0].numpy()
#video_kpt_dataset[0][0]#res[0][0].numpy()
#video_kpt_dataset[0][0]
unshuffled = False
if unshuffled:
    loader_iter = iter(d_loader)
    res = next(loader_iter)
    assert np.all(res[0][0][6].numpy() == video_kpt_dataset[0][0][6])
# -

from action_transformer.models import action_transfomer_mod
nb_actions=len(label_to_idx)
model = action_transfomer_mod.ActionTransformer(1,2,nb_actions=nb_actions)

from torch import nn,optim
from torch.optim import lr_scheduler
optimizer = optim.AdamW(model.parameters(),lr=lr)

from torchmetrics import Metric
import torch
class LossAgregator(Metric):
    """agregate loss over multiple batches"""
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0,dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0,dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, loss,out_model):
        self.loss += loss
        self.total += out_model.shape[0]

    def compute(self):
        return self.loss.float() / self.total


# +
loss_fn = nn.CrossEntropyLoss()
def loss_batch_fn(out_model,batch):
    res = loss_fn(out_model,batch["label"])
    return res
def extract_predictions(out_model,batch):
    res = out_model.argmax(1)
    return res

def extract_targets(out_model,batch):
    res = batch["label"]
    return res

def prepare_inputs(batch):
    res = batch["kpts"],batch["is_detection_present"]
    return res



# -

import torchmetrics
accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=model.nb_actions)
metrics = {"accuracy":accuracy_metric,
           "loss": LossAgregator()}

# +
from action_transformer.model_handling import model_handler_mod
import torch
import copy
device = "cuda" if torch.cuda.is_available() else "cpu" 
trainer = model_handler_mod.ModelHandler(model,optimizer,
                                              train_video_dataloader,
                                               loss_batch_fn=loss_batch_fn,
                                              metrics=copy.deepcopy(metrics),
                                              extract_predictions=extract_predictions,
                                              extract_targets=extract_targets,
                                              prepare_inputs=prepare_inputs,
                                              device=device,
                                              train_else_val=True)

validator = model_handler_mod.ModelHandler(model,None,
                                              val_video_dataloader,
                                               loss_batch_fn=loss_batch_fn,
                                              metrics=copy.deepcopy(metrics),
                                              extract_predictions=extract_predictions,
                                              extract_targets=extract_targets,
                                              prepare_inputs=prepare_inputs,
                                              device=device,
                                              train_else_val=False)

