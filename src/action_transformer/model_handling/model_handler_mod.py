from itertools import islice
from torch import optim
import torch,itertools
import numpy as np
import time
from . import utils

import abc
#TODO: add integration tests to test the adequancy between the model trainer and the various elements
#of 

class ModelHandler:
    batch_iter_print = 10
    def __init__(self,model,optimizer,scheduler,
                 data_loader,
                 loss_batch_fn,
                 metrics,
                 extract_predictions,extract_targets,prepare_inputs,
                 device,device_converter,train_else_val):
        """_summary_

        Args:
            model (nn.Module): should match the data_loader description
            optimizer (optim.optimizer.Optimizer): pytorch "Optimizer" object 
            scheduler (optim.lr_scheduler.LRScheduler): pytorch "Scheduler" object 
            data_loader (data.dataDataLoader): pytorch dataloader
            loss_batch_fn (callable): loss function applied on the output of the model on a batch
            metrics : dictionnary of "Metric" object from the torchmetrics framework,

            device_converter (callable): that convert a batch of data from one device to the other
            (Remark, not necessarily all the data must be moved to the device, in order for the model
            to be applied)

            device (str): device on which the training is done
            train_else_val (bool): if True apply gradient calculation and the optimizer,else


            extract_predictions (callable) : that extracts predictions from the batch and the output of the model
            extract_targets (callable) ; that extracts the targets from the batch and the output of the model
            prepare_inputs (callable) : extracts from the batchs a set the sequence batch that is fed to the model
        """

        self.device = device
        self.model = model.to(self.device)

        self.data_loader = data_loader

        self.train_else_val = train_else_val
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # callable to add
        
        self.loss_batch_fn = loss_batch_fn
        self.metrics = metrics

        self.device_converter = device_converter
        self.extract_predictions = extract_predictions
        self.extract_targets = extract_targets
        self.prepare_inputs = prepare_inputs



    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def reset_optimizer(self):
        self.optimizer = self.optimizer.__class__(self.model.parameters(), **self.optimizer.defaults)


    def iterate_on_batch(self,batch):
        """_summary_

        Args:
            batch (Iterable[torch.Tensor]): iterable of tensors , such that the first dimension
             (of each tensor) indexes instances on batch, it should match the forward method 
             of the model otherwise disable it and just keep the output of the model 

        Returns:
           loss,out_model : returns output of the model,
        """
        if self.train_else_val:
            self.optimizer.zero_grad()
        else:
            assert self.optimizer is None

        batch = self.device_converter(batch,self.device)
        inpt_model = self.prepare_inputs(batch)

        out_model = self.model(*inpt_model)


        loss = self.loss_batch_fn(out_model,batch)

        preds = self.extract_predictions(out_model,batch)
        targets = self.extract_targets(out_model,batch) 
        # out_model can be used in some case for the method extract_targets if 

        [metric.update(preds,targets) for metric in self.metrics.values()]
        
        if self.train_else_val:
            loss.backward()
            self.optimizer.step()
        
        return loss,out_model

    
    def iterate_on_epoch(self,data_loader,train_else_val):
        losses = []
        outs_model = []
        self.reset_metrics()
        for batch_idx,batch in enumerate(data_loader):
            loss,out_model = self.iterate_on_batch(batch,batch_idx,train_else_val=train_else_val)
            losses.append(loss)
            outs_model.append(out_model)
            if batch_idx%self.batch_iter_print == 0:
                for name,metric in self.metrics.items():
                    print(f"value for metric {name} is :  {metric.compute()}")

        metrics_on_epoch = {name:metric.compute() for (name,metric) in self.metrics.items()}
        return metrics_on_epoch
    
    
