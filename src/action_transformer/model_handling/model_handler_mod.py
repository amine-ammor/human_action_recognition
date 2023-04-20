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
    def __init__(self,model,optimizer,
                 data_loader,
                 loss_batch_fn,metrics,
                 extract_predictions,extract_targets,prepare_inputs,
                 device,train_else_val):
        """_summary_

        Args:
            model (nn.Module): should match the data_loader description
            optimizer (optim.optimizer.Optimizer): pytorch "Optimizer" object 
            data_loader (data.dataDataLoader): pytorch dataloader
            loss_batch_fn (callable): loss function applied on the output of the model on a batch
            metrics : dictionnary of "Metric" object from the torchmetrics framework,

            device (str): device on which the training is done
            train_else_val (bool): if True apply gradient calculation and the optimizer,else


            extract_predictions (callable) : that extracts predictions from the batch and the output of the model
            extract_targets (callable) ; that extracts the targets from the batch and the output of the model
            prepare_inputs (callable) : extracts from the batchs a set the sequence batch that is fed to the model
        """

        self.device = device
        self.model = model

        self.data_loader = data_loader

        self.train_else_val = train_else_val
        self.optimizer = optimizer
        
        # callable to add
        
        self.loss_batch_fn = loss_batch_fn
        self.metrics = metrics

        self.extract_predictions = extract_predictions
        self.extract_targets = extract_targets
        self.prepare_inputs = prepare_inputs



        self.model = self.model.to(self.device)
        for (key,val) in self.metrics.items():
            self.metrics[key] = val.to(device)

        if "loss" in self.metrics:
            self.loss_agregator = self.metrics["loss"]


    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def reset_optimizer(self):
        self.optimizer = self.optimizer.__class__(self.model.parameters(), **self.optimizer.defaults)

    def convert_to_device(self,batch):
        """
            move batch (each value of the dictionnary) to self.device
        """
        for key,val in batch.items():
            batch[key] = val.to(self.device)
        return batch

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

        batch = self.convert_to_device(batch)
        
        inpt_model = self.prepare_inputs(batch)

        out_model = self.model(*inpt_model)


        loss = self.loss_batch_fn(out_model,batch)

        # out_model can be used in some case for the method extract_targets if 
        
        preds = self.extract_predictions(out_model,batch)
        targets = self.extract_targets(out_model,batch) 

        [metric.update(preds,targets) for name,metric in self.metrics.items() if name != "loss"]

        if hasattr(self,"loss_agregator"):
            self.loss_agregator.update(loss,out_model)
        
        if self.train_else_val:
            loss.backward()
            self.optimizer.step()
        
        return loss,out_model

    def get_metrics_vals(self):
        """since beginning of epoch run"""
        metrics_on_epoch = {name:metric.compute() for (name,metric) in self.metrics.items()}
        return metrics_on_epoch
    
    def print_metrics(self):
        for name,metric_val in self.get_metrics_vals().items():
            print(f"value for metric {name} is :  {metric_val}")

    def iterate_on_epoch(self):
        losses = []
        self.reset_metrics()
        for batch_idx,batch in enumerate(self.data_loader):
            loss,out_model = self.iterate_on_batch(batch)
            losses.append(float(loss))
            # updating the streaming metrics

            loss_agregator = self.metrics["loss"]
            loss_agregator.update(loss,out_model)

            #printing metric values once per while
            if batch_idx%self.batch_iter_print == 0:
                self.print_metrics()
            
            # free the memory
            del loss,out_model

        metrics_on_epoch = {name:metric.compute() for (name,metric) in self.metrics.items()}
        return metrics_on_epoch
    
    
