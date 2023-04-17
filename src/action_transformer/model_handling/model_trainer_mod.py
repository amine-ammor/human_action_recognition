import torch




class ModelTrainer:

    def validate_on_batch(self,batch):
        out = self.iterate_on_batch(batch,train_else_val=False)
        return out

    def train_on_batch(self,batch):
        out = self.iterate_on_batch(batch,train_else_val=True)
        return out


def train_on_batch(self):
    out = self.iterate_on_epoch(self.train_data_loader,train_else_val=True)
    return out


@torch.no_grad()
def validate_on_epoch(self):
    out = self.iterate_on_epoch(self.val_data_loader,train_else_val=False)
    return out

def set_learning_rate(self,lr):
    for g in self.optimizer.param_groups:
        g['lr'] = lr



            
    # def lr_range_test(self,min_lr_search,max_lr_search,size_experiments_step,cycle_schedule,method):
        
    #     total_iters_schedule = size_experiments_step//cycle_schedule

    #     self.reset_optimizer()

    #     if method=="exponential":
    #         self.set_learning_rate(min_lr_search)
    #         gamma = np.exp(np.log(max_lr_search/min_lr_search)/total_iters_schedule)
    #         scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=gamma)
    #     elif method == "linear":
    #         self.set_learning_rate(max_lr_search)
    #         start_factor = max_lr_search/min_lr_search
    #         scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,start_factor=start_factor,end_factor=1.0,total_iters = total_iters_schedule)
    #     else:
    #         raise ValueError

    #     # iterate over the train_data_loader, while shuffling at each epoch , for a certain number of iteratiosn
    #     cyclic_shuffled_train_data_loader = (el  for _ in itertools.count(start=0,step=1) for el in train_data_loader)
    #     cyclic_shuffled_train_data_loader = itertools.islice(cyclic_shuffled_train_data_loader,size_experiments_step)


    #     losses_lr = []
    #     lrs = []

    #     for idx,batch in enumerate(cyclic_shuffled_train_data_loader):
    #         print(idx)
    #         loss,nb_words = model_trainer.train_on_batch(batch)
    #         loss = float(loss)
    #         if idx%cycle_schedule==0:
    #             print(idx,scheduler.get_last_lr())
    #             scheduler.step()
    #         print(loss/nb_words,scheduler.get_last_lr())
    #         losses_lr.append(loss/nb_words)
    #         lrs.append(scheduler.get_last_lr())
    #     return losses_lr,