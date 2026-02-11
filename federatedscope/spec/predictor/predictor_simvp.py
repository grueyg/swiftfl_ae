
import torch
import numpy as np
from torch.multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader,  Subset
from federatedscope.spec.predictor import BasePredictor
from federatedscope.spec.predictor import PredictorDataset
from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

class GradPredictor(BasePredictor):
    def __init__(self, model, config, send_channel, recv_channel,  data_channel):
        super().__init__(model, config)
        self.send_channel = send_channel
        self.recv_channel = recv_channel
        self.data_channel = data_channel
        self.server_device = config.device

        self.cfg = config.spec.predictor
        self.criterion = get_criterion(**self.cfg.criterion[0])
        self.optimizer = get_optimizer(model=model, **self.cfg.optimizer[0])
        self.scheduler = get_scheduler(optimizer=self.optimizer, **self.cfg.scheduler[0])

        self.device = self.cfg.device
        self.pre_seq_length = self.cfg.pre_seq_length // self.cfg.sum_seq_length
        self.aft_seq_length = self.cfg.aft_seq_length // self.cfg.sum_seq_length
        self.epochs = self.cfg.epochs
        self.do_validation = self.cfg.do_validation
        self.train_size = self.cfg.train_size

        self.metrics = Metrics()
        self.init_shared_variable()

    def init_shared_variable(self):
        # self.model.share_memory()
        self.start_epoch = torch.multiprocessing.Value('i', 0)
        
        manager = Manager()
        self.optimizer_state_dict = manager.dict()
        self.scheduler_state_dict = manager.dict()
        self.update_shared_variable()

    def load_shared_variable(self):
        self.optimizer.load_state_dict(dict(self.optimizer_state_dict))
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.scheduler.load_state_dict(dict(self.scheduler_state_dict))
        
    def update_shared_variable(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        self.optimizer_state_dict.update(self.optimizer.state_dict())
        self.scheduler_state_dict.update(self.scheduler.state_dict())
        
    def get_dataloader(self, data):
        dataset = PredictorDataset(data, pre_seq_length=self.pre_seq_length)
        if not self.do_validation:
            return DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False), None
        else:
            index = np.random.permutation(np.arange(len(dataset)))
            train_size = int(self.train_size * len(dataset))
            if isinstance(dataset, Dataset):
                train_dataset = Subset(dataset, index[:train_size])
                val_dataset = Subset(dataset, index[train_size:])
            return DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False), \
                   DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=False)
        
    def train(self):
        self.logger = self._setup_logger(self.cfg.type)
        while True:
            if not self.data_channel.empty():
                msg = self.data_channel.get()
                if msg == 'finish_FL':
                    break
                else:
                    self.dataloader, self.valid_dataloader = self.get_dataloader(msg)
                    self.model.to(self.device)
                    self.metrics.reset()
                    for _ in range(self.epochs):
                        self.start_epoch.value += 1
                        train_loss, valid_loss, msg = self._train_epoch()
                        self.metrics.update(train_loss, valid_loss)
                        result = {'epoch': self.start_epoch.value,
                                  'train_loss': train_loss,
                                  'valid_loss': valid_loss}
                        self.logger.info(result)
                        if msg == 'finish':
                            break
                    self.send_channel.put((self.metrics.msg(), self.model.cpu().state_dict()))

    def _train_epoch(self):
        self.model.train()
        train_loss = 0
        msg = None
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data, target = data.to(self.device), target.to(self.device)
            if self.aft_seq_length == self.pre_seq_length:
                output = self.model(data)
            elif self.aft_seq_length < self.pre_seq_length:
                output = self.model(data)
                output = output[:, :self.aft_seq_length]
            if self.cfg.criterion[0]['criterion_type'] == 'CosineEmbeddingLoss':
                loss = self.criterion(output, target, torch.ones(len(output)))
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            if not self.recv_channel.empty():
                msg = self.recv_channel.get()
                if msg == 'finish':
                    break
        train_loss /= (batch_idx + 1)
        if self.do_validation:
            valid_loss = self._valid_epoch()
        return train_loss, valid_loss, msg
    
    def _valid_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                if self.aft_seq_length == self.pre_seq_length:
                    output = self.model(data)
                elif self.aft_seq_length < self.pre_seq_length:
                    output = self.model(data)
                    output = output[:, :self.aft_seq_length]
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss/ (batch_idx + 1)

    def predict(self, data):
        self.model.eval()
        self.model.to(self.server_device)
        result = {}
        with torch.no_grad():
            for name, grad in data.items():
                grad = grad.to(self.server_device)
                result[name] = torch.sum(self.model(grad)[:, :self.aft_seq_length], dim=1).cpu()
        return result
        
class Metrics(object):

    def __init__(self):
        self.sum_train_loss = 0
        self.sum_valid_loss = 0
        self.epochs = 0

    def reset(self):
        self.sum_train_loss = 0
        self.sum_valid_loss = 0
        self.epochs = 0

    def update(self, train_loss, valid_loss):
        self.sum_train_loss += train_loss
        self.sum_valid_loss += valid_loss
        self.epochs += 1

    def msg(self):
        mean_train_loss = self.sum_train_loss / self.epochs
        mean_valid_loss = self.sum_valid_loss / self.epochs
        msg = {'Predictor': 'default',
               'epoch_num': self.epochs, 
               'train_loss': mean_train_loss, 
               'valid_loss': mean_valid_loss}
        return msg