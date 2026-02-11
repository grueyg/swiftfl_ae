import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from federatedscope.register import register_optimizer

class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, **kwargs):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                sc = sc.to(p.device)
                cc = cc.to(p.device)
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])

