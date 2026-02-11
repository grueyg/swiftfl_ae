import torch
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.utils import move_to
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE

class SpecTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

        self.reset_hook_in_train("on_batch_forward", "_hook_on_batch_forward_flop_count")
        # self.reset_hook_in_eval("on_batch_forward","_hook_on_batch_forward_regularizer")
 
    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        if 'resnet' in self.cfg.model.type and len(x) == 1:
            x = torch.cat([x,x], dim=0)
            label = torch.cat([label, label], dim=0)
        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

class SpecNLPTrainer(SpecTrainer):

    def _hook_on_batch_forward(self, ctx):
        x, label = [move_to(_, ctx.device) for _ in ctx.data_batch]
        if isinstance(x, dict):
            pred = ctx.model(**x)[0]
        else:
            pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred

        ctx.batch_size = len(label)
