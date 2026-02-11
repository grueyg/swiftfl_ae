import torch
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.spec.exp.fjord.od.samplers import ODSampler
from federatedscope.core.trainers.utils import move_to

class FjORDTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

        self.reset_hook_in_train("on_batch_forward", "_hook_on_batch_forward_flop_count")
        # self.reset_hook_in_eval("on_batch_forward","_hook_on_batch_forward_regularizer")

    def init_p(self, p_s, max_p):
        self.p_s = p_s
        self.max_p = max_p

    # def init_sampler(self):
    #     self.sampler = ODSampler(p_s=self.p_s, max_p=self.max_p, model=self.ctx.model)
    #     self.max_sampler = ODSampler(p_s=[self.max_p], max_p=self.max_p, model=self.ctx.model)

    def _hook_on_batch_forward(self, ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        if 'resnet' in self.cfg.model.type and len(x) == 1:
            x = torch.cat([x,x], dim=0)
            label = torch.cat([label, label], dim=0)

        if self.cfg.fjord.know_distill and ctx.cur_mode == MODE.TRAIN:
            full_output = ctx.model(x, sampler=ctx.max_sampler)
            full_loss = ctx.criterion(full_output, label)
            full_loss.backward()
            label = full_output.detach().softmax(dim=1)
        
        pred = ctx.model(x, sampler=ctx.sampler)

        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(ctx.criterion(pred, label), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)

    @lifecycle(LIFECYCLE.ROUTINE)
    def _run_routine(self, mode, hooks_set, dataset_name=None):
        self.ctx.sampler = CtxVar(ODSampler(p_s=self.p_s, max_p=self.max_p, model=self.ctx.model),
                                  LIFECYCLE.ROUTINE)
        self.ctx.max_sampler = CtxVar(ODSampler(p_s=[self.max_p], max_p=self.max_p, model=self.ctx.model),
                                      LIFECYCLE.ROUTINE)

        for hook in hooks_set["on_fit_start"]:
            hook(self.ctx)

        self._run_epoch(hooks_set)

        for hook in hooks_set["on_fit_end"]:
            hook(self.ctx)

        return self.ctx.num_samples
    

class FjORDNLPTrainer(FjORDTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

    def _hook_on_batch_forward(self, ctx):
        x, label = [move_to(_, ctx.device) for _ in ctx.data_batch]

        if self.cfg.fjord.know_distill and ctx.cur_mode == MODE.TRAIN:
            if isinstance(x, dict):
                full_output = ctx.model(**x, sampler=ctx.max_sampler)[0]
            else:
                full_output = ctx.model(**x)
            full_loss = ctx.criterion(full_output, label)
            full_loss.backward()
            label = full_output.detach().softmax(dim=1)
        
        pred = ctx.model(**x, sampler=ctx.sampler)

        if isinstance(x, dict):
            pred = ctx.model(**x, sampler=ctx.sampler)[0]
        else:
            pred = ctx.model(x, sampler=ctx.sampler)

        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred

        ctx.batch_size = len(label)

    

