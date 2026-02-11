import copy
import torch
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.decorators import use_diff
from federatedscope.core.trainers.enums import MODE, LIFECYCLE

class ScaffoldTrainer(GeneralTorchTrainer):
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)

        self.reset_hook_in_train("on_batch_forward", "_hook_on_batch_forward_flop_count")
        # self.reset_hook_in_eval("on_batch_forward","_hook_on_batch_forward_regularizer")

        self.client_c = []
        for param in self.ctx.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

    def update_c(self, global_c):
        self.global_c = global_c
        self.global_model = copy.deepcopy(self.ctx.model)

    def update_yc(self):
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.ctx.model.parameters()):
            ci.data = ci - c + 1/self.ctx.num_train_batch/self.ctx.num_train_epoch/self.cfg.train.optimizer.lr * (x - yi)

    def delta_yc(self):
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.ctx.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1/self.ctx.num_train_batch/self.ctx.num_train_epoch/self.cfg.train.optimizer.lr * (x - yi))

        return delta_y, delta_c
    
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

    def _hook_on_batch_backward(self, ctx):
        """
        Note:
          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.optimizer``                   Update by gradient
            ``ctx.loss_task``                   Backward propagation
            ``ctx.scheduler``                   Update by gradient
            ==================================  ===========================
        """
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)

        ctx.optimizer.step(self.global_c, self.client_c)
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    # @use_diff
    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)
        self.update_yc()

        dy, dc = self.delta_yc()

        return num_samples, self.ctx.eval_metrics, dy, dc