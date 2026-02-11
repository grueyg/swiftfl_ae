import torch
import copy
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler

class SpecUpdater():
    def __init__(self, model, device, config) -> None:
        self.model = copy.deepcopy(model)
        self.device = device
        self.cfg = config
        self.optimizer = get_optimizer(self.model, **self.cfg.train.optimizer)
        self.scheduler = get_scheduler(self.optimizer, **self.cfg.train.scheduler)
    
    def update_by_aggregated_gradient(self, model_state_dict, gradient, strict=False):
        self.model.load_state_dict(copy.deepcopy(model_state_dict), strict=strict)
        for name, parameter in self.model.named_parameters():
            # parameter.grad = gradient[name].to(parameter.device).type(parameter.grad.dtype)
            if name in gradient:
                parameter.grad = gradient[name].to(parameter.device).type(parameter.dtype)
            else:
                parameter.grad = torch.zeros_like(parameter).to(parameter.device).type(parameter.dtype)
        self.optimizer.step()
        return self.get_model_para()

    def get_model_para(self):
        if self.cfg.federate.process_num > 1:
            return self.model.state_dict()
        else:
            return self.model.state_dict() if self.cfg.federate.share_local_model \
                else self.model.cpu().state_dict()
