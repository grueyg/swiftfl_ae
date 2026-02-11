import copy
import torch
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
import logging

logger = logging.getLogger(__name__)

class FedNovaAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model, device, config)

        self.global_momentum_buffer = dict()
        
    def aggregate(self, agg_info, tau_eff=0):
        params = copy.deepcopy(self.model.cpu().state_dict())
        sample_sizes = []
        norm_grads = []
        tau_effs = []
        for sample_size, norm_grad, tau in agg_info['client_feedback']:
            sample_sizes.append(sample_size)
        sample_size_total = sum(sample_sizes)

        for sample_size, norm_grad, tau in agg_info['client_feedback']:
            norm_grads.append(norm_grad)
            tau_effs.append(tau * sample_size / sample_size_total)
            
        # get tau_eff
        tau_eff = sum(tau_effs)
        # get cum grad
        # cum_grad = tau_eff * sum(norm_grads)
        cum_grad = norm_grads[0]
        for k in norm_grads[0].keys():
            for i in range(0, len(norm_grads)):
                if i == 0:
                    cum_grad[k] = norm_grads[i][k] * tau_eff * (sample_sizes[i] / sample_size_total)
                else:
                    cum_grad[k] += norm_grads[i][k] * tau_eff * (sample_sizes[i] / sample_size_total)
        # update params
        for k in params.keys():
            if 'tracked' in k:
                continue
            if self.cfg.train.optimizer.gmf != 0:
                if k not in self.global_momentum_buffer:
                    buf = self.global_momentum_buffer[k] = torch.clone(
                        cum_grad[k]
                    ).detach()
                    buf.div_(self.cfg.train.optimizer.learning_rate)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.cfg.train.optimizer.gmf).add_(1 / self.cfg.train.optimizer.learning_rate, cum_grad[k])
                params[k].sub_(self.cfg.train.optimizer.learning_rate, buf)
            else:
                params[k].sub_(cum_grad[k])

        return params