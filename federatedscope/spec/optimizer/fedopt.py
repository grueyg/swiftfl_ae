from copy import deepcopy
from collections import OrderedDict

import torch
from torch.nn import Module

from collections import Counter, OrderedDict
from typing import List, Tuple, Union

def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        List of parameters [, names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters


class AdaptiveOptimizer:
    def __init__(
        self,
        type: str,
        model: Module,
        beta1: float,
        beta2: float,
        lr: float,
        tau: float,
    ):
        self.update = { 
            "adagrad": self._update_adagrad,
            "yogi": self._update_yogi,
            "adam": self._update_adam,
        }[type]
        self.model = model
        self.lr = lr
        self.tau = tau
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentums = [
            torch.zeros_like(param) for param in trainable_params(self.model)
        ]
        self.velocities = deepcopy(self.momentums)
        self.delta_list: List[torch.Tensor] = None

    @torch.no_grad()
    def step(
        self, delta_cache: List[OrderedDict[str, torch.Tensor]], weights: torch.Tensor
    ):
        # compute weighted delta
        list_delta_cache = [
            [-diff for diff in delta_dict.values()] for delta_dict in delta_cache
        ]
        delta_list = []
        for delta in zip(*list_delta_cache):
            delta_list.append(torch.sum(torch.stack(delta, dim=-1) * weights, dim=-1))

        # update momentums
        for m, delta in zip(self.momentums, delta_list):
            m.data = self.beta1 * m + (1 - self.beta1) * delta

        # update velocities according to different rules
        self.update(delta_list)

        # update model parameters
        for param, m, v in zip(
            trainable_params(self.model), self.momentums, self.velocities
        ):
            param.data = param.data + self.lr * (m / (v.sqrt() + self.tau))

    def _update_adagrad(self, delta_list):
        for v, delta in zip(self.velocities, delta_list):
            v.data = v + delta**2

    def _update_yogi(self, delta_list):
        for v, delta in zip(self.velocities, delta_list):
            delta_pow2 = delta**2
            v.data = v - (1 - self.beta2) * delta_pow2 * torch.sign(v - delta_pow2)

    def _update_adam(self, delta_list):
        for v, delta in zip(self.velocities, delta_list):
            v.data = self.beta2 * v + (1 - self.beta2) * delta**2