"""Utility functions for fjord."""
import os
from typing import List, Optional, OrderedDict

import numpy as np
import torch
from torch.nn import Module


def get_parameters(net: Module) -> List[np.ndarray]:
    """Get statedict parameters as a list of numpy arrays.

    :param net: PyTorch model
    :return: List of numpy arrays
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: Module, parameters: List[np.ndarray]) -> None:
    """Load parameters into PyTorch model.

    :param net: PyTorch model
    :param parameters: List of numpy arrays
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

