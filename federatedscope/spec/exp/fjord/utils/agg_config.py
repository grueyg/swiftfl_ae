"""Flower client implementing FjORD."""
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from federatedscope.spec.exp.fjord.od.layers import ODBatchNorm2d, ODConv2d, ODLinear
from federatedscope.spec.exp.fjord.od.samplers import ODSampler

FJORD_CONFIG_TYPE = Dict[
    Union[str, float],
    List[Any],
]

def get_layer_from_state_dict(model: Module, state_dict_key: str) -> Module:
    """Get the layer corresponding to the given state dict key.

    :param model: The model.
    :param state_dict_key: The state dict key.
    :return: The module corresponding to the given state dict key.
    """
    keys = state_dict_key.split(".")
    module = model
    # The last keyc orresponds to the parameter name
    # (e.g., weight or bias)
    for key in keys[:-1]:
        module = getattr(module, key)
    return module


def net_to_state_dict_layers(net: Module) -> List[Module]:
    """Get the state_dict of the model.

    :param net: The model.
    :return: The state_dict of the model.
    """
    layers = []
    for key, _ in net.state_dict().items():
        layer = get_layer_from_state_dict(net, key)
        layers.append(layer)
    return layers


def get_agg_config(
    net: Module, local_data: Tensor, p_s: List[float]
) -> FJORD_CONFIG_TYPE:
    """Get the aggregation configuration of the model.

    :param net: The model.
    :param trainloader: The training set.
    :param p_s: The p values used
    :return: The aggregation configuration of the model.
    """

    device = next(net.parameters()).device
    local_data = local_data.to(device)
    layers = net_to_state_dict_layers(net)
    # init min dims in networks
    config: FJORD_CONFIG_TYPE = {p: [{} for _ in layers] for p in p_s}
    config["layer"] = []
    config["layer_p"] = []
    with torch.no_grad():
        for p in p_s:
            max_sampler = ODSampler(
                p_s=[p],
                max_p=p,
                model=net,
            )
            net(local_data, sampler=max_sampler)
            for i, layer in enumerate(layers):
                if isinstance(layer, (ODConv2d, ODLinear)):
                    config[p][i]["in_dim"] = layer.last_input_dim
                    config[p][i]["out_dim"] = layer.last_output_dim
                elif isinstance(layer, ODBatchNorm2d):
                    config[p][i]["in_dim"] = None
                    config[p][i]["out_dim"] = layer.p_to_num_features[p]
                elif isinstance(layer, torch.nn.BatchNorm2d):
                    pass
                else:
                    raise ValueError(f"Unsupported layer {layer.__class__.__name__}")
    for layer in layers:
        config["layer"].append(layer.__class__.__name__)
        if hasattr(layer, "p"):
            config["layer_p"].append(layer.p)
        else:
            config["layer_p"].append(None)
    return config

