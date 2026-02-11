import copy
import torch
import numpy as np
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor
import numpy as np
from federatedscope.spec.exp.fjord.utils.agg_config import FJORD_CONFIG_TYPE
from federatedscope.core.aggregators import ClientsAvgAggregator

import logging

logger = logging.getLogger(__name__)

class FjORDAggregator(ClientsAvgAggregator):
    def init_fjord_config(self, p_s, fjord_config):
        self.p_s = p_s
        self.fjord_config = fjord_config
        
    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model
    
    def _para_weighted_avg(self, models, recover_fun=None):

        p_max_s = []
        num_examples = []
        for i in range(len(models)):
            sample_size, (_, p_max) = models[i]
            p_max_s.append(p_max)
            num_examples.append(sample_size)
        _, (avg_model, _) = models[0]
        original_model = self.model.state_dict()

        for i,key in enumerate(avg_model.keys()):
            layer_updates = [model[key] * sample_size for sample_size, (model, _) in models]
            avg_model[key] = fjord_average(i,
                                           layer_updates, 
                                           num_examples,
                                           p_max_s, 
                                           self.p_s,
                                           self.fjord_config, 
                                           original_model[key])

        return avg_model


def get_p_layer_updates(
    p: float,
    layer_updates: List[Tensor],
    num_examples: List[int],
    p_max_s: List[float],
) -> Tuple[List[Tensor], int]:
    """Get layer updates for given p width.

    :param p: p-value
    :param layer_updates: list of layer updates from clients
    :param num_examples: list of number of examples from clients
    :param p_max_s: list of p_max values from clients
    """
    # get layers that were updated for given p
    # i.e., for the clients with p_max >= p
    layer_updates_p = [
        layer_update
        for p_max, layer_update in zip(p_max_s, layer_updates)
        if p_max >= p
    ]
    num_examples_p = sum(n for p_max, n in zip(p_max_s, num_examples) if p_max >= p)
    return layer_updates_p, num_examples_p


def fjord_average(  # pylint: disable=too-many-arguments
    i: int,
    layer_updates: List[Tensor],
    num_examples: List[int],
    p_max_s: List[float],
    p_s: List[float],
    fjord_config: FJORD_CONFIG_TYPE,
    layer_original_parameters: List[Tensor],
) -> Tensor:
    """Compute average per layer for given updates.

    :param i: index of the layer
    :param layer_updates: list of layer updates from clients
    :param num_examples: list of number of examples from clients
    :param p_max_s: list of p_max values from clients
    :param p_s: list of p values
    :param fjord_config: fjord config
    :param original_parameters: original model parameters
    :return: average of layer
    """
    # if no client updated the given part of the model,
    # reuse previous parameters
    update = deepcopy(layer_original_parameters)

    # BatchNorm2d layers, only average over the p_max_s
    # that are greater than corresponding p of the layer
    # i.e., only update the layers that were updated
    if fjord_config["layer_p"][i] is not None:
        p = fjord_config["layer_p"][i]
        layer_updates_p, num_examples_p = get_p_layer_updates(
            p, layer_updates, num_examples, p_max_s
        )
        if len(layer_updates_p) == 0:
            return update

        assert num_examples_p > 0
        return reduce(torch.add, layer_updates_p) / num_examples_p
    if fjord_config["layer"][i] in ["ODLinear", "ODConv2d", "ODBatchNorm2d"]:
        # perform nested updates
        for p in p_s[::-1]:
            layer_updates_p, num_examples_p = get_p_layer_updates(
                p, layer_updates, num_examples, p_max_s
            )
            if len(layer_updates_p) == 0:
                continue
            in_dim = (
                int(fjord_config[p][i]["in_dim"])
                if fjord_config[p][i]["in_dim"]
                else None
            )
            out_dim = (
                int(fjord_config[p][i]["out_dim"])
                if fjord_config[p][i]["out_dim"]
                else None
            )
            assert num_examples_p > 0
            # check whether the parameter to update is bias or weight
            if len(update.shape) == 1:
                # bias or ODBatchNorm2d
                layer_updates_p = [
                    layer_update[:out_dim] for layer_update in layer_updates_p
                ]
                update[:out_dim] = reduce(torch.add, layer_updates_p) / num_examples_p
            else:
                # weight
                layer_updates_p = [
                    layer_update[:out_dim, :in_dim] for layer_update in layer_updates_p
                ]
                update[:out_dim, :in_dim] = (
                    reduce(torch.add, layer_updates_p) / num_examples_p
                )
        return update

    raise ValueError(f"Unsupported layer {fjord_config['layer'][i]}")

