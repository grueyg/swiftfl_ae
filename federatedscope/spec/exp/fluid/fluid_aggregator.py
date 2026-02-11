import copy
import torch
import numpy as np
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

from collections import OrderedDict
import numpy as np
from federatedscope.spec.exp.fjord.utils.agg_config import FJORD_CONFIG_TYPE
from federatedscope.core.aggregators import ClientsAvgAggregator

import logging

logger = logging.getLogger(__name__)

class FLuIDggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model, device, config)

        self.trainable_para_name = list(dict(list(model.named_parameters())).keys())
        self.origWeightsShape = [p.shape for p in model.parameters()]

    def aggregate(self, agg_info):
        results = agg_info["client_feedback"]
        droppedWeights = agg_info["droppedWeights"]
        avg_model = self.aggregate_drop(results, droppedWeights)

        return avg_model
    
    def aggregate_drop(self,
                       results: List[Tuple[int, Tuple[Dict,str]]],
                       dropWeights: Dict[str, List]) -> Dict:
        """Compute weighted average for a federated drop technique """

        # initialize list to keep track of the total number of examples used during training for each neuron
        # since we are dropping neurons from the model for some clients, so the num examples that each neuron
        # trained on will be different
        num_examples_total = sum([num_examples for _, num_examples, _ in results])
        total_examples_wDrop = []
        for i in range(len(self.origWeightsShape)):
            total_examples_wDrop.append(
                torch.full(self.origWeightsShape[i], num_examples_total))

        # transform the list of weights into original format
        # We will expand sub-models to the global model shape by filling in 0s
        # for dropped weights
        transformedResults = []
        for (cweights, num_examples, cid) in results:
            transformedWeights = [cweights[key] for key in self.trainable_para_name]

            # no transformation needed if not a straggler
            if cid not in dropWeights:
                transformedResults.append((transformedWeights, num_examples))
                continue

            # client was a straggler:
            for layer, [row, col] in enumerate(dropWeights[cid]):

                colLen = len(col)
                rowLen = len(row)

                # for each row that's dropped add a row in the weight parameter
                # with all 0s
                if (rowLen != 0):
                    transformedWeights[layer][row, ...] = 0
                    # since the row was dropped, the neuron did not train with this client's data
                    # Hence remove client's data count from total examples
                    # trained for related weights.
                    total_examples_wDrop[layer][row, ...] -= num_examples

                # for each row that's dropped add a row in the weight parameter
                # with all 0s
                if (colLen != 0):
                    transformedWeights[layer][:, col, ...] = 0
                    # since the colum was dropped, the neuron did not train with this client's data
                    # Hence remove client's data count from total examples
                    # trained for related weights.
                    total_examples_wDrop[layer][:, col, ...] -= num_examples

                    # Check if any number of examples for any weights were
                    # subtracted twice if both its row and col was dropped.
                    for r in row:
                        for c in col:
                            total_examples_wDrop[layer][r, c, ...] += num_examples

            # Append the transformed client model to the result list, with
            # number of examples that each individual weight trained on.
            transformedResults.append((transformedWeights, num_examples))

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [[layer * num_examples for layer in weights]
                            for weights, num_examples in transformedResults]

        # Compute average weights of each layer
        weights_prime = [
            torch.divide(reduce(torch.add, layer_updates), total_examples_wDrop[i])
            for i, layer_updates in enumerate(zip(*weighted_weights))
        ]

        avg_model = set_parameters(self.trainable_para_name, weights_prime)

        return avg_model

    
def set_parameters(model_para_name, parameters):
    state_dict = OrderedDict({k: v for k, v in zip(model_para_name, parameters)})
    return state_dict
