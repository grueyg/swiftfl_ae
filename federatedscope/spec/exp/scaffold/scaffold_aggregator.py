import copy
import torch
import numpy as np
from federatedscope.core.aggregators import ClientsAvgAggregator
import logging

logger = logging.getLogger(__name__)

class ScaffoldAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model, device, config)

        self.server_learning_rate = self.cfg.scaffold.server_learning_rate
        
    def aggregate(self, agg_info):
        global_c = agg_info['global_c']
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        global_model_para, global_c = self._para_weighted_avg(models, global_c, recover_fun=recover_fun)

        return global_model_para, global_c
    
    def _para_weighted_avg(self, models, global_c, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        global_model = copy.deepcopy(self.model)

        num_clients = len(models)

        for sample_size, (dy, dc) in models:
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / num_clients * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / num_clients

        return global_model.state_dict(), global_c
    
