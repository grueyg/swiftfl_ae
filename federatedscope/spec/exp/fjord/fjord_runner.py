import sys
import pickle
import logging
import numpy as np
from collections import deque

from federatedscope.core.workers import Client, Server
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from federatedscope.spec.spec_runner import SpecStandaloneRunner
from federatedscope.spec.exp.fjord.utils.agg_config import get_agg_config

logger = logging.getLogger(__name__)

class FjORDStandaloneRunner(SpecStandaloneRunner):
    def __init__(self, data, server_class=Client, client_class=Server, config=None, client_configs=None):
        super().__init__(data, server_class, client_class, config, client_configs)
    
    def _set_up(self):
        """
        To set up server and client for standalone mode.
        """
        self.is_run_online = True if self.cfg.federate.online_aggr else False
        self.shared_comm_queue = deque()

        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.method == "global":
            self.cfg.defrost()
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:
            if len(self.resource_info) < self.cfg.federate.client_num + 1:
                replace = True
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num + 1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()
        # assume the client-wise data are consistent in their input&output
        # shape
        self._shared_client_model = get_model(
            self.cfg.model, self.data[1], backend=self.cfg.backend
        ) if self.cfg.federate.share_local_model else None


        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

        # in standalone mode, by default, we print the trainer info only
        # once for better logs readability
        trainer_representative = self.client[1].trainer
        if trainer_representative is not None and hasattr(
                trainer_representative, 'print_trainer_meta_info'):
            trainer_representative.print_trainer_meta_info()

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info
        }
        return client_data, kw


