import random
import copy
import torch
from collections import OrderedDict

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_param_dict
from federatedscope.spec.worker.base_worker import baseServer
from federatedscope.spec.exp.fluid.strategy import ResNet18Strategy, ConvNet2Strategy


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FLuIDServer(baseServer):
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)

        if self._cfg.model.type == 'resnet18':
            self.fluid_strategy = ResNet18Strategy(self.client_duration_info,
                                                   config=self._cfg,
                                                   model=self.models[0])
        elif self._cfg.model.type == 'convnet2':
            self.fluid_strategy = ConvNet2Strategy(self.client_duration_info,
                                                   config=self._cfg,
                                                   model=self.models[0])
        elif self._cfg.model.type == 'lstm':
            self.idxList = [0, 3]
            self.idxConvFC = 10

    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num):
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()
            staleness = list()

            for client_id in train_msg_buffer.keys():
                sample_size, para = train_msg_buffer[client_id]
                msg_list.append((para, sample_size, client_id))

                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                msg_list.append(content)

                staleness.append((client_id, self.state - state))

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)


            self.fluid_strategy.find_stable_and_min(msg_list, self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
                'droppedWeights': self.fluid_strategy.get_droppedWeights()
            }
            # logger.info(f'The staleness is {staleness}')

            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

            self.straggler = self.fluid_strategy.update_straggler(msg_list, self.state)

        return aggregated_num
    
    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        
        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.models[0].state_dict()

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state
        
        if self.state > 1 and msg_type == 'model_para':

            non_straggler_clients = [cid for cid in receiver if cid not in self.straggler]
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=non_straggler_clients,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=(model_para,)))
            
            straggler_clients = [cid for cid in receiver if cid in self.straggler]
            for cid in straggler_clients:
                droppedWeights = self.fluid_strategy.drop_dynamic(model_para, cid)
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[cid],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=(model_para, droppedWeights)))
        else: # 'evaluate
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receiver,
                        state=min(rnd, self.total_round_num),
                        timestamp=self.cur_timestamp,
                        content=(model_para,)))
