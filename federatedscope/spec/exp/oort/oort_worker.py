import os
import math
import copy
import torch
import numpy as np
from federatedscope.core.message import Message
from federatedscope.core.workers import Server, Client
from federatedscope.spec.worker import baseServer, baseClient
from federatedscope.core.auxiliaries.sampler_builder import get_sampler

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OortClient(baseClient):
    def __init__(self, ID=-1, server_id=None, state=-1, config=None, data=None, model=None, device='cpu', strategy=None, is_unseen_client=False, *args, **kwargs):
        super().__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)

    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)
        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)
        self.state = round

        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global"]
        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            # for these cases (1) unseen client (2) isolated_global_mode,
            # we do not local train and upload local model
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(
                    f"[Local/Global mode] Client #{self.ID} has been "
                    f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()
            sample_size, model_para_all, results = self.trainer.train()
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            train_loss = self._get_train_loss(train_log_res)

        # Return the feedbacks to the server after local update
        if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                ['krum', 'normbounding', 'median', 'trimmedmean',
                    'bulyan']:
            # Return the model delta when using asynchronous training
            # protocol, because the staled updated might be discounted
            # and cause that the sum of the aggregated weights might
            # not be equal to 1
            shared_model_para = self._calculate_model_delta(
                init_model=content, updated_model=model_para_all)
        else:
            shared_model_para = model_para_all

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, shared_model_para, train_loss)))

    def _get_train_loss(self, train_log_res):
        if 'Results_raw' in train_log_res:
            return train_log_res['Results_raw']['train_loss']
        else:
            return train_log_res['Results_weighted_avg']['train_loss']


class OortServer(baseServer):
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)
        
        self.oort_feed_backs = {}
        for idx, (duration, data_size) in enumerate(zip(self.client_duration_info, self.client_data_size)):
            self.oort_feed_backs[idx+1] = {
                'duration': duration,
                'reward': data_size,
                'gradient': data_size
            }

        self.train_size_per_round = [0] + kwargs['train_size_per_round']
        self.last_model_parameters = [torch.clone(p.data) for p in model.parameters()]

        self.avgUtilLastEpoch = 0.
        self.avgGradientUtilLastEpoch = 0.

        total_sample_size = sum(self.train_size_per_round)
        self.ratioSample = [0] + [sample_size / total_sample_size for sample_size in self.train_size_per_round]

    def init_sampler(self):
        # get sampler
        if 'client_resource' in self._cfg.federate.join_in_info:
            client_resource = [
                self.join_in_info[client_index]['client_resource']
                for client_index in np.arange(1, self.client_num + 1)
            ]
        else:
            client_resource = self.oort_feed_backs

        if self.sampler is None:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num=self.client_num,
                client_info=client_resource,
                cfg=self._cfg)
    
    def callback_funcs_model_para(self, message: Message):
        self.update_sampler_from_training_clients(message)

        return super().callback_funcs_model_para(message)

    def update_sampler_from_training_clients(self, message: Message):
        timestamp = message.timestamp
        round = message.state
        sender = message.sender
        sample_size, cur_model, train_loss = message.content

        if self._cfg.oort.capacity_bin == True:
            if not self._cfg.oort.enable_adapt_local_epoch:
                size_of_sample_bin = min(self.oort_feed_backs[sender]['reward'], self.train_size_per_round[sender]) 
                # oort_feed_backs[sender]['reward'] = client_data_size
            else:
                size_of_sample_bin = min(self.oort_feed_backs[sender]['reward'], sample_size)

        # register the score
        clientUtility = math.sqrt(train_loss) * size_of_sample_bin
        gradient_l2_norm = 0
        # apply the update into the global model if the client is involved
        with torch.no_grad():
            for para1, para2 in zip(cur_model.values(), self.model.state_dict().values()):
                if not (para1.dtype.is_floating_point and para2.dtype.is_floating_point):
                    continue
                gradient_l2_norm = (torch.norm(para1 - para2, 2)**2).item()

        gradientUtility = math.sqrt(gradient_l2_norm) * size_of_sample_bin/100

        self.sampler.registerScore(sender, clientUtility, gradientUtility, \
                                   auxi=math.sqrt(train_loss), \
                                   time_stamp=self.state + 1, \
                                   duration=self.oort_feed_backs[sender]['duration'])
                                   # state + 1 to avoid being 0

        self.avgUtilLastEpoch += self.ratioSample[sender] * clientUtility
        self.avgGradientUtilLastEpoch += self.ratioSample[sender] * gradientUtility
        message.content = (sample_size, cur_model)

    def update_sampler_from_explored_clients(self, exploredClients):
        for client_id in exploredClients:
            self.sampler.registerScore(client_id, self.avgUtilLastEpoch, self.avgGradientUtilLastEpoch,
                                       time_stamp=self.state,
                                       duration=self.oort_feed_backs[client_id]['duration'],
                                       success=True)
        self.avgUtilLastEpoch = 0.
        self.avgGradientUtilLastEpoch = 0.

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            sample_client_num = int(self._cfg.oort.overcommit * sample_client_num)
            if self.state > 0:
                receiver, exploredClients = self.sampler.sample(size=sample_client_num)
                self.update_sampler_from_explored_clients(exploredClients)
            else:
                receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        model_para = self.models[0].state_dict()

        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')
