import math
import pickle
import os
import numpy as np
import logging

from federatedscope.core.sampler import Sampler
from federatedscope.spec.utils.tool import get_top_k_indices
from federatedscope.spec.exp.oort.oort import create_training_selector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
class OortSampler(Sampler):
    # client num, mode:"oort", score:"loss",
    def __init__(self, client_num, client_info, cfg):
        super(OortSampler, self).__init__(client_num)
        self.count = 0
        self.client_num = client_num
        self.virtual_client_clock = {}
        self.global_virtual_clock = 0.0
        self.stats_util_accumulator = []
        self.loss_accumulator = []

        self.sample_client_num = cfg.federate.sample_client_num
        self.ucbSampler = create_training_selector(args=cfg.oort)

        self.duration_info = []
        self.register_client(client_info)

    @property
    def working_clients(self):
        return np.where(self.client_state == 0)[0][1:]
    
    @property
    def idle_clients(self):
        return np.where(self.client_state == 1)[0]


    def register_client(self, client_info):
        for client_id in range(1, self.client_num + 1):
            feed_backs = client_info[client_id]
            self.ucbSampler.register_client(client_id, feed_backs)
            self.duration_info.append(feed_backs['duration'])
        self.duration_info = np.asarray([float('inf')] + self.duration_info)
        
    def sample(self, size):
        self.count += 1

        if len(self.idle_clients) <= size:
            return self.idle_clients

        sampled_clients = None
        clients_online_set = set(self.idle_clients.tolist())

        if self.count > 1:
            tmp_sampled_clients= self.ucbSampler.select_participant(
                size, feasible_clients=clients_online_set)
            sampled_clients, dummy_clients = self.get_top_k_clients(tmp_sampled_clients)
            self.change_state(sampled_clients, 'working')
            return sampled_clients, dummy_clients
        else:
            sampled_clients = np.random.choice(self.idle_clients,
                                               size=size,
                                               replace=False).tolist()
            
            self.change_state(sampled_clients, 'working')
            return sampled_clients
    
    def get_top_k_clients(self, sampledClientsRealTemp):
        completionTimes = self.duration_info[sampledClientsRealTemp]
        numToRealRun = self.sample_client_num

        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
        top_k_index = sortedWorkersByCompletion[:numToRealRun]
        clients_to_run = [sampledClientsRealTemp[k] for k in top_k_index]
        ## TODO: return the adaptive local epoch
        dummy_clients = [sampledClientsRealTemp[k] for k in sortedWorkersByCompletion[numToRealRun:]]

        return clients_to_run, dummy_clients

    def registerScore(self, clientId, reward, gradient,auxi=1.0, time_stamp=0, duration=1., success=True):
        # currently, we only use distance as reward
        feedbacks = {
            'reward': reward,
            'gradient': gradient,
            'duration': duration,
            'status': True,
            'time_stamp': time_stamp
        }

        self.ucbSampler.update_client_util(clientId, feedbacks=feedbacks)
    