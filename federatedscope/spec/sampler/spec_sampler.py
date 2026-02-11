import numpy as np
import logging
from federatedscope.core.sampler import Sampler
from federatedscope.spec.decorators import record_client_state, recover_client_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SpecOptimalSampler(Sampler):
    def __init__(self, client_num, client_info, cfg):
        super().__init__(client_num)
        self.client_num = client_num
        self.sample_client_num = cfg.federate.sample_client_num
        self.overselection = cfg.spec.overselection
        self.init_client_info(client_info)

    @property
    def working_clients(self):
        return np.where(self.client_state == 0)[0][1:]
    
    @property
    def idle_clients(self):
        return np.where(self.client_state == 1)[0]

    def init_client_info(self, client_info):
        self.client_info = np.asarray(
            [float('inf')] + [x for x in client_info])
            # client_info[0] is preversed for the server
        assert len(self.client_info) == len(
            self.client_state
        ), "The first dimension of client_info is mismatched with client_num"

    def sample(self, size, sample_type='random_normal'):
        # size = int(size * self.overselection)
        sampled_clients = np.random.choice(self.idle_clients,
                                           size=size,
                                           replace=False).tolist()
        self.change_state(sampled_clients, 'working')
        return sampled_clients
        
    def spec_sample_in_normal(self, size=0):
        sorted_info, sorted_indices, prefix_sum_info = self.prepare_sorted_clients('working')
        max_sorted_info = sorted_info[-1]
        n = len(sorted_info)
        (index, ans) = (0, 0)
        # We don't need to search from t_0 because f(t_mid) > ... > f(t_0).
        mid_position = self.linear_search(sorted_info, sorted_info[-1] / 2, 0, 0)

        if size > 0:
            start_position = max(mid_position, size - 1)
        else:
            start_position = mid_position

        for i in range(start_position, n):
            if i + 1 == size or size == 0:
                sample_sum = prefix_sum_info[i]
            else:
                sample_sum = prefix_sum_info[i] - prefix_sum_info[i - size]

            res = (prefix_sum_info[-1] + sample_sum) \
                / (n * max(max_sorted_info, 2 * sorted_info[i]))
                
            if res > ans: 
                (ans, index) = (res, i)

        if size > 0:
            sampled_clients = sorted_indices[index - size + 1: index + 1]
        else:
            sampled_clients = sorted_indices[0: index + 1]

        print(1 - ans, sorted_info[-1], 2 * sorted_info[index], len(sampled_clients))
        sample_result = {'spec_sample_num': len(sampled_clients),
                         'waiting_rate': 1 - ans,
                         'max_time_before_sample:': sorted_info[-1],
                         'max_time_after_ample': max(sorted_info[-1], 2 * sorted_info[index])} 
        logger.info(sample_result)
        self.next_round_time = sample_result['max_time_after_sample']
        return sampled_clients.tolist()

    def spec_sample_in_all(self, size=0):
        working_info, working_indices, working_prefix_sum = self.prepare_sorted_clients('working')
        idle_info, idle_indices, idle_prefix_sum = self.prepare_sorted_clients('idle')
        longest_working_time_set = self.generate_maxtime_set() # 2T \cup CuT

        N = len(self.client_info) - 1
        n = len(working_info)
        max_working_info = working_info[-1]
        (p, q, s, ans, maxts) = (0, 0, 0, 0, 0)

        mid_position = self.linear_search(longest_working_time_set, max_working_info, 0, 0)
        max_idle_size = min(size, N - n) if size > 0 else N - n

        for idle_size in range(0, max_idle_size + 1):
            (p_, q_) = (0, 0)
            # for maxs in longest_working_time_set:
            for i in range(mid_position, N):
                maxs = longest_working_time_set[i]
                p_ = self.linear_search(working_info, maxs / 2, size - idle_size, p_)
                q_ = self.linear_search(idle_info, maxs, idle_size, q_)

                if p_ < 0 or q_ < 0:
                    (p_, q_) = (0, 0)
                    continue

                if p_ + 1 == size - idle_size or size == 0:
                    working_sum =  working_prefix_sum[p_]
                else:
                    working_sum = working_prefix_sum[p_] - working_prefix_sum[p_ - size + idle_size]
                if q_ + 1 == idle_size:
                    idle_sum = idle_prefix_sum[q_]
                else:
                    idle_sum = idle_prefix_sum[q_] - idle_prefix_sum[q_ - idle_size]

                res = (working_prefix_sum[-1] + working_sum + idle_sum) \
                    / ((n + idle_size) * max(max_working_info, maxs))
                
                if res > ans:
                    (p, q, s, ans, maxts) = (p_, q_, idle_size, res, maxs)

        if size > 0:
            if s == 0:
                sampled_clients =  working_indices[p - size + 1: p + 1]
            elif s == size:
                sampled_clients =  idle_indices[q - size + 1: q + 1]
            else:
                sampled_clients =  np.concatenate([working_indices[p - size + s + 1: p + 1], 
                                                   idle_indices[q - s + 1: q + 1]])
        else:
            if s == 0:
                sampled_clients = working_indices[0: p + 1]
            else:
                sampled_clients = np.concatenate([working_indices[0: p + 1], 
                                                  idle_indices[q - s + 1: q + 1]])

        sample_result = {'spec_sample_num': len(sampled_clients),
                         'waiting_rate': 1 - ans,
                         'max_time_before_sample:': max_working_info,
                         'max_time_after_sample': max(max_working_info, maxts)} 
        # logger.info(sample_result)
        self.change_state(sampled_clients, 'working')
        self.next_round_time = sample_result['max_time_after_sample']
        self.spec_waiting_rate = 1 - ans
        return sampled_clients.tolist()

    def linear_search(self, sorted_info, target, least_num_of_element_before_target, last_position):
        # find p such that a[p] <= target < a[p+1] when p <= n-1
        # or p such that a[p] > target when p = n - 1
        end_position = len(sorted_info) - 1
        if last_position >= end_position:
            return last_position
        i = last_position
        while i <= end_position and sorted_info[i] <= target:
            i += 1 
        if i == 0: # the first element is greater than target
            return -1
        if i < least_num_of_element_before_target:
            return -1
        return i-1
    
    def prepare_sorted_clients(self, state):
        if state == 'working':
            client_indices = self.working_clients
        elif state == 'idle':
            client_indices = self.idle_clients
        else:
            raise ValueError()
        
        client_info = self.client_info[client_indices]
        sorted_info, sorted_indices = self.sort_by_client_info(client_info, client_indices)
        prefix_sum_info = np.cumsum(sorted_info)

        return sorted_info, sorted_indices, prefix_sum_info
    
    def sort_by_client_info(self, client_info, client_indices):
        sorted_indices = np.argsort(client_info)
        sorted_info = client_info[sorted_indices]
        sorted_indices = client_indices[sorted_indices]
        return sorted_info, sorted_indices
    
    def generate_maxtime_set(self):
        new_info = self.client_info.copy()
        new_info[self.working_clients] *= 2
        return np.sort(new_info[1:])
    
    def calculate_working_rate(self, normal_clients, spec_clients=None):
        if not isinstance(normal_clients, np.ndarray):
            normal_clients = np.array(normal_clients)
        normal_info = self.client_info[normal_clients]

        if not spec_clients:
            return np.max(normal_info), np.average(normal_info) / np.max(normal_info)
        else:
            if not isinstance(spec_clients, np.ndarray):
                spec_clients = np.array(spec_clients)

            normal_info = self.client_info[normal_clients]
            spec_info = self.client_info[spec_clients]

            normal_clients = set(normal_clients)
            spec_clients = set(spec_clients)
            client_num = len(normal_clients.union(spec_clients))

            intersection_clients = normal_clients.intersection(spec_clients) # S_1 = T \cap S
            difference_clients = spec_clients.difference(intersection_clients) # S_2 = S \ S_1

            difference_info = self.client_info[list(difference_clients)]
            intersection_info = self.client_info[list(intersection_clients)]

            max_normal_info = np.max(normal_info) if len(normal_info) != 0 else 0
            max_difference_info = np.max(difference_info) if len(difference_info) != 0 else 0
            max_intersection_info = np.max(intersection_info) if len(intersection_info) != 0 else 0

            max_working_time = np.max([max_normal_info, max_difference_info, 2 * max_intersection_info])
            
            return max_working_time, (np.sum(normal_info) + np.sum(spec_info)) / (client_num * max_working_time)

    def get_top_k_clients(self, sampledClientsTemp, topk, client_info):
        completionTimes = client_info[sampledClientsTemp]

        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
        top_k_index = sortedWorkersByCompletion[:topk]
        clients_to_run = [sampledClientsTemp[k] for k in top_k_index]
        ## TODO: return the adaptive local epoch
        # dummy_clients = [sampledClientsRealTemp[k] for k in sortedWorkersByCompletion[numToRealRun:]]

        return clients_to_run

class IdpSampler(SpecOptimalSampler):
    def __init__(self, client_num, client_info, cfg):
        super().__init__(client_num, client_info, cfg)
        self.client_state_buffer = dict()

    @record_client_state
    def sample(self, size, sample_type='random_normal', round=None):
        return super().sample(size, sample_type)
    
    @recover_client_state
    def spec_sample_in_normal(self, size=0, round=None):
        return super().spec_sample_in_normal(size)

    
    @recover_client_state
    def spec_sample_in_all(self, size=0, round=None):
        return super().spec_sample_in_all(size)
    
    def evaluate_sample(self, size, sample_type='random_normal'):
        if sample_type == 'random_normal':
            idle_clients = np.nonzero(self.client_state)[0]
        else:
            idle_clients = np.array(list(range(1, self.client_num + 1)))
        sampled_clients = np.random.choice(idle_clients,
                                           size=size,
                                           replace=False).tolist()
        self.change_state(sampled_clients, 'working')
        return sampled_clients

class SpecOptTimeSampler(SpecOptimalSampler):
    def __init__(self, client_num, client_info, cfg):
        self.client_data_size = [0] + client_info['data_size']
        super().__init__(client_num, client_info['duration_info'], cfg)

    def sample(self, size, sample_type='random_normal'):
        if sample_type == 'random_normal':
            sampled_clients = super().sample(size, sample_type)
        elif sample_type == 'random_topk':
            topk = int(self.overselection * self.sample_client_num)
            sampled_clients = self.get_top_k_clients(self.idle_clients,
                                                     topk=topk,
                                                     client_info=self.client_info)
            sampled_clients = np.random.choice(sampled_clients,
                                            size=size,
                                            replace=False).tolist()
            self.change_state(sampled_clients, 'working')
        return sampled_clients
    