import copy
import numpy as np

def record_client_state(func):
    def wrapper(self, size, sample_type='random_normal', round=None):
        sampled_clients = func(self, size, sample_type=sample_type, round=round)

        round_time, working_rate = self.calculate_working_rate(sampled_clients)
        time_msg = {'round_time': round_time, 'round_waiting_rate': 1 - working_rate}

        if round is not None:
            self.client_state_buffer[round] = copy.deepcopy(self.client_state)
        return time_msg, sampled_clients
    return wrapper

def recover_client_state(func):
    def wrapper(self, size, round=None):
        tmp_state = self.client_state
        self.client_state = copy.deepcopy(self.client_state_buffer[round])
        sampled_clients = func(self, size, round=round)

        normal_clients = np.where(self.client_state_buffer[round] == 0)[0][1:]
        round_time, working_rate = self.calculate_working_rate(normal_clients, np.array(sampled_clients))
        time_msg = {'round_time': round_time, 'round_waiting_rate': 1 - working_rate}
        
        self.client_state = tmp_state
        self.client_state_buffer.pop(round, None)

        return time_msg, sampled_clients
    return wrapper