import sys
import pickle
from collections import defaultdict
from federatedscope.core.fed_runner import StandaloneRunner
from federatedscope.core.workers import Client, Server
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.trainers.utils import calculate_batch_epoch_num

class SpecStandaloneRunner(StandaloneRunner):
    def __init__(self, data, server_class=Client, client_class=Server, config=None, client_configs=None):
        super().__init__(data, server_class, client_class, config, client_configs)
    
    def _get_server_args(self, resource_info=None, client_resource_info=None):
        if self.server_id in self.data:
            server_data = self.data[self.server_id]
            model = get_model(self.cfg.model,
                              server_data,
                              backend=self.cfg.backend)
        else:
            # server_data = next(iter(self.data[1]['train']))
            server_data = None
            data_representative = self.data[1]
            model = get_model(
                self.cfg.model, data_representative, backend=self.cfg.backend
            )  # get the model according to client's data if the server
            # does not own data
        model_size = sys.getsizeof(pickle.dumps(model)) / 1024.0 * 8.

        client_data_size = [len(self.data[client_id].train_data) for client_id in self.data if client_id != 0]
        client_train_size_per_round = self.calculate_train_size(client_data_size)
    
        client_duration_info = []
        # client_feed_back = {}

        for capacity, train_size in zip(client_resource_info, client_train_size_per_round):
            computation_time = self.cfg.federate.augmentation_factor * \
                               float(capacity['computation']) * \
                               train_size / 1000
            # TODO: comp_time = num_batch_per_epoch * local_epoch / comp_speed  
            communication_time = 2 * model_size / float(capacity['communication'])
            round_time = computation_time + communication_time
            client_duration_info.append(round_time)

        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info,
            'client_resource_info': client_resource_info,
            'client_duration_info': client_duration_info,
            'client_data_size': client_data_size, 
            'train_size_per_round': client_train_size_per_round
        }

        return server_data, model, kw
    
    def calculate_train_size(self, client_data_size):
        if self.cfg.train.batch_or_epoch == 'epoch':
            return [data_size * self.cfg.train.local_update_steps for data_size in client_data_size]
        else:
            client_train_size_per_round = []
            for length in client_data_size:
                num_batch, num_batch_last_epoch, num_epoch, num_total_batch = \
                    calculate_batch_epoch_num(
                        self.cfg.train.local_update_steps *
                        self.cfg.grad.grad_accum_count,
                        self.cfg.train.batch_or_epoch,
                        length,
                        self.cfg.dataloader.batch_size,
                        self.cfg.dataloader.drop_last)
                if num_batch_last_epoch == num_batch:
                    train_size = num_epoch * length
                else:
                    train_size = (num_epoch - 1) * length + num_batch_last_epoch * self.cfg.dataloader.batch_size
                client_train_size_per_round.append(train_size)
            return client_train_size_per_round
