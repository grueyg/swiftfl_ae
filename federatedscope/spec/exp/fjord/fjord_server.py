import numpy as np

from federatedscope.core.auxiliaries.utils import merge_param_dict
from federatedscope.spec.worker.base_worker import baseServer
from federatedscope.spec.exp.fjord.utils.agg_config import get_agg_config


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FjORDServer(baseServer):
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)

        self.p_s = config.fjord.p_s
        data_representative = next(iter(data['test']))[0]
        self.fjord_config = get_agg_config(model, data_representative, self.p_s)

        self.aggregator.init_fjord_config(self.p_s, self.fjord_config)

        self.trainer.init_p(p_s=self.p_s, max_p=1.0)

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
                if self.model_num == 1:
                    msg_list.append(train_msg_buffer[client_id])
                else:
                    train_data_size, model_para_multiple = \
                        train_msg_buffer[client_id]
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                # The staleness of the messages in train_msg_buffer
                # should be 0
                staleness.append((client_id, 0))

            for staled_message in self.staled_msg_buffer:
                state, client_id, content = staled_message
                if self.model_num == 1:
                    msg_list.append(content)
                else:
                    train_data_size, model_para_multiple = content
                    msg_list.append(
                        (train_data_size, model_para_multiple[model_idx]))

                staleness.append((client_id, self.state - state))

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(self.models[0].state_dict(),
                                            msg_list,
                                            rnd=self.state)

            # Aggregate
            aggregated_num = len(msg_list)
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            # logger.info(f'The staleness is {staleness}')
            result = aggregator.aggregate(agg_info)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

        return aggregated_num