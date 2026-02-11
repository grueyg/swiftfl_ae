import copy
import sys
import pickle
import numpy as np 

from federatedscope.core.message import Message
from federatedscope.core.workers import Server, Client
from federatedscope.core.auxiliaries.utils import calculate_time_cost, merge_dict_of_results
from federatedscope.core.auxiliaries.sampler_builder import get_sampler

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class baseClient(Client):

    def __init__(self, ID=-1, server_id=None, state=-1, config=None, data=None, model=None, device='cpu', strategy=None, is_unseen_client=False, *args, **kwargs):
        super().__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)

        self.cur_timestamp = 0
        self.augmentation_factor = config.federate.augmentation_factor

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None

        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth,
            augmentation_factor=self.augmentation_factor)
        
        if init_timestamp >= self.cur_timestamp:
            self.cur_timestamp = init_timestamp + comp_cost + comm_cost
        else:
            self.cur_timestamp = self.cur_timestamp + comp_cost + comm_cost

        return self.cur_timestamp
    
class baseServer(Server):

    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        server_config = copy.deepcopy(config)
        server_config.defrost()
        server_config.dataloader.batch_size=512
        server_config.freeze()

        super().__init__(ID, state, server_config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)

        self.timestamp_buffer = dict()
        self.staled_timestamp_buffer = list()
        self.last_round_timestamp = 0
        self.total_waiting_time = 0

        self.client_duration_info = kwargs['client_duration_info']
        self.client_data_size = kwargs['client_data_size']
        self.train_size_per_round = kwargs['train_size_per_round']

    def check_buffer(self, cur_round, min_received_num, check_eval_result=False, update_waiting=True):
        buffer_flag =  super().check_buffer(cur_round, min_received_num, check_eval_result)
        if buffer_flag and update_waiting and not check_eval_result:
            self.update_avg_waiting_rate(cur_round)
        return buffer_flag

    def callback_funcs_model_para(self, message: Message):
        timestamp = message.timestamp
        round = message.state
        sender = message.sender

        message.timestamp  =  self.update_timestamp(round, sender, timestamp)
   
        return super().callback_funcs_model_para(message)

    def check_and_move_on(self, check_eval_result=False, min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving, \
        some events (such as perform aggregation, evaluation, and move to \
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for \
                evaluation; and check the message buffer for training \
                otherwise.
            min_received_num: number of minimal received message, used for \
                async mode
        """
        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
                self.state += 1
                if self.state % self._cfg.eval.freq == 0 \
                    and self.state != self.total_round_num \
                    or self.state <= self._cfg.eval.start_interval_eval_round:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

        else:
            move_on_flag = False

        if move_on_flag:
            self.last_round_timestamp = self.cur_timestamp

        return move_on_flag

    def update_avg_waiting_rate(self, cur_round):
        round_time = self.cur_timestamp - self.last_round_timestamp
        if len(self.staled_timestamp_buffer) == 0:
            clients_waiting_time = [self.cur_timestamp - t for t in self.timestamp_buffer[cur_round].values()]
        elif cur_round not in self.timestamp_buffer:
            clients_waiting_time = [self.cur_timestamp - t[2] for t in self.staled_timestamp_buffer]
        else:
            cur_round_waiting_time = [self.cur_timestamp - t for t in self.timestamp_buffer[cur_round].values()]
            staled_waiting_time = [self.cur_timestamp - t[2] for t in self.staled_timestamp_buffer]
            clients_waiting_time = cur_round_waiting_time + staled_waiting_time
        round_waiting_time = np.mean(clients_waiting_time)
        round_waiting_rate =  round_waiting_time / round_time

        self.timestamp_buffer.pop(cur_round, None)
        self.total_waiting_time += round_waiting_time

        tmp_dict = {'Round': cur_round,
                    'timestamp': self.cur_timestamp,
                    'round_time': round_time,
                    'avg_waiting_time': round_waiting_time,
                    'avg_waiting_rate': round_waiting_rate}
        self._monitor.save_formatted_results(tmp_dict, save_file_name='waiting_rate.log')
        logger.info(tmp_dict)

        if self.state + 1 == self.total_round_num:
            finish_fl_msg = {'Round num': self.total_round_num,
                             'Total_time': self.cur_timestamp,
                             'Total_waiting_time': self.total_waiting_time,
                             'Avg_round_time': self.cur_timestamp / self.total_round_num,
                             'Waiting_rate': self.total_waiting_time / self.cur_timestamp}
            self._monitor.save_formatted_results(finish_fl_msg, save_file_name='waiting_rate.log')
            logger.info(finish_fl_msg)

    def update_timestamp(self, round, sender, timestamp):

        if round == self.state:
            if round not in self.timestamp_buffer:
                self.timestamp_buffer[round] = dict()
            if sender not in self.timestamp_buffer[round] \
                or timestamp > self.timestamp_buffer[round][sender]:
                self.timestamp_buffer[round][sender] = timestamp
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            self.staled_timestamp_buffer.append((round, sender, timestamp))

        if timestamp > self.cur_timestamp:
            self.cur_timestamp = timestamp
        else:
            timestamp = self.cur_timestamp
            
        return timestamp

    def merge_eval_results_from_all_clients(self):
        """
        Merge evaluation results from all clients, update best, \
        log the merged results and save them into eval_results.log

        Returns:
            the formatted merged results
        """
        round = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][round]
        eval_res_participated_clients = []
        eval_res_unseen_clients = []
        for client_id in eval_msg_buffer:
            if eval_msg_buffer[client_id] is None:
                continue
            if client_id in self.unseen_clients_id:
                eval_res_unseen_clients.append(eval_msg_buffer[client_id])
            else:
                eval_res_participated_clients.append(
                    eval_msg_buffer[client_id])

        formatted_logs_all_set = dict()
        for merge_type, eval_res_set in [("participated",
                                          eval_res_participated_clients),
                                         ("unseen", eval_res_unseen_clients)]:
            if eval_res_set != []:
                metrics_all_clients = dict()
                for client_eval_results in eval_res_set:
                    for key in client_eval_results.keys():
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(
                            float(client_eval_results[key]))
                formatted_logs = self._monitor.format_eval_res(
                    metrics_all_clients,
                    rnd=round,
                    role='Server #',
                    forms=self._cfg.eval.report)
                if merge_type == "unseen":
                    for key, val in copy.deepcopy(formatted_logs).items():
                        if isinstance(val, dict):
                            # to avoid the overrides of results using the
                            # same name, we use new keys with postfix `unseen`:
                            # 'Results_weighted_avg' ->
                            # 'Results_weighted_avg_unseen'
                            formatted_logs[key + "_unseen"] = val
                            del formatted_logs[key]
                logger.info(formatted_logs)
                formatted_logs_all_set.update(formatted_logs)
                self._monitor.update_best_result(
                    self.best_results,
                    metrics_all_clients,
                    results_type="unseen_client_best_individual"
                    if merge_type == "unseen" else "client_best_individual")
                tmp_res = self.change_formatted_results(formatted_logs)
                self._monitor.save_formatted_results(tmp_res)
                for form in self._cfg.eval.report:
                    if form != "raw":
                        metric_name = form + "_unseen" if merge_type == \
                                                          "unseen" else form
                        self._monitor.update_best_result(
                            self.best_results,
                            formatted_logs[f"Results_{metric_name}"],
                            results_type=f"unseen_client_summarized_{form}"
                            if merge_type == "unseen" else
                            f"client_summarized_{form}")

        return formatted_logs_all_set
    
    def eval(self):
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """

        if self._cfg.federate.make_global_eval:
            # By default, the evaluation is conducted one-by-one for all
            # internal models;
            # for other cases such as ensemble, override the eval function
            for i in range(self.model_num):
                trainer = self.trainers[i]
                # Preform evaluation in server
                metrics = {}
                for split in self._cfg.eval.split:
                    eval_metrics = trainer.evaluate(
                        target_data_split_name=split)
                    metrics.update(**eval_metrics)
                formatted_eval_res = self._monitor.format_eval_res(
                    metrics,
                    rnd=self.state,
                    role='Server #',
                    forms=self._cfg.eval.report,
                    return_raw=self._cfg.federate.make_global_eval)
                self._monitor.update_best_result(
                    self.best_results,
                    formatted_eval_res['Results_raw'],
                    results_type="server_global_eval")
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                tmp_res = self.change_formatted_results(formatted_eval_res)
                self._monitor.save_formatted_results(tmp_res)
                logger.info(formatted_eval_res)
            self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate',
                                      filter_unseen_clients=False)
            
    def change_formatted_results(self, formatted_results):
        if "Results_weighted_avg" in formatted_results:
            changed_results = {'Role': formatted_results['Role'],
                               'Round': formatted_results['Round'],
                               'Timestamp': self.cur_timestamp,
                               'Round_time': self.cur_timestamp-self.last_round_timestamp,
                               'Results_weighted_avg': formatted_results['Results_weighted_avg']}
        else:
            changed_results = {'Role': formatted_results['Role'],
                               'Round': formatted_results['Round'],
                               'Timestamp': self.cur_timestamp,
                               'Round_time': self.cur_timestamp-self.last_round_timestamp,
                               'Results_raw': formatted_results['Results_raw']}
        return changed_results
    
    def trigger_for_time_up(self, check_timestamp=None):
        """
        The handler for time up: modify the currency timestamp \
        and check the trigger condition
        """
        if self.is_finish:
            return False

        if check_timestamp is not None and \
                check_timestamp < self.deadline_for_cur_round or \
                self.state == self.total_round_num:
            return False

        self.cur_timestamp = self.deadline_for_cur_round
        self.check_and_move_on()
        return True
    
    def save_client_eval_results(self):
        pass

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """

        if self.check_client_join_in():
            if self._cfg.federate.use_ss or self._cfg.vertical.use:
                self.broadcast_client_address()

            self.init_sampler()

            # change the deadline if the asyn.aggregator is `time up`
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                self.deadline_for_cur_round = self.cur_timestamp + \
                                               self._cfg.asyn.time_budget

            # start feature engineering
            self.trigger_for_feat_engr(
                self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': self.sample_client_num
                })

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state))
            
    def init_sampler(self):
        # get sampler
        if 'client_resource' in self._cfg.federate.join_in_info:
            client_resource = [
                self.join_in_info[client_index]['client_resource']
                for client_index in np.arange(1, self.client_num + 1)
            ]
        else:
            client_resource = self.client_duration_info

        if self.sampler is None:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num=self.client_num,
                client_info=client_resource,
                cfg=self._cfg)