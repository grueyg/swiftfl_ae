import os
import copy
import numpy as np
import torch
import torch.multiprocessing as mp
from federatedscope.core.message import Message
from federatedscope.spec.trainer import SpecUpdater
from federatedscope.spec.predictor import PredictorData
from federatedscope.core.auxiliaries.utils import merge_param_dict, merge_dict_of_results
from federatedscope.spec.predictor.predictor_builder import get_predictor
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.spec.utils.tool import gradient_operator
from federatedscope.spec.worker.base_worker import baseServer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class idpSpecServer(baseServer):
    
    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)

        self.register_handlers('spec_model_para', self.callback_funcs_spec_model_para,
                               ['model_para', 'evaluate', 'finish'])

        # normal FL gradient
        self.aggregated_gradient = {}
        
        # spec FL gradient
        self.spec_model = copy.deepcopy(model)
        if config.federate.process_num > 1:
            self.spec_model.to(self.device)
        self.spec_msg_buffer = {'train':dict()}
        self.spec_aggregated_gradient = {}
        self.do_compensation = config.spec.do_compensation
        self.cur_round_spec_sample_num = 0
        # Build model updater
        self.updater = SpecUpdater(self.model, device, config)

        self.init_predictor()

    @property
    def spec_state(self):
        return self.state + self.aft_seq_length

    def init_predictor(self):
        # predictor args
        self.warm_round = self._cfg.spec.warm_round
        self.pre_seq_length = self._cfg.spec.predictor.pre_seq_length
        self.aft_seq_length = self._cfg.spec.predictor.aft_seq_length
        self.train_seq_length = self._cfg.spec.predictor.train_seq_length
        self.start_spec_round = self._cfg.spec.start_spec_round
        
        self.predictor_data = PredictorData(self._cfg.spec.predictor, self.model)
        self.server2predictor_queue = mp.Queue()
        self.predictor2server_queue = mp.Queue()
        self.data_channel = mp.Queue()
        self.predictor = get_predictor(self._cfg, 
                                       send_channel=self.predictor2server_queue,
                                       recv_channel=self.server2predictor_queue,
                                       data_channel=self.data_channel)
        # predicted gradient
        self.predicted_gradient = {}

    def make_train_data(self):
        if len(self.predictor_data) < self.train_seq_length:
            self.predictor_data.add_train_data(self.aggregated_gradient[self.state])
        else:
            self.predictor_data.update_train_data(self.aggregated_gradient[self.state])

    def synchronize_predictor(self, content='finish'):
        send_flag = True
        while True:
            if not self.predictor2server_queue.empty():
                msg, model_para = self.predictor2server_queue.get()
                self.predictor.model.load_state_dict(model_para)
                train_msg = {"Round": self.state - 1}
                train_msg.update(msg)
                self._monitor.save_formatted_results(train_msg, save_file_name=self._cfg.spec.predictor.type + '_training.log')
                logger.info(msg)
                break
            elif send_flag:
                send_flag = False
                self.server2predictor_queue.put(content)

    def spec_predict(self):
        self.synchronize_predictor()
        predict_data = self.predictor_data.get_predict_data()
        predict_result =  self.predictor.predict(predict_data)
        self.predicted_gradient[self.spec_state] = \
            self.predictor_data.reconstruct_model_gradient(predict_result)
        self.model_para_result = self.updater.update_by_aggregated_gradient(\
            self.model.state_dict(), self.predicted_gradient[self.spec_state])

    def spec_train(self):
        train_data = self.predictor_data.get_train_data()
        self.data_channel.put(train_data)
        if self.state == self.start_spec_round:
            process = mp.Process(target=self.predictor.train)
            process.start()
            # process.join()

    def _start_new_training_round(self, aggregated_num=0):
        self.broadcast_model_para(msg_type='model_para',
                                    sample_client_num=self.sample_client_num) 
        if self.state - 1 > self.start_spec_round and \
            self.state % self._cfg.spec.predict_freq == 0:
            self.broadcast_spec_model_para(
                msg_type='spec_model_para',
                sample_client_num=self._cfg.spec.sample_client_num)

    def callback_funcs_spec_model_para(self, message: Message):
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        # update the currency timestamp according to the received message
        actual_round = round - self.aft_seq_length
        message.timestamp = self.update_timestamp(actual_round, sender, timestamp)
        
        if round == self.state:
            if round not in self.spec_msg_buffer['train']:  
                self.spec_msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.spec_msg_buffer['train'][round][sender] = content
        else:  
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date speculative message from round #{round} state #{self.state}')

        move_on_flag = self.check_and_move_on()

        return move_on_flag
    
    def aggregate_predict_and_spec(self):
        if self._cfg.spec.normal_spec_lambda > 0:
            weight_predict = self._cfg.spec.normal_spec_lambda
        else:
            weight_predict = self.normal_set_size / (self.normal_set_size + self.spec_set_size)
        weight_spec = 1 - weight_predict
        model_para_predict = gradient_operator(torch.mul, self.model.state_dict(), weight_predict)
        model_para_spec = gradient_operator(torch.mul, self.spec_model.state_dict(), weight_spec)
        model_para_result = gradient_operator(torch.add, model_para_predict, model_para_spec)

        self.spec_model.load_state_dict(model_para_result, strict=False)

    def correction_spec_and_compensation(self):
        '''
        spec_grad_{n+a+1} := spec_grad_{n+a+1} 
            + \eta * \lambda * grad_{n+1} \odot grad_{n+1} \odot (\sum_{i=n+1}^{n+a} grad_{i} - predict_grad_{n+a})
        '''
        if self._cfg.spec.normal_spec_lambda > 0:
            weight_predict = self._cfg.spec.normal_spec_lambda
        else:
            weight_predict = self.normal_set_size / (self.normal_set_size + self.spec_set_size)
        alpha = -1 * self._cfg.train.optimizer.lr * self._cfg.spec.hessian_approx_lambda
        hessian_approximation = gradient_operator(torch.mul,
                                self.aggregated_gradient[self.state - self.aft_seq_length],
                                self.aggregated_gradient[self.state - self.aft_seq_length])
        
        grad = self.aggregated_gradient[self.state - self.aft_seq_length + 1]
        if self.aft_seq_length > 1:
            for round in range(self.state - self.aft_seq_length + 2, self.state + 1):
                grad = gradient_operator(torch.add, grad, self.aggregated_gradient[round])

        predicted_gradient = merge_param_dict(copy.deepcopy(grad), self.predicted_gradient[self.state])
        error = gradient_operator(torch.sub, grad, predicted_gradient)
        delta = gradient_operator(torch.mul, hessian_approximation, error)
        delta = gradient_operator(torch.mul, delta, alpha * weight_predict)
        self.spec_aggregated_gradient[self.state] = gradient_operator(torch.add,
                self.spec_aggregated_gradient[self.state], delta)

    def _perform_federated_aggregation(self, msg_buffer, round, model):
        """
        Perform federated aggregation and update the global model
        """
        train_msg_buffer = msg_buffer['train'][round]  # 取出当前聚合轮次的
        for model_idx in range(self.model_num):  # model_num为单个客户端参与联邦学习训练的模型数，默认为1
            aggregator = self.aggregators[model_idx]

            # Prepare non-spec msg
            msg_list, staleness = self._prepare_train_msg(train_msg_buffer, model_idx)

            # Trigger the monitor here (for training)
            self._monitor.calc_model_metric(model.state_dict(),
                                            msg_list,
                                            rnd=round)
            
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
                'staleness': staleness,
            }
            # logger.info(f'The staleness is {staleness}')
            result, train_set_size = aggregator.aggregate(agg_info)
            grad = gradient_operator(torch.sub, result, model.state_dict().copy())
            grad = gradient_operator(torch.div, grad, -1 * self._cfg.train.optimizer.lr)
            # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)

        return grad, train_set_size

    def _perform_spec(self):
        if self.state % self._cfg.spec.predict_freq == 0:
            if self.state == self.start_spec_round + 1:
                self.spec_aggregated_gradient[self.state] = copy.deepcopy(self.model.state_dict())
            if self.state > self.start_spec_round + 1:
                self.spec_aggregated_gradient[self.state], self.spec_set_size = \
                    self._perform_federated_aggregation(self.spec_msg_buffer, self.state, self.spec_model)
            if self.state > self.start_spec_round + self.aft_seq_length:
                if self.do_compensation:
                    self.correction_spec_and_compensation()
                self.aggregate_predict_and_spec()

        if self.state > self.warm_round:
            self.make_train_data()
        if self.state >= self.start_spec_round:
            self.spec_train()
        if self.state > self.start_spec_round and \
            (self.state + 1) % self._cfg.spec.predict_freq == 0:
                self.spec_predict()
            
    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):

        min_received_num = self._cfg.federate.sample_client_num
        if self._cfg.spec.sample_client_num > 0:
            spec_min_received_num = self._cfg.spec.sample_client_num
        else:
            spec_min_received_num = self.cur_round_spec_sample_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation

        if self.state > self.start_spec_round + 1 and self.state % self._cfg.spec.predict_freq == 0:
            buffer_flag = self.check_buffer(self.state, min_received_num, check_eval_result, update_waiting=False) \
                and self.check_spec_buffer(self.state, spec_min_received_num, check_eval_result)
            if buffer_flag and not check_eval_result:
                self.update_avg_waiting_rate(self.state)
        else:
            buffer_flag = self.check_buffer(self.state, min_received_num, check_eval_result)

        if buffer_flag:
            if not check_eval_result:
                # Receiving enough feedback in the training process
                self.aggregated_gradient[self.state], self.normal_set_size = \
                    self._perform_federated_aggregation(self.msg_buffer, self.state, self.model)

                self._perform_spec()
                self.clear_buffer()
                self.state += 1

                if self.state < self._cfg.spec.recover_round:
                    eval_freq = self._cfg.eval.freq
                else:
                    eval_freq = self._cfg.spec.recover_eval_freq
                if self.state % eval_freq == 0 and \
                    self.state != self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')

                    # Start a new training round
                    self._start_new_training_round()
                else:
                    # Final Evaluate
                    # self.synchronize_predictor()
                    self.data_channel.put('finish_FL')
                    logger.info('Server: Training is finished! Starting evaluation.')
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

    def check_spec_buffer(self,
                          cur_round,
                          min_received_num,
                          check_eval_result=False):

        if check_eval_result:
            return True
        else:
            if cur_round not in self.spec_msg_buffer['train']: 
                cur_buffer = dict()  
            else:
                cur_buffer = self.spec_msg_buffer['train'][cur_round] 
            
            return len(cur_buffer) >= min_received_num

    def _prepare_train_msg(self, train_msg_buffer, model_idx):
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
            staleness.append((client_id, 0))  # 当前client_id不是staleness

        for staled_message in self.staled_msg_buffer:
            state, client_id, content = staled_message
            if self.model_num == 1:
                msg_list.append(content)
            else:
                train_data_size, model_para_multiple = content
                msg_list.append(
                    (train_data_size, model_para_multiple[model_idx]))

            staleness.append((client_id, self.state - state))

        return msg_list, staleness

    def clear_buffer(self):

        # Clear the msg_buffer $S
        self.msg_buffer['train'][self.state].clear()
        self.msg_buffer['train'][self.state + 1] = dict()
        self.staled_msg_buffer.clear()

        if self.state <= self.start_spec_round:
            self.aggregated_gradient.clear()
        elif self.state > self.start_spec_round + self.aft_seq_length:
            self.aggregated_gradient.pop(self.state - self.aft_seq_length, None)

        if self.state % self._cfg.spec.predict_freq == 0:
            # Clear the spec_msg_buffer $S
            if self.state > self.start_spec_round + 1:
                self.spec_msg_buffer['train'][self.state].clear()
                self.spec_msg_buffer['train'][self.state + 1] = dict()
            
            # Clear the spec_aggregated_gradient
            if self.state > self.start_spec_round + self.aft_seq_length:
                self.spec_aggregated_gradient.pop(self.state, None)
                self.predicted_gradient.pop(self.state, None)

    def broadcast_spec_model_para(self,
                                  msg_type='spec_model_para',
                                  sample_client_num=-1):
        sample_type = self._cfg.spec.sampler
        if sample_client_num >= 0:
            if sample_type == 'optimal_normal':
                receiver = self.sampler.spec_sample_in_normal(size=sample_client_num)
            elif sample_type == 'optimal_all':
                receiver = self.sampler.spec_sample_in_all(size=sample_client_num)
            else:
                receiver = self.sampler.sample(size=sample_client_num,
                                               sample_type=sample_type)
            if self._cfg.spec.sample_client_num == 0:
                self.cur_round_spec_sample_num = len(receiver)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        if self.state < self._cfg.spec.recover_round:
            model_para = self.spec_model.state_dict()
        else:
            model_para = self.model.state_dict()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))
    
    def broadcast_model_para(self, msg_type='model_para', sample_client_num=-1, filter_unseen_clients=True):

        if sample_client_num > 0:
            if self.state > self.start_spec_round and \
                self.state < self._cfg.spec.recover_round :
                receiver = self.sampler.sample(size=sample_client_num,
                                               sample_type='random_topk')
            else:
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
        # rnd = self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para))

    def eval(self):
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """

        if self._cfg.federate.make_global_eval:
            metrics = {}
            for split in self._cfg.eval.split:
                eval_metrics = self.trainer.evaluate(
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
            # self._monitor.save_formatted_results(formatted_eval_res)
            logger.info(formatted_eval_res)
            self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_spec_model_para(msg_type='evaluate')

    def init_sampler(self):
        # get sampler
        if 'client_resource' in self._cfg.federate.join_in_info:
            client_resource = [
                self.join_in_info[client_index]['client_resource']
                for client_index in np.arange(1, self.client_num + 1)
            ]
        else:
            if self._cfg.federate.sampler == 'spec':
                client_resource = self.client_duration_info
            elif self._cfg.federate.sampler == 'spec_opt':
                client_resource = {}
                client_resource['duration_info'] = self.client_duration_info
                client_resource['data_size'] = self.client_data_size

        if self.sampler is None:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num=self.client_num,
                client_info=client_resource,
                cfg=self._cfg)
            
        return super().init_sampler()