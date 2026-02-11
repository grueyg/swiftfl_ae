import copy
import torch
import numpy as np
from federatedscope.core.workers import Server, Client
from federatedscope.spec.worker import baseClient
from federatedscope.core.message import Message
from federatedscope.spec.exp.fluid.fluid_aggregator import set_parameters

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FLuIDClient(baseClient):
    def __init__(self, ID=-1, server_id=None, state=-1, config=None, data=None, model=None, device='cpu', strategy=None, is_unseen_client=False, *args, **kwargs):
        super().__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)
        self.trainable_para_name = list(dict(list(model.named_parameters())).keys())

    def callback_funcs_for_model_para(self, message: Message):
        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        if len(content) == 1:
            content = content[0]
        else:
            content = self.drop_neuron_weights(content[0], content[1])

        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)
        self.trainer.update(content, strict=self._cfg.federate.share_local_model)
        self.state = round

        if self.early_stopper.early_stopped and \
                self._monitor.local_convergence_round == 0:
            logger.info(
                f"[Normal FL Mode] Client #{self.ID} has been locally "
                f"early stopped. "
                f"The next FL update may result in negative effect")
            self._monitor.local_converged()

        sample_size, model_para_all, results = self.trainer.train()

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=self.state,
            role='Client #{}'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, model_para_all)))

    def drop_neuron_weights(self, model_para, dropWeights_loc):
        Model_Weights = []
        for key in self.trainable_para_name:
            Model_Weights.append(model_para[key])

        for layer, [row, col] in enumerate(dropWeights_loc):
            colLen = len(col)
            rowLen = len(row)
            if (rowLen != 0):
                Model_Weights[layer][row, ...] = 0
            if (colLen != 0):
                Model_Weights[layer][:, col, ...] = 0

        return model_para
    

        
    