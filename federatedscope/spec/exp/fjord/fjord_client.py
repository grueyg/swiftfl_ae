import copy
import torch
import numpy as np
from federatedscope.core.workers import Server, Client
from federatedscope.spec.worker import baseClient
from federatedscope.core.message import Message
from federatedscope.spec.exp.fjord.utils.agg_config import get_agg_config

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FjORDClient(baseClient):
    def __init__(self, ID=-1, server_id=None, state=-1, config=None, data=None, model=None, device='cpu', strategy=None, is_unseen_client=False, *args, **kwargs):
        super().__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)

        if config.fjord.client_tier_allocation == 'uniform':
            cid_to_max_p = {cid+1: value for cid, value in \
                            enumerate(np.linspace(0.2, 1.2, config.federate.client_num))}
        else:
            raise ValueError(f"Unknown client_tier_allocation: {config.fjord.client_tier_allocation}")
        self.max_p = cid_to_max_p[self.ID]
        self.p_s = config.fjord.p_s

        self.trainer.init_p(p_s=self.p_s, max_p=self.max_p)


    def callback_funcs_for_model_para(self, message: Message):
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

        client_res = (model_para_all, self.max_p)

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, client_res)))
        
    

        
    