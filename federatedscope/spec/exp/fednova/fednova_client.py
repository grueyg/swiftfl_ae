import copy
from federatedscope.core.workers import Server, Client
from federatedscope.spec.worker import baseClient
from federatedscope.core.message import Message
import torch

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FedNovaClient(baseClient):
    
    def get_local_norm_grad(self, opt, cur_params, init_params, weight=0):
        if weight == 0:
            weight = opt.ratio
        grad_dict = {}
        for k in cur_params.keys():
            if 'tracked' in k:
                continue
            scale = 1.0 / opt.local_normalizing_vec
            cum_grad = init_params[k] - cur_params[k]
            cum_grad.mul_(scale)
            grad_dict[k] = cum_grad
        return grad_dict
    
    def get_local_tau_eff(self, opt):
        # if opt.mu != 0:
        #     return opt.local_steps * opt.ratio
        # else:
        #     return opt.local_normalizing_vec * opt.ratio
        if opt.mu != 0:
            return opt.local_steps
        else:
            return opt.local_normalizing_vec
        
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

        norm_grad = self.get_local_norm_grad(self.trainer.ctx.optimizer,
                                             model_para_all, content)
        tau_eff = self.get_local_tau_eff(self.trainer.ctx.optimizer)

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, norm_grad, tau_eff)))

        
    