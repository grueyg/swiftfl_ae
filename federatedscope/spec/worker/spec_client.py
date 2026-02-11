from federatedscope.spec.worker.base_worker import baseClient
from federatedscope.core.message import Message

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SpecClient(baseClient):

    def __init__(self, ID=-1, server_id=None, state=-1, config=None, data=None, model=None, device='cpu', strategy=None, is_unseen_client=False, *args, **kwargs):
        super().__init__(ID, server_id, state, config, data, model, device, strategy, is_unseen_client, *args, **kwargs)
        

        self.register_handlers('spec_model_para',
                               self.callback_funcs_for_spec_model_para,
                               [None])
        
    def callback_funcs_for_spec_model_para(self, message: Message):
        """
        The handling function for receiving speculative model parameters, \
        which triggers the local speculative training process. \

        Arguments:
            message: The received message
        """
        spec_round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device)
        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model)

        # self.state = round
        
        if self.early_stopper.early_stopped and \
                self._monitor.local_convergence_round == 0:
            logger.info(
                f"[Spec FL Mode] Client #{self.ID} has been locally "
                f"early stopped. "
                f"The next FL update may result in negative effect")
            self._monitor.local_converged()

        sample_size, model_para_all, results = self.trainer.train()

        train_log_res = self._monitor.format_eval_res(
            results,
            rnd=spec_round,
            role='Client #{} speculative training'.format(self.ID),
            return_raw=True)
        logger.info(train_log_res)
        
        if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
            self._monitor.save_formatted_results(train_log_res,
                                                    save_file_name="")
        
        shared_model_para = model_para_all

        self.comm_manager.send(
            Message(msg_type='spec_model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=spec_round,
                    timestamp=self._gen_timestamp(
                        init_timestamp=timestamp,
                        instance_number=sample_size),
                    content=(sample_size, shared_model_para)))
