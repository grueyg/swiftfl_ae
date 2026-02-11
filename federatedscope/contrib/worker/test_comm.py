import logging
from federatedscope.core.workers import Client
from federatedscope.core.workers import Server
from federatedscope.core.message import Message
from federatedscope.register import register_worker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class testServer(Server):

    def __init__(self, ID=-1, state=0, config=None, data=None, model=None, client_num=5, total_round_num=10, device='cpu', strategy=None, unseen_clients_id=None, **kwargs):
        super().__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, unseen_clients_id, **kwargs)
 
    def _start_new_training_round(self, aggregated_num=0):

        self.broadcast_test_model_para(msg_type='model_para', sample_client_num=1) 
        self.broadcast_test_model_para(msg_type='model_para', sample_client_num=-1) 
            
    def broadcast_test_model_para(self,
                                  msg_type='test_model_para',
                                  sample_client_num=-1):
        
        if sample_client_num > 0:
            receiver = [1,2] # only on process 1
        else:
            receiver = [8,9] # only on process 2

        # bug config
        # if sample_client_num > 0:
        #     receiver = [1,2] # only on process 1
        # else:
        #     receiver = [2,9] # on porcess 1 and process 2

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=self.model.state_dict()))
        
def call_test_worker(method):
    if method == 'test_comm':
        worker_builder = {'client': Client, 'server': testServer}
        return worker_builder

register_worker('test_comm', call_test_worker)
