from federatedscope.spec.predictor.model.model import Predictor_Model
from federatedscope.spec.predictor import GradPredictor

def get_predictor(config, send_channel, recv_channel, data_channel):

    if config.spec.predictor.type == 'default':
        model_args = config.spec.predictor.model_args[0]
        model = Predictor_Model(**model_args)
        return GradPredictor(model=model, config=config,
                              send_channel=send_channel,
                              recv_channel=recv_channel,
                              data_channel=data_channel)
    else:
        raise TypeError

