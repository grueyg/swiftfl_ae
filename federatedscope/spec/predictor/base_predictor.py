import abc
import os
import logging

class BasePredictor(abc.ABC):
    def __init__(self, model, config, **kwargs):
        self.model = model
        self.outdir = config.outdir
        self.server_parallel = config.federate.process_num > 1
        self.kwargs = kwargs

    @abc.abstractmethod
    def train(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def _train_epoch(self):
        raise NotImplementedError
    
    def _setup_logger(self, name):
        predictor_logger = logging.getLogger(name)
        predictor_logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        predictor_logger.addHandler(console_handler)

        # create file handler which logs even debug messages
        # file_handler  = logging.FileHandler(os.path.join(self.outdir, name + '_training.log'), mode='a')
        # file_handler.setLevel(logging.DEBUG)
        # logger_formatter = logging.Formatter(
        #     "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        # file_handler .setFormatter(logger_formatter)
        # predictor_logger.addHandler(file_handler)
        return predictor_logger
