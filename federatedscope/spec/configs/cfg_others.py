from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_others_cfg(cfg):

    cfg.federate.augmentation_factor = 1.0
    cfg.eval.start_interval_eval_round = 0
    cfg.model.norm_layer = ''

    cfg.register_cfg_check_fun(assert_extend_others_cfg)

def assert_extend_others_cfg(cfg):
    pass


register_config("spec_others", extend_others_cfg)

