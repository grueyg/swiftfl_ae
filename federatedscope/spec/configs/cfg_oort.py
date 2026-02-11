from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_oort_cfg(cfg):
    # for Oort
    cfg.oort = CN()
    cfg.oort.use = False
    cfg.oort.pacer_delta = 5
    cfg.oort.pacer_step = 20
    cfg.oort.exploration_alpha = 0.3
    cfg.oort.exploration_factor = 0.9
    cfg.oort.exploration_decay = 0.98
    cfg.oort.exploration_min = 0.2
    cfg.oort.sample_window = 5.0
    cfg.oort.round_threshold = 10
    cfg.oort.blacklist_rounds = -1
    cfg.oort.clip_bound = 0.98
    cfg.oort.round_penalty = 2.0
    cfg.oort.cut_off_util = 0.7

    cfg.oort.capacity_bin = True
    cfg.oort.enable_adapt_local_epoch = True

    cfg.oort.overcommit = 1.3
    
    cfg.register_cfg_check_fun(assert_oort_cfg)

def assert_oort_cfg(cfg):
    if not cfg.oort.use:
        return True
    
    
register_config("oort", extend_oort_cfg)