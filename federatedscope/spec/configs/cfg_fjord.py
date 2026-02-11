from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_fjord_cfg(cfg):
    # for fjord
    cfg.fjord = CN()
    cfg.fjord.use = False

    cfg.fjord.p_s = [0.2, 0.4, 0.6, 0.8, 1.0]
    cfg.fjord.client_tier_allocation = 'uniform'
    cfg.fjord.know_distill = False

    cfg.register_cfg_check_fun(assert_fjord_cfg)

def assert_fjord_cfg(cfg):
    if not cfg.fjord.use:
        return True
    
    cfg.model.p_s = cfg.fjord.p_s
    
    
register_config("fjord", extend_fjord_cfg)