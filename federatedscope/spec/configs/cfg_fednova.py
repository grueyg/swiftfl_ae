from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_fednova_cfg(cfg):
    # for FedNova
    cfg.fednova = CN()
    cfg.fednova.use = False

    cfg.register_cfg_check_fun(assert_fednova_cfg)

def assert_fednova_cfg(cfg):
    if not cfg.fednova.use:
        return True
    
    
register_config("fednova", extend_fednova_cfg)