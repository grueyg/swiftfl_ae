from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_scaffold_cfg(cfg):
    # for scaffold
    cfg.scaffold = CN()
    cfg.scaffold.use = False
    cfg.scaffold.server_learning_rate = 1.0

    cfg.register_cfg_check_fun(assert_scaffold_cfg)

def assert_scaffold_cfg(cfg):
    if not cfg.scaffold.use:
        return True
    
    
register_config("scaffold", extend_scaffold_cfg)