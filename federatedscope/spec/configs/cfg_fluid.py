from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_fluid_cfg(cfg):
    # for fluid
    cfg.fluid = CN()
    cfg.fluid.use = False

    cfg.fluid.p_val = 0.95

    cfg.register_cfg_check_fun(assert_fluid_cfg)

def assert_fluid_cfg(cfg):
    if not cfg.fluid.use:
        return True
    
    
register_config("fluid", extend_fluid_cfg)