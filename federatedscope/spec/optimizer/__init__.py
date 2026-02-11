from federatedscope.register import register_optimizer
from federatedscope.spec.exp.fednova.fednova_optim import FedNovaOptimizer
from federatedscope.spec.exp.scaffold.scaffold_optim import SCAFFOLDOptimizer
from federatedscope.spec.optimizer.fedopt import AdaptiveOptimizer
from federatedscope.spec.optimizer.fedyogi import Yogi

def call_spec_optimizer(model, type, lr, **kwargs):
    if type.lower() == 'fednova':
        optimizer = FedNovaOptimizer(model.parameters(),
                                     lr=lr,
                                     **kwargs)
    elif type.lower() == 'scaffold':
        optimizer = SCAFFOLDOptimizer(model.parameters(),
                                      lr=lr,
                                      **kwargs)
    # elif type.lower() in ['adagrad', 'yogi', 'adam']:
    #     optimizer = AdaptiveOptimizer(type, model, lr=lr, **kwargs)
    elif type.lower() == 'yogi':
        optimizer = Yogi(model.parameters(), lr=lr, **kwargs)
    else:
        return None
    return optimizer

register_optimizer('spec', call_spec_optimizer)