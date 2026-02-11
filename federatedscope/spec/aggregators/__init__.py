from federatedscope.register import register_aggregator
from federatedscope.spec.aggregators.spec_aggregator import SpecAggregator
from federatedscope.spec.aggregators.async_aggregator import FedAsyncAggregator
from federatedscope.spec.exp.fednova.fednova_aggregator import FedNovaAggregator
from federatedscope.spec.exp.scaffold.scaffold_aggregator import ScaffoldAggregator
from federatedscope.spec.exp.fjord.fjord_aggregator import FjORDAggregator
from federatedscope.spec.exp.fluid.fluid_aggregator import FLuIDggregator

def call_spec_aggregator(model, device, config):
    if config.spec.use:
        return SpecAggregator(model=model, device=device, config=config)
    if config.federate.method.lower() == 'fedasync':
        return FedAsyncAggregator(model=model, device=device, config=config)
    if config.federate.method.lower() == 'fednova':
        return FedNovaAggregator(model=model, device=device, config=config)
    if config.federate.method.lower() == 'scaffold':
        return ScaffoldAggregator(model=model, device=device, config=config)
    if config.federate.method.lower() == 'fjord':
        return FjORDAggregator(model=model, device=device, config=config)
    if config.federate.method.lower() == 'fluid':
        return FLuIDggregator(model=model, device=device, config=config)
    return None

register_aggregator('spec_aggregator', call_spec_aggregator)

__all__ = ['SpecAggregator', 'FedAsyncAggregator', 'FedNovaAggregator', 'ScaffoldAggregator']