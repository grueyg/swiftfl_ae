from federatedscope.core.aggregators import ClientsAvgAggregator

class SpecAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model, device, config)

    def aggregate(self, agg_info):
        aggregated_gradient =  super().aggregate(agg_info)
        models = agg_info["client_feedback"]
        training_set_size = sum([sample_size for sample_size, _ in models])
        return aggregated_gradient, training_set_size
