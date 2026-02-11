import torch
from federatedscope.core.aggregators import ClientsAvgAggregator

class FedAsyncAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model, device, config)

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        staleness = [x[1]
                     for x in agg_info['staleness']]  # (client_id, staleness)
        avg_model = self._para_weighted_avg(models,
                                            recover_fun=recover_fun,
                                            staleness=staleness)
        return avg_model

    def discount_func(self, staleness):
        return (1.0 /(1.0 + staleness)) * self.cfg.asyn.staleness_discount_factor

    def _para_weighted_avg(self, models, recover_fun=None, staleness=None):

        avg_model = self.model.state_dict()
        for key in avg_model:
            for i in range(len(models)):
                _, local_model = models[i]

                assert staleness is not None
                weight = self.discount_func(staleness[i])
                if isinstance(local_model[key], torch.Tensor):
                    local_model[key] = local_model[key].float()
                else:
                    local_model[key] = torch.FloatTensor(local_model[key])

                avg_model[key] = (1-weight) * avg_model[key] + weight * local_model[key]

        return avg_model