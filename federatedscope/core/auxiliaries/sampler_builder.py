import logging

from federatedscope.core.sampler import UniformSampler, GroupSampler, \
    ResponsivenessRealtedSampler
from federatedscope.spec.sampler import SpecOptimalSampler, IdpSampler, \
    SpecOptTimeSampler
from federatedscope.spec.exp.oort.oort_sampler import OortSampler

logger = logging.getLogger(__name__)


def get_sampler(sample_strategy='uniform',
                client_num=None,
                client_info=None,
                bins=10,
                cfg=None):
    """
    This function builds a sampler for sampling clients who should join the \
    aggregation per communication round.

    Args:
        sample_strategy: Sampling strategy of sampler
        client_num: total number of client joining the FL course
        client_info: client information
        bins: size of bins for group sampler

    Returns:
        An instantiated Sampler to sample during aggregation.

    Note:
      The key-value pairs of built-in sampler and source are shown below:
        ===================================  ==============================
        Sampling strategy                    Source
        ===================================  ==============================
        ``uniform``                          ``core.sampler.UniformSampler``
        ``group``                            ``core.sampler.GroupSampler``
        ===================================  ==============================
    """
    if sample_strategy == 'uniform':
        return UniformSampler(client_num=client_num)
    elif sample_strategy == 'responsiveness':
        return ResponsivenessRealtedSampler(client_num=client_num,
                                            client_info=client_info)
    elif sample_strategy == 'group':
        return GroupSampler(client_num=client_num,
                            client_info=client_info,
                            bins=bins)
    elif sample_strategy == 'oort':
        return OortSampler(client_num=client_num,
                           client_info = client_info,
                           cfg=cfg)
    elif sample_strategy == 'spec':
        return SpecOptimalSampler(client_num=client_num,
                                  client_info=client_info,
                                  cfg=cfg)
    elif sample_strategy == 'idp':
        return IdpSampler(client_num=client_num,
                          client_info=client_info,
                          cfg=cfg)
    elif sample_strategy == 'spec_opt':
        return SpecOptTimeSampler(client_num=client_num,
                                  client_info=client_info,
                                  cfg=cfg)
    else:
        raise ValueError(
            f"The sample strategy {sample_strategy} has not been provided.")