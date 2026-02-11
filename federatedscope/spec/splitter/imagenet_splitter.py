import numpy as np
from federatedscope.core.splitters import BaseSplitter
from federatedscope.register import register_splitter
from federatedscope.spec.splitter.utils import \
    dirichlet_distribution_noniid_slice, split_according_to_prior, split_list_by_normal_distribution

class LDASplitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=0.5):
        self.alpha = alpha
        super(LDASplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        # tmp_dataset = [ds for ds in dataset.targets]
        # label = np.array([y for x, y in tmp_dataset])
        label = np.array(dataset.targets)
        if prior is None:
            idx_slice, label_slice = dirichlet_distribution_noniid_slice(label, 
                                                                         self.client_num,
                                                                         self.alpha,
                                                                         prior=prior)
        else:
            idx_slice = split_according_to_prior(label, self.client_num, prior)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        if prior is None:
            return data_list, label_slice
        else:
            return data_list


def call_imagenet_lda_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'imagenet_lda':
        splitter = LDASplitter(client_num, **kwargs)
        return splitter

register_splitter('imagenet_lda', call_imagenet_lda_splitter)