import torch
from torch.utils.data import Dataset
from einops import rearrange

class PredictorDataset(Dataset):

    def __init__(self, data, pre_seq_length, preprocess=False, p=20):
        super(PredictorDataset, self).__init__()
        self.data = data
        self.pre_seq_length = pre_seq_length
        self.preprocess = preprocess
        self.p = p

    def __getitem__(self, index):
        data = self.data[index][:self.pre_seq_length]
        label = self.data[index][self.pre_seq_length:]
        if self.preprocess:
            data = self._preprocess(data, self.p)
            label = self._preprocess(label, self.p)
        return data, label
    
    def __len__(self):
        return len(self.data)
    
    def _less_than_zero_and_p(self, gradient, p=20):
        # 将a中小于p的元素赋值为-1，其余元素赋值为1
        # mask = torch.where( gradient < (torch.exp(torch.tensor(-p)) * (-1)), -1, 1)
        mask = torch.where(gradient < torch.exp(torch.tensor(-p)), 1, -1)
        return mask

    def _abs_greater_than_p(self, gradient, p=20):
        # 对a中的元素进行处理
        gradient = torch.where(torch.abs(gradient) > torch.exp(torch.tensor(-p)),
                               torch.log(torch.abs(gradient))/p,
                               torch.tensor(-1.))
        return gradient

    def _preprocess(self, gradient, p):
        mask = self._less_than_zero_and_p(gradient, p)
        gradient = self._abs_greater_than_p(gradient, p) * mask
        return gradient

class PredictorData():
      
    def __init__(self, predictor_config, model):
        self.cfg = predictor_config
        self.patch_size = self.cfg.patch_size
        self.topk = self.cfg.topk
        self.trainable_para_names = self._get_trainable_para_names(model)
        self.grad_series = []
        self.gard_norm2_series = []

    def add_train_data(self, model_gradient):
        grad_dict = {}
        nomr2_dict = {}
        for name in self.trainable_para_names:
            grad = model_gradient[name]
            grad = self._split_layer_to_patch(grad)
            grad_dict[name] = grad
            nomr2_dict[name] = [torch.norm(patch_grad) for patch_grad in grad]
        self.grad_series.append(grad_dict)
        self.gard_norm2_series.append(nomr2_dict)

    def update_train_data(self, model_gradient):
        del self.grad_series[0]
        del self.gard_norm2_series[0]
        self.add_train_data(model_gradient)

    def get_train_data(self):
        sum_seq_len = self.cfg.sum_seq_length
        train_data = []
        indices = self._get_topk_patch_index()
        for name in self.trainable_para_names:
            data = torch.stack([grad[name][indices[name]] for grad in self.grad_series], dim=1) # (num, seq_len, k^2, p, p)
            data = rearrange(data, 'n (m s) k p1 p2 -> n m s k p1 p2 ', s=sum_seq_len) # (num, seq_len_r, sum_len, k^2, p, p)
            data = torch.sum(data, dim=2) # (num, seq_len_r, k^2, p, p)
            train_data.extend([*data])
        return train_data
 
    def get_predict_data(self):
        aft_seq_len = self.cfg.aft_seq_length
        sum_seq_len = self.cfg.sum_seq_length
        result = {}
        for name in self.trainable_para_names:
            data = torch.stack([grad[name] for grad in self.grad_series[aft_seq_len:]], dim=1) # (num, seq_len, k^2, p, p)
            data = rearrange(data, 'n (m s) k p1 p2 -> n m s k p1 p2 ', s=sum_seq_len)
            result[name] = torch.sum(data, dim=2)
        return result
    
    def reconstruct_model_gradient(self, data):
        return {name : self._reconstruct_layer_from_patch(grad, self.trainable_para_names[name]) \
                for name, grad in data.items()}

    def _get_trainable_para_names(self, model:torch.nn.Module):
        model_type = self.cfg.model_type
        patch_size = self.cfg.patch_size
        in_channel = self.cfg.in_channel
        trainable_para_names = {}
        for name, parameters in model.named_parameters():
            if model_type == 'cv':
                if len(parameters.shape) == 4:
                    Cout, Cin, kernel_size, _ = parameters.shape
                    if (Cout % patch_size == 0 ) and (Cin % patch_size == 0) and kernel_size**2 == in_channel:
                        if Cout // patch_size <= 4:
                            trainable_para_names[name] = parameters.shape
            elif model_type == 'nlp':
                if len(parameters.shape) == 2 and 'embeddings' not in name:
                    Cout, Cin = parameters.shape
                    if (Cout % patch_size == 0 ) and (Cin % patch_size == 0):
                        trainable_para_names[name] = parameters.shape
        return trainable_para_names
     
    def _split_layer_to_patch(self, grad:torch.Tensor):
        p = self.patch_size
        if self.cfg.model_type == 'cv':
            return rearrange(grad, '(m p1) (n p2) k1 k2 -> (m n) (k1 k2) p1 p2 ', p1=p, p2=p)
        elif self.cfg.model_type == 'nlp':
            return rearrange(grad, '(m p1) (n p2) -> (m n) 1 p1 p2 ', p1=p, p2=p)
    
    def _reconstruct_layer_from_patch(self, result:torch.Tensor, original_shape):
        p = self.patch_size
        m, n = original_shape[0] // p, original_shape[1] // p
        if self.cfg.model_type == 'cv':
            k1, k2 = original_shape[2], original_shape[3]
            return rearrange(result, '(m n) (k1 k2) p1 p2 -> (m p1) (n p2) k1 k2', m=m, n=n, k1=k1, k2=k2, p1=p, p2=p)
        elif self.cfg.model_type == 'nlp':
            return rearrange(result, '(m n) 1 p1 p2 -> (m p1) (n p2)', m=m, n=n, p1=p, p2=p)

    def _calculate_series_norm2_sum(self):
        result_dict = {}
        trainable_para_names = self.gard_norm2_series[0].keys()
        for para in trainable_para_names:
            norm2_sum = [sum(items) for items in zip(*[d[para] for d in self.gard_norm2_series])]
            result_dict[para] = norm2_sum
        return result_dict
    
    def _get_topk_patch_index(self):
        gard_norm2_sum = self._calculate_series_norm2_sum()
        all_values = [value for sublist in gard_norm2_sum.values() for value in sublist]
        sorted_values = sorted(all_values, reverse=True)
        top_k_range = int(len(sorted_values) * self.topk)
        result = {}
        for key, values in gard_norm2_sum.items():
            indices = [index for index, value in enumerate(values) if value in sorted_values[:top_k_range]]
            result[key] = indices
        return result
    
    def __len__(self):
        return len(self.grad_series)

