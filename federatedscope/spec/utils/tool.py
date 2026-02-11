import os
import torch
import shutil
import numpy as np

def gradient_operator(func, model_a, b):
    with torch.no_grad():
        if isinstance(b, dict):
            return {key : func(model_a[key], b[key]) for key in model_a}
        else:
            return {key : func(model_a[key], b) for key in model_a}

def save_config(src_path, des_dir):
    file_name = os.path.basename(src_path)
    shutil.copy(src_path, os.path.join(des_dir, file_name))


def get_top_k_indices(array, index_list, k):
    """
    返回数组中对应于给定索引列表中前k大元素的原始索引。
    """
    # 提取索引列表中的元素
    elements = array[index_list]

    # 使用稳定排序算法找到前k大元素的索引
    top_k_indices = np.argsort(-elements, kind='mergesort')[:k]

    # 获取这些元素在原始数组中的索引
    original_indices = [index_list[i] for i in top_k_indices]

    return original_indices