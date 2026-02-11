import pickle
import os
import sys

def save_split_lists_by_size(data_dict, max_file_size, file_prefix):
    for key, value in data_dict.items():
        total_size = 0
        chunk_size = 0
        chunk_num = 0
        sub_list = []

        for item in value:
            sub_list.append(item)
            chunk_size += 1
            total_size += sys.getsizeof(pickle.dumps(sub_list)) / (1024 * 1024)

            if total_size >= max_file_size:
                print(len(sub_list))
                save_file_path = f'{file_prefix}_{key}_{chunk_num}.pkl'
                with open(save_file_path, 'wb') as f:
                    pickle.dump(sub_list, f)
                chunk_num += 1
                sub_list = []
                total_size = 0
                chunk_size = 0

        if chunk_size > 0:
            save_file_path = f'{file_prefix}_{key}_{chunk_num}.pkl'
            with open(save_file_path, 'wb') as f:
                pickle.dump(sub_list, f)

def load_merged_lists(file_prefix):
    merged_dict = {}

    file_list = [f for f in os.listdir() if f.startswith(file_prefix) and f.endswith('.pkl')]
    file_list.sort()

    for file_name in file_list:
        key = file_name.split('_')[1]  # 获取文件名中的键名
        with open(file_name, 'rb') as f:
            sub_list = pickle.load(f)
        merged_dict[key] = sub_list

    return merged_dict
