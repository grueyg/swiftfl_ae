import os
import logging
import pickle
from datasets import load_from_disk
from transformers import AlbertTokenizerFast, MobileBertTokenizerFast
from federatedscope.register import register_data
from federatedscope.core.data import BaseDataTranslator
from federatedscope.core.data import ClientData, StandaloneDataDict
logger = logging.getLogger(__name__)

class qqpTranslator(BaseDataTranslator):
    def __init__(self, global_cfg, client_cfgs=None):
        super().__init__(global_cfg, client_cfgs)

    def split(self, datadict):
        train, val= datadict['train'], datadict['val']
        datadict = self.split_to_client(train, val, val)
        datadict.pop(0, None)
        return datadict
    
def load_tokenizer(model_type: str):
    if 'albert' in model_type.lower():
        return AlbertTokenizerFast.from_pretrained(
            '/public/home/checheng/data/pretrained_weights/albert-base-v2', local_files_only=True)
    elif 'mobilebert' in model_type.lower():
        return MobileBertTokenizerFast.from_pretrained(
            '/public/home/checheng/data/pretrained_weights/google/mobilebert-uncased', local_files_only=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def get_qqp_dataset(config):
    if config.data.args:
        raw_args = config.data.args[0]
    else:
        raw_args = {}
    assert 'max_len' in raw_args, "Miss key 'max_len' in " \
                                    "`config.data.args`."
    # if os.path.exists(raw_args['cache']):
    #     data_split_dict = pickle.load(raw_args['cache'])
    #     return data_split_dict
    
    dataset = load_from_disk(config.data.root)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("To load huggingface tokenizer")
    tokenizer = load_tokenizer(config.model.type)

    for split in dataset:
        if split == 'test': continue
        x_all = [[i['text1'] for i in dataset[split]], [i['text2'] for i in dataset[split]]]
        targets = [i['label'] for i in dataset[split]]

        x_all = tokenizer(text=x_all[0],
                          text_pair=x_all[1],
                          return_tensors='pt',
                          padding=True,
                          truncation=True,
                          max_length=raw_args['max_len'])
        data = [{key: value[i]
                    for key, value in x_all.items()}
                for i in range(len(next(iter(x_all.values()))))]
        dataset[split] = (data, targets)

    data_split_dict = {
        'train': [(x, y)
                    for x, y in zip(dataset['train'][0], dataset['train'][1])
                    ],
        'val': [(x, y) for x, y in zip(dataset['validation'][0],
                                        dataset['validation'][1])],
    }

    # with open(f'data/qqp/cache/data_split_tuple.pkl', 'wb') as f:
    #     pickle.dump(data_split_dict, f)
    return data_split_dict

def load_qqp_dataset(config, client_cfgs=None):
    data_split_dict = get_qqp_dataset(config)
    translator = qqpTranslator(config)
    return translator(data_split_dict), config

def call_qqp_dt(config, client_cfgs):
    if config.data.type == "qqp":
        data, modified_config = load_qqp_dataset(config, client_cfgs)
        return data, modified_config

register_data("qqp_dt", call_qqp_dt)