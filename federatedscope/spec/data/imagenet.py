import logging
from torchvision.datasets.imagenet import ImageNet
from torchvision.transforms import transforms
from federatedscope.register import register_data
from federatedscope.core.data import BaseDataTranslator
from federatedscope.core.data import ClientData, StandaloneDataDict

logger = logging.getLogger(__name__)

def get_imagenet_transform():

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, val_transforms

class ImageNetTranslator(BaseDataTranslator):

    def __init__(self, global_cfg, client_cfgs=None):
        super().__init__(global_cfg, client_cfgs)

    def __call__(self, train_dataset, val_dataset, test_dataset):
        datadict = self.split(train_dataset, val_dataset, test_dataset)
        datadict = StandaloneDataDict(datadict, self.global_cfg)

        return datadict
        
    def split(self, train, val, test):
        """
        Perform ML split and FL split.

        Returns:
            dict of ``ClientData`` with client_idx as key to build \
            ``StandaloneDataDict``
        """
        datadict = self.split_to_client(train, val, test)
        return datadict
    
    def split_to_client(self, train, val, test):
        """
        Split dataset to clients and build ``ClientData``.

        Returns:
            dict: dict of ``ClientData`` with ``client_idx`` as key.
        """

        # Initialization
        client_num = self.global_cfg.federate.client_num
        split_train, split_val, split_test = [[None] * client_num] * 3
        train_label_distribution = None

        # Split train/val/test to client
        if len(train) > 0:
            split_train, train_label_distribution = self.splitter(train)
        if len(val) > 0:
            split_val = self.splitter(val, prior=train_label_distribution)
        if len(test) > 0:
            split_test = self.splitter(test, prior=train_label_distribution)

        # Build data dict with `ClientData`, key `0` for server.
        # data_dict = {
        #     0: ClientData(self.global_cfg, train=train, val=val, test=test)
        # }
        data_dict = {}
        for client_id in range(1, client_num + 1):
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg
            data_dict[client_id] = ClientData(client_cfg,
                                              train=split_train[client_id - 1],
                                              val=split_val[client_id - 1],
                                              test=split_test[client_id - 1])
        return data_dict

def load_imagenet_dt(config, client_cfgs=None):
    file_dir = config.data.root
    train_transforms, val_transforms = get_imagenet_transform()
    train_dataset = ImageNet(file_dir, split='train', transform=train_transforms)
    val_dataset = ImageNet(file_dir, split='val', transform=val_transforms)
    test_dataset = val_dataset
    translator = ImageNetTranslator(config, client_cfgs)
    return translator(train_dataset, val_dataset, test_dataset), config

def call_imagenet_dt(config, client_cfgs):
    if config.data.type == "imagenet":
        data, modified_config = load_imagenet_dt(config, client_cfgs)
        return data, modified_config


register_data("imagenet_dt", call_imagenet_dt)