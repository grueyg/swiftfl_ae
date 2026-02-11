from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torchvision.models import mobilenet_v2, shufflenet_v2_x0_5
from federatedscope.register import register_model

resnet = {'resnet18': resnet18,
          'resnet34': resnet34,
          'resnet50': resnet50,
          'resnet101': resnet101}

class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2, num_channels=num_channels,
                                 eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x

def call_cv_models(model_config, local_data):
    model_type = model_config.type.lower()
    if model_type == 'mobilenet_v2':
        model = mobilenet_v2(num_classes=model_config.out_channels)
    elif model_type == 'shufflenet_v2_x0_5':
        model = shufflenet_v2_x0_5(num_classes=model_config.out_channels)
    elif model_type in resnet.keys():
        if model_config.norm_layer.lower() == 'group_norm':
            model = resnet[model_type](num_classes=model_config.out_channels, norm_layer=MyGroupNorm)
        else:
            model = resnet[model_type](num_classes=model_config.out_channels)
    else:
        model = None
    return model
    
register_model('cv_models', call_cv_models)