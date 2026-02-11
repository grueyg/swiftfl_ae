from federatedscope.register import register_model
from federatedscope.spec.exp.fjord.models import ResNet18, ConvNet2, LSTM

def call_od_models(model_config, input_shape):
    model_type = model_config.type.lower()
    if model_type == 'resnet18_od':
        model = ResNet18(od=True, p_s=model_config.p_s)
    elif model_type == 'convnet2_od':
        model = ConvNet2(od=True, p_s=model_config.p_s,
                         in_channels=input_shape[-3],
                         h=input_shape[-2],
                         w=input_shape[-1],
                         hidden=model_config.hidden,
                         class_num=model_config.out_channels,
                         dropout=model_config.dropout)
    elif model_type == 'lstm_od':
        model = LSTM(od=True, p_s=model_config.p_s,
                     in_channels=model_config.in_channels,
                     hidden=model_config.hidden,
                     out_channels=model_config.out_channels,
                     dropout=model_config.dropout)
    else:
        model = None
    return model
    
register_model('od_models', call_od_models)