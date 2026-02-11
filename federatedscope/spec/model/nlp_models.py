from transformers import AlbertConfig, AlbertForSequenceClassification
from transformers import MobileBertConfig, MobileBertForSequenceClassification
from federatedscope.register import register_model
import torch.nn as nn
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, hidden, out_channels, embed_size, n_layers=2, dropout=0.0, **kwargs):
        super(LSTMClassifier, self).__init__()
        self.hidden = hidden
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

        self.decoder = nn.Linear(hidden, out_channels)

    def forward(self, input_):
        output, (hidden_state, cell_state) = self.rnn(input_)

        last_output = output[:, -1, :]
        
        class_scores = self.decoder(last_output)
        
        return class_scores


def call_nlp_models(model_config, local_data):
    if model_config.type.lower() == 'albert-base-v2':
        model = AlbertForSequenceClassification.from_pretrained(
            '/public/home/checheng/data/pretrained_weights/albert-base-v2', num_labels=model_config.out_channels)
        return model
    elif model_config.type.lower() == 'mobilebert-uncased':
        model = MobileBertForSequenceClassification.from_pretrained(
            '/public/home/checheng/data/pretrained_weights/google/mobilebert-uncased', num_labels=model_config.out_channels
        )
        return model
    elif model_config.type.lower() == 'mobilebert-tiny':
        model = MobileBertForSequenceClassification.from_pretrained(
          '/public/home/checheng/data/pretrained_weights/ybelkada/tiny-mobilebertmodel', num_labels=model_config.out_channels
        )
        return model
    elif model_config.type.lower() == 'lstm_classifier':
        model = LSTMClassifier(hidden=model_config.hidden,
                               out_channels=model_config.out_channels,
                               embed_size=model_config.embed_size,
                               dropout=model_config.dropout)
        return model
        
    
register_model('nlp_models', call_nlp_models)




