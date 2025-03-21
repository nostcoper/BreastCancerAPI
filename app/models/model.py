
import torch
import torch.nn as nn
from torchvision import models

class _ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(_ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.features[-2:].parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.features(x)
        return features.mean([2, 3])

class _AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(_AttentionModule, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        scores = self.attention(x) 
        weights = torch.softmax(scores, dim=1)
        context = (x * weights).sum(dim=1)
        return context

class _LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(_LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.45)
        self.attention = _AttentionModule(hidden_size)
        self.classifier = nn.Sequential(
                                        nn.Linear(hidden_size * 2, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(256, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        output = self.classifier(context)
        return output

class SequenceClassificationModel(nn.Module):
    def __init__(self, lstm_hidden_size=512, lstm_num_layers=2):
        super(SequenceClassificationModel, self).__init__()
        self.feature_extractor = _ResNetFeatureExtractor()
        self.sequence_classifier = _LSTMClassifier(input_size=512, 
                                                  hidden_size=lstm_hidden_size,
                                                  num_layers=lstm_num_layers)
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)  
        features = features.view(batch_size, seq_len, -1)
        output = self.sequence_classifier(features)
        return output