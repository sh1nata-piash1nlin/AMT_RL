import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class ConvNet(nn.Module):
    def __init__(self, hparams):
        super(ConvNet, self).__init__()
        self.convs = nn.ModuleList()
        in_channels = 1  # Spectrogram input has 1 channel
        for i in range(len(hparams['num_filters'])):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hparams['num_filters'][i], 
                              kernel_size=(hparams['temporal_sizes'][i], hparams['freq_sizes'][i]), 
                              padding='same'),
                    nn.BatchNorm2d(hparams['num_filters'][i]),
                    nn.ReLU(),
                    nn.MaxPool2d((1, hparams['pool_sizes'][i])) if hparams['pool_sizes'][i] > 1 else nn.Identity(),
                    nn.Dropout(1 - hparams['dropout_keep_amts'][i])
                )
            )
            in_channels = hparams['num_filters'][i]
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * 229, hparams['fc_size'])
        self.dropout = nn.Dropout(1 - hparams['fc_dropout_keep_amt'])
        
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
    def forward(self, x, lengths=None):
        x, _ = self.lstm(x)
        return x

class AcousticModel(nn.Module):
    def __init__(self, hparams):
        super(AcousticModel, self).__init__()
        self.convnet = ConvNet(hparams)
        self.lstm = BiLSTM(hparams['fc_size'], hparams['onset_lstm_units'])
        self.fc = nn.Linear(hparams['onset_lstm_units'] * 2, 88)  # 88 piano keys

    def forward(self, x, lengths=None):
        x = self.convnet(x)
        x = self.lstm(x, lengths)
        x = torch.sigmoid(self.fc(x))
        return x

class PianoTranscriptionModel(nn.Module):
    def __init__(self, hparams):
        super(PianoTranscriptionModel, self).__init__()
        self.onset_model = AcousticModel(hparams)
        self.frame_model = AcousticModel(hparams)
        self.lstm = BiLSTM(hparams['fc_size'] + hparams['onset_lstm_units'] * 2, hparams['combined_lstm_units'])
        self.fc = nn.Linear(hparams['combined_lstm_units'] * 2, 88)
        
    def forward(self, x, lengths=None):
        onset_probs = self.onset_model(x, lengths)
        frame_probs = self.frame_model(x, lengths)
        combined = torch.cat((onset_probs, frame_probs), dim=-1)
        combined = self.lstm(combined, lengths)
        frame_predictions = torch.sigmoid(self.fc(combined))
        return onset_probs, frame_predictions

# Example hyperparameters
def get_hparams():
    return {
        'onset_lstm_units': 128,
        'frame_lstm_units': 128,
        'combined_lstm_units': 128,
        'fc_size': 768,
        'num_filters': [48, 48, 96],
        'temporal_sizes': [3, 3, 3],
        'freq_sizes': [3, 3, 3],
        'pool_sizes': [1, 2, 2],
        'dropout_keep_amts': [1.0, 0.75, 0.75],
        'fc_dropout_keep_amt': 0.5
    }

# Example usage
hparams = get_hparams()
model = PianoTranscriptionModel(hparams)
x = torch.randn(8, 1, 1000, 229)  # (batch_size, channels, time, freq_bins)
onset_probs, frame_probs = model(x)
