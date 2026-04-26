import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, hidden_size, output_size=1):
        super().__init__()

        # This is prebuilt LSTM (comparison model). I am NOT using it as my custom model.
        # LSTM have gates so it can keep info longer than basic RNN sometimes.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM expects (batch, seq_len, features), so i unsqueeze to make feature=1.
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # it use the last time step output to predict next horizon.
        return self.fc(out[:, -1, :])
