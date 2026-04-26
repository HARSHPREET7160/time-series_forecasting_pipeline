import torch
import torch.nn as nn

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()

        self.hidden_size = hidden_size

        # I implement the recurrence manually instead of using nn.RNN/nn.GRU/nn.LSTM:
        # h_t = tanh(Wx * x_t + Wh * h_{t-1}) this is the equation we studied
        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
)

    def forward(self, x):
        # x comes as (batch, seq_len) where each element is one scalar from time series.
        batch_size, seq_len = x.shape
        # Start with an "empty memory" hidden state; it will accumulate information across the window.
        h = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            x_t = x[:, t].unsqueeze(1)
            # Here is the main recurrence:
            # current hidden = f(current input + previous hidden)
            h = torch.tanh(self.Wx(x_t) + self.Wh(h))

        # mapping hidden state to final prediction.
        return self.fc(h)
