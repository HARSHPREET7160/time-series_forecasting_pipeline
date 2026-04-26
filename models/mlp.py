import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, window_size, output_size=1):
        super().__init__()
        # This is baseline: I flatten the window and predict, no memory and no recurrence.
        # It is like normal regression on last windowsize values.
        self.net = nn.Sequential(
            nn.Linear(window_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x shape is (batch, seq_len). For MLP I just treat seq_len as features.
        x = x.view(x.size(0), -1)
        return self.net(x)
