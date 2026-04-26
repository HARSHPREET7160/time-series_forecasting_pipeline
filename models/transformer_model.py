import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, hidden_size, output_size=1):
        super().__init__()

        # I use transformer encoder just for comparison (it is prebuilt).
        # It can attend to whole window at once, unlike simple RNN that process step by step.
        self.embedding = nn.Linear(1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_size,
            nhead=1,
            batch_first=True
)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input come as (batch, seq_len). Transformer wants features, so I add last dim as 1.
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        # After transformer, I take last token representation to predict next horizon.
        x = self.transformer(x)
        return self.fc(x[:, -1, :])
