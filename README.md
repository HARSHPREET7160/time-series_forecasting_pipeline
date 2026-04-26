## Time-Series Forecasting Pipeline (Lab Makeup)

Roll number: `102317160`

Derived parameters (visible in `utils/config.py`):

- `window_size = (sum of digits) % 10 + 8`
- `prediction_horizon = (last 2 digits) % 3 + 1`
- `hidden_size = (first 3 digits) % 16 + 8`

Model assignment:

- Last digit is even (`0`) -> **Custom RNN** implemented from scratch in `models/custom_rnn.py`

Datasets used:

- `data/Electric_Production.csv`
- `data/AirPassengers.csv`

Runs (on each dataset) in `train.py`:

- MLP baseline (no recurrence)
- Custom RNN (from scratch, no `nn.RNN/nn.GRU/nn.LSTM`)
- Prebuilt LSTM (comparison)
- Transformer encoder (comparison)

Ablation (Custom RNN) in `ablation.py`:

- half window size
- original window size
- double window size

Note from me: I tried to keep code simple and explainable, so in viva I can point to exact lines
and tell what is input window, what is target, and how model is predicting.
