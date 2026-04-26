import torch
import torch.nn as nn
from models.mlp import MLP
from models.custom_rnn import CustomRNN
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from utils.data_loader import load_data
from utils.metrics import calculate_metrics
from utils.plots import plot_loss, plot_predictions
from utils.config import ROLL_NUMBER, window_size, prediction_horizon, hidden_size

# I made this file as the main "runner" of my full pipeline.
# My intention is: same preprocessing + same split style, then compare multiple models fairly.
# I am keeping it simple and readable because in viva I need to explain step by step.

device = torch.device("cpu")
# I keep cpu because my code should run on any machine, and for these small datasets cpu is enough.

def run_pipeline(path, title):

    print(f"\n========== {title} ==========\n")
    print(f"ROLL_NUMBER = {ROLL_NUMBER}")
    print(f"window_size = {window_size}, prediction_horizon = {prediction_horizon}, hidden_size = {hidden_size}\n")

    # I do chronological split inside load_data (not random), because time series should not leak future info.
    # Also I normalize using train statistics style (here full-series mean/std for simplicity, but no shuffle).
    X_train, y_train, X_test, y_test, mean, std = load_data(path, window_size=window_size, horizon=prediction_horizon)
    print(f"Windowing shapes: X_train={tuple(X_train.shape)}, y_train={tuple(y_train.shape)}")
    # I print small example so it is very clear what is one input window and what is target horizon.
    print(f"Example window->target: X[0][:5]={X_train[0, :5].tolist()} ... y[0]={y_train[0].tolist()}\n")
    # Note: X values are normalized (z-score). Model always see normalized numbers, then I denormalize outputs later.

    models = {
        # MLP: it only see fixed window as features, it dont have memory across time steps.
        "MLP": MLP(window_size, output_size=prediction_horizon),
        # Custom_RNN: my manual recurrence, I am not using nn.RNN/nn.GRU/nn.LSTM here.
        "Custom_RNN": CustomRNN(1, hidden_size, output_size=prediction_horizon),
        # LSTM and Transformer are prebuilt, I am using them for comparison only.
        "LSTM": LSTMModel(hidden_size, output_size=prediction_horizon),
        "Transformer": TransformerModel(hidden_size, output_size=prediction_horizon)
    }

    for name, model in models.items():

        if name == "LSTM":
            epochs = 40
        else:
            epochs = 20

        print(f"\nTraining {name}...\n")

        # I change lr a bit because different models train at different speed/stability on same data.
        if name == "LSTM":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        elif name == "Custom_RNN":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # I use MSE for training objective because it is standard regression loss for forecasting.
        loss_fn = nn.MSELoss()
        losses = []

        batch_size = 32
        # I use mini-batch training so it is not too slow and also gradients are more stable.

        for epoch in range(epochs):

            model.train()
            epoch_loss = 0

            for i in range(0, len(X_train), batch_size):

                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()

                preds = model(batch_X)

                # I print this only once, so sir can see what exactly I am feeding to model
                # and what model is returning (prediction). This is like "proof" of input-output flow.
                if epoch == 0 and i == 0:
                    print(f"{name} input batch shape = {tuple(batch_X.shape)} (each row is one window)")
                    print(f"{name} output preds shape = {tuple(preds.shape)} (next {prediction_horizon} value(s))\n")

                loss = loss_fn(preds, batch_y)

                loss.backward()
                # I clip gradients because RNN type models can explode gradient sometimes.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (len(X_train) // batch_size + 1)
            losses.append(epoch_loss)

            print(f"{name} Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

        # I plot training loss so I can see if it is learning or just stuck/overfitting.
        plot_loss(losses, f"{name} Training Loss")

        model.eval()
        with torch.no_grad():

            preds = model(X_test)
            # Here predictions are generated: I just do forward pass on test windows.
            # preds is still in normalized scale until I multiply by std and add mean.

            # Denormalize because I want metrics in original units (not z-score scale).
            preds_np = preds.numpy() * std + mean
            y_true_np = y_test.numpy() * std + mean

            # Quick small peek (not all) so I can see first few predicted vs actual values.
            print(f"{name} first pred/true: pred={preds_np[0].tolist()} true={y_true_np[0].tolist()}")

            mse, mae, rmse = calculate_metrics(y_true_np, preds_np)

            print(f"\n{name} Results:")
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")

            # This plot is the real proof: where my model fail and where it match.
            plot_predictions(y_true_np, preds_np, f"{name} Predictions")


run_pipeline("data/Electric_Production.csv", "ELECTRICITY DATASET")
run_pipeline("data/AirPassengers.csv", "AIR PASSENGERS DATASET")
