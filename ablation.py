import torch
import torch.nn as nn

from models.custom_rnn import CustomRNN
from utils.config import ROLL_NUMBER, window_size, prediction_horizon, hidden_size
from utils.data_loader import load_data
from utils.metrics import calculate_metrics


device = torch.device("cpu")
# I keep cpu here also, because ablation is more runs and I dont want any gpu dependency.


def run_ablation(path, title):
    # This file is only for ablation: I keep model same, and only change window_size.
    # Idea is: window size show how much past info I give, and it directly affect forecasting.
    print(f"\n========== ABLATION: {title} ==========\n")
    print(f"ROLL_NUMBER = {ROLL_NUMBER}")
    print(f"window_size = {window_size}, prediction_horizon = {prediction_horizon}, hidden_size = {hidden_size}\n")

    window_sizes = [max(1, window_size // 2), window_size, window_size * 2]

    for w in window_sizes:
        print(f"\n---- window_size = {w} ----\n")

        # Same dataset, same normalization, same chronological split.
        # Only window length change, so I can compare performance fairly.
        X_train, y_train, X_test, y_test, mean, std = load_data(
            path, window_size=w, horizon=prediction_horizon
        )
        print(f"Windowing shapes: X_train={tuple(X_train.shape)}, y_train={tuple(y_train.shape)}")
        print(f"Example window->target: X[0][:5]={X_train[0, :5].tolist()} ... y[0]={y_train[0].tolist()}\n")

        model = CustomRNN(1, hidden_size, output_size=prediction_horizon).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        loss_fn = nn.MSELoss()

        # Simple training loop; the point here is to compare the effect of window size.
        for _epoch in range(15):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = loss_fn(preds, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test)

        # Denormalize back to original units for meaningful metrics.
        preds_np = preds.numpy() * std + mean
        y_true_np = y_test.numpy() * std + mean

        # I report same metrics as main training: MSE, MAE, RMSE.
        mse, mae, rmse = calculate_metrics(y_true_np, preds_np)
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")


run_ablation("data/Electric_Production.csv", "ELECTRICITY DATASET")
run_ablation("data/AirPassengers.csv", "AIR PASSENGERS DATASET")
