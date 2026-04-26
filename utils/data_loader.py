import pandas as pd
import torch
from utils.windowing import create_sequences

def load_data(path, window_size=None, horizon=None):
    # I made one common loader so my pipeline is same for both datasets.
    # Main things I handle here:
    # 1) sort by time
    # 2) normalize (z-score) so training is easier
    # 3) create sliding windows for supervised learning
    # 4) chronological split (no shuffle)

    df = pd.read_csv(path)

    if 'DATE' in df.columns:
        # Electricity dataset format: DATE + Value
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE')
        data = df['Value'].values.astype(float)

    elif 'Month' in df.columns:
        # AirPassengers dataset format: Month + #Passengers
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.sort_values('Month')
        data = df.iloc[:, 1].values.astype(float)

    else:
        raise ValueError("Unsupported dataset format")

    # Normalization: I do mean/std because models learn faster on stable scale.
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    if window_size is None:
        from utils.config import window_size

    if horizon is None:
        from utils.config import prediction_horizon as horizon

    # Windowing: X = past window_size points, y = next horizon points.
    # In tensor terms: X will be (num_samples, window_size) and y will be (num_samples, horizon).
    # This is exactly what I feed into models in train.py.
    X, y = create_sequences(data, window_size, horizon)

    split = int(0.8 * len(X))
    # Chronological split: first 80% train, last 20% test.

    X_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.float32).reshape(len(y[:split]), -1)

    X_test = torch.tensor(X[split:], dtype=torch.float32)
    y_test = torch.tensor(y[split:], dtype=torch.float32).reshape(len(y[split:]), -1)

    return X_train, y_train, X_test, y_test, mean, std
