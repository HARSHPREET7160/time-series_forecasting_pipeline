import numpy as np

def create_sequences(data, window_size, horizon):
    """
    Convert a 1D time series into supervised learning samples.

    Why: sequence models (and even MLP baselines) need fixed-shape inputs, so we turn the continuous series into
    sliding windows of past values (X) paired with the next `horizon` value(s) to predict (y).

    In my words: I am basically cutting the long time series into many small examples.
    Example: if window=3 and horizon=1, then [x1,x2,x3] -> [x4], then [x2,x3,x4] -> [x5], like that.
    """
    X, y = [], []

    for i in range(len(data) - window_size - horizon + 1):
        # Past window as input
        X.append(data[i:i+window_size])
        # Next value(s) as target
        y.append(data[i+window_size:i+window_size+horizon])

    return np.array(X), np.array(y)
