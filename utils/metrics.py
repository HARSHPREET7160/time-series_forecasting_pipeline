import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    # I have 3 metrics because it tell what is best:
    # MSE punish big errors more, MAE is more direct average error, RMSE is sqrt of MSE in same unit.
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return mse, mae, rmse
