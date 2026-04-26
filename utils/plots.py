import matplotlib.pyplot as plt
import os
import re

def _safe_name(title):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(title)).strip("_")
    return name or "plot"

def plot_loss(losses, title, save_dir="plots", show=True):
    # I save plots also, because in GitHub it should show outputs (not only pop-up window).
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{_safe_name(title)}.png"), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def plot_predictions(y_true, y_pred, title, save_dir="plots", show=True):
    # y_true and y_pred come from model outputs (after denormalize in train.py).
    # If horizon > 1, then it is 2D array, so for simple view I plot only first step.
    # For my roll no horizon is 1, but I keep this for general.
    if hasattr(y_true, "ndim") and y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = y_true[:, 0]
    if hasattr(y_pred, "ndim") and y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = y_pred[:, 0]

    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{_safe_name(title)}.png"), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
