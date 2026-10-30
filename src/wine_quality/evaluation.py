"""Regression metrics and a final residual plot."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

matplotlib.use("Agg")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE and R-squared for one set of predictions."""
    return {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def create_residual_plot(
    actual: np.ndarray, predicted: np.ndarray, output_path: Path
) -> None:
    """Save residuals against predictions to help check model errors."""
    residuals = actual - predicted
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predicted, residuals, alpha=0.35, s=18)
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_title("Residuals on the untouched test set")
    axis.set_xlabel("Predicted quality")
    axis.set_ylabel("Actual minus predicted quality")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
