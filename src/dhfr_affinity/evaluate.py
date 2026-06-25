"""Evaluation helpers: comparison tables and prediction-vs-truth plots."""
from __future__ import annotations

import numpy as np
import pandas as pd


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results: {model_name: {'metrics': {...}}} -> tidy DataFrame."""
    rows = []
    for name, res in results.items():
        row = {"model": name}
        row.update(res["metrics"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_predictions(y_true, y_pred, title: str = "", ax=None):
    """Scatter of predicted vs. true pIC50 with a y=x reference line."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ax.scatter(y_true, y_pred, s=14, alpha=0.5, edgecolor="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1)
    ax.set_xlabel("experimental pIC50")
    ax.set_ylabel("predicted pIC50")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_history(history: dict, title: str = "training", ax=None):
    """Plot training loss and validation RMSE over epochs."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train loss (MSE)")
    ax.plot(history["valid_rmse"], label="valid RMSE")
    ax.set_xlabel("epoch")
    ax.set_title(title)
    ax.legend()
    return ax
