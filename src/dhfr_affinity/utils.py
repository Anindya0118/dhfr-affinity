"""Shared utilities: reproducibility, device selection, regression metrics."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and Torch (if installed) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device ('cuda' or 'cpu')."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    pearson: float
    spearman: float
    n: int

    def as_dict(self) -> dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "n": self.n,
        }


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    """Compute RMSE / MAE / Pearson / Spearman for affinity regression.

    Pearson tells you linear correlation; Spearman tells you ranking quality,
    which is what a chemist actually cares about (which compound to make next).
    """
    from scipy.stats import pearsonr, spearmanr

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # guard against constant predictions (correlation undefined)
    pear = float(pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 1e-8 else float("nan")
    spear = float(spearmanr(y_true, y_pred)[0]) if np.std(y_pred) > 1e-8 else float("nan")
    return RegressionMetrics(rmse, mae, pear, spear, len(y_true))
