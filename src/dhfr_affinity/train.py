"""Training loop for the affinity models.

Plain, readable PyTorch: MSE loss, Adam, early stopping on validation RMSE.
Works for both TwoBranchAffinityModel and LigandOnlyModel since both take a
single batched PyG object and return a [B] prediction.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dhfr_affinity.utils import regression_metrics


def make_loaders(splits: dict, batch_size: int = 64):
    """splits: {'train','valid','test'} -> lists of PyG Data. Returns loaders."""
    return {
        name: DataLoader(data, batch_size=batch_size, shuffle=(name == "train"))
        for name, data in splits.items()
    }


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out.detach().cpu().numpy())
        trues.append(batch.y.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_model(
    model,
    loaders: dict,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    verbose: bool = True,
):
    """Train with early stopping; return (best_model, history dict)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_rmse, best_state, wait = float("inf"), None, 0
    history = {"train_loss": [], "valid_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.num_graphs
        epoch_loss /= len(loaders["train"].dataset)

        vp, vt = _predict(model, loaders["valid"], device)
        v_rmse = regression_metrics(vt, vp).rmse
        history["train_loss"].append(epoch_loss)
        history["valid_rmse"].append(v_rmse)

        if v_rmse < best_rmse - 1e-4:
            best_rmse, best_state, wait = v_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch:3d} | train_loss {epoch_loss:.4f} | valid_rmse {v_rmse:.4f}")

        if wait >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best valid_rmse {best_rmse:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_split(model, loader, device) -> dict:
    """Return regression metrics on a loader plus raw preds/trues for plotting."""
    preds, trues = _predict(model, loader, device)
    m = regression_metrics(trues, preds).as_dict()
    return {"metrics": m, "y_pred": preds, "y_true": trues}
