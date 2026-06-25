"""Interpretability: which atoms drive a predicted affinity.

A lightweight, dependency-free attribution: take the gradient of the prediction
with respect to the atom feature matrix and reduce to a per-atom saliency. This
gives a chemist-readable signal -- 'the model leans on this part of the
molecule' -- which is the interpretability angle that differentiates this work
and speaks directly to the 'can you trust / explain the prediction?' theme.

For a more rigorous treatment, swap in Integrated Gradients or
torch_geometric.explain.GNNExplainer; this is the fast first pass.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def atom_attribution(model, data, device) -> np.ndarray:
    """Per-atom saliency for a single PyG `Data` molecule.

    Returns a 1-D array of length num_atoms (higher = more influential).
    """
    model.eval()
    batch = next(iter(DataLoader([data], batch_size=1))).to(device)
    batch.x.requires_grad_(True)
    pred = model(batch)
    pred.backward()
    # L2 norm of the gradient over each atom's feature vector
    grad = batch.x.grad.detach().cpu().numpy()  # [num_atoms, feat_dim]
    saliency = np.linalg.norm(grad, axis=1)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency


def draw_attribution(smiles: str, saliency: np.ndarray):
    """Return an RDKit image of the molecule coloured by atom saliency."""
    from rdkit import Chem
    from rdkit.Chem.Draw import SimilarityMaps
    import matplotlib.pyplot as plt

    mol = Chem.MolFromSmiles(smiles)
    weights = [float(s) for s in saliency[: mol.GetNumAtoms()]]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap="bwr")
    return fig
