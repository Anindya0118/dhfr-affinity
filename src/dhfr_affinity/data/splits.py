"""Dataset splitting.

The default is a **scaffold split**, not a random split. Random splits leak:
near-identical analogues end up in both train and test, so the model looks
great but has really just memorised chemotypes. Scaffold splitting assigns
whole Bemis-Murcko scaffolds to a single fold, forcing the model to
generalise to unseen chemical series -- a much fairer estimate of how the
model behaves on genuinely new chemistry. (This is the 'is the prediction
in-domain?' discipline applied to the data split.)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(
    df: pd.DataFrame,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 42,
    smiles_col: str = "smiles",
) -> dict[str, pd.DataFrame]:
    """Split by Bemis-Murcko scaffold. Returns {'train','valid','test'} frames.

    Largest scaffold groups are placed first into train, encouraging the test
    set to contain rarer / novel scaffolds.
    """
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for idx, smi in zip(df.index, df[smiles_col]):
        scaffolds[_murcko_scaffold(smi)].append(idx)

    # order scaffold groups largest-first
    groups = sorted(scaffolds.values(), key=len, reverse=True)

    n = len(df)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    train_idx, valid_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx += group
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx += group
        else:
            test_idx += group

    rng = np.random.default_rng(seed)
    for idx_list in (train_idx, valid_idx, test_idx):
        rng.shuffle(idx_list)

    out = {
        "train": df.loc[train_idx].reset_index(drop=True),
        "valid": df.loc[valid_idx].reset_index(drop=True),
        "test": df.loc[test_idx].reset_index(drop=True),
    }
    print(
        f"[split:scaffold] train={len(out['train'])} "
        f"valid={len(out['valid'])} test={len(out['test'])} "
        f"({len(groups)} scaffolds)"
    )
    return out


def random_split(
    df: pd.DataFrame,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Random split -- provided only as an optimistic baseline for comparison."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    parts = {
        "train": df.iloc[idx[:n_train]].reset_index(drop=True),
        "valid": df.iloc[idx[n_train : n_train + n_valid]].reset_index(drop=True),
        "test": df.iloc[idx[n_train + n_valid :]].reset_index(drop=True),
    }
    print(
        f"[split:random] train={len(parts['train'])} "
        f"valid={len(parts['valid'])} test={len(parts['test'])}"
    )
    return parts
