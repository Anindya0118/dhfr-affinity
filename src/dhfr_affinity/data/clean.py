"""Clean raw ChEMBL bioactivities into a model-ready table.

Steps:
  1. keep only nM-unit potency rows with an exact ('=') relation
  2. convert potency to pIC50 = 9 - log10(value_in_nM)  (i.e. -log10(M))
  3. canonicalise SMILES with RDKit and drop anything unparseable
  4. collapse duplicate (molecule, target) pairs to their median pIC50
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_pic50(value_nm: float) -> float:
    """Convert a potency in nanomolar to pIC50 (= -log10 of molar concentration)."""
    value_nm = float(value_nm)
    if value_nm <= 0:
        return np.nan
    return 9.0 - np.log10(value_nm)


def _canonical_smiles(smiles: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def clean_bioactivities(
    df: pd.DataFrame,
    keep_relations: tuple[str, ...] = ("=",),
    pic50_bounds: tuple[float, float] = (3.0, 12.0),
) -> pd.DataFrame:
    """Return a cleaned, deduplicated DataFrame with a `pic50` label column.

    Output columns: organism, target_chembl_id, molecule_chembl_id, smiles, pic50
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings

    df = df.copy()

    # 1. unit + relation filtering
    df = df[df["standard_units"].astype(str).str.lower() == "nm"]
    if keep_relations:
        df = df[df["standard_relation"].isin(keep_relations)]
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]

    # 2. potency -> pIC50
    df["pic50"] = df["standard_value"].apply(to_pic50)
    lo, hi = pic50_bounds
    df = df[(df["pic50"] >= lo) & (df["pic50"] <= hi)]

    # 3. canonicalise SMILES
    df["smiles"] = df["smiles"].apply(_canonical_smiles)
    df = df.dropna(subset=["smiles"])

    # 4. deduplicate: median pIC50 per (molecule, target)
    grouped = (
        df.groupby(["organism", "target_chembl_id", "molecule_chembl_id", "smiles"], as_index=False)
        .agg(pic50=("pic50", "median"))
    )

    print(
        f"[clean] {len(grouped)} unique (molecule, target) pairs "
        f"across {grouped['organism'].nunique()} organisms"
    )
    return grouped.reset_index(drop=True)
