"""Ligand featurisation with RDKit.

Two representations:
  - smiles_to_graph(): atom/bond graph as a PyTorch Geometric `Data` object,
    consumed by the GNN ligand branch.
  - morgan_fingerprint(): a fixed-length ECFP-style bit vector, used by the
    classical baseline (so you can show the GNN actually earns its complexity).

Keeping the feature builders here -- not inside the model -- means the same
featuriser feeds training, evaluation and the FastAPI serving layer.
"""
from __future__ import annotations

import numpy as np

# ---- atom / bond feature vocabularies -------------------------------------
_ATOM_LIST = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # B C N O F P S Cl Br I
_DEGREES = [0, 1, 2, 3, 4, 5]
_HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
_NUM_HS = [0, 1, 2, 3, 4]

# atom feature dim = onehot(atom) + onehot(degree) + onehot(hyb) + onehot(Hs)
#                    + [formal_charge, is_aromatic, is_in_ring]
ATOM_FEATURE_DIM = (
    len(_ATOM_LIST) + 1
    + len(_DEGREES) + 1
    + len(_HYBRIDIZATIONS)
    + len(_NUM_HS) + 1
    + 3
)
# bond feature dim = onehot(bondtype:4) + [is_conjugated, is_in_ring]
BOND_FEATURE_DIM = 4 + 2


def _one_hot(value, choices) -> list[int]:
    """One-hot with a trailing 'unknown' bucket."""
    vec = [0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1
    except ValueError:
        vec[-1] = 1
    return vec


def _one_hot_exact(value, choices) -> list[int]:
    """One-hot with no unknown bucket (for closed vocabularies)."""
    vec = [0] * len(choices)
    if value in choices:
        vec[choices.index(value)] = 1
    return vec


def _atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), _ATOM_LIST)
        + _one_hot(atom.GetTotalDegree(), _DEGREES)
        + _one_hot_exact(str(atom.GetHybridization()), _HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalNumHs(), _NUM_HS)
        + [
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ]
    )


def _bond_features(bond) -> list[float]:
    from rdkit import Chem

    bt = bond.GetBondType()
    type_onehot = [
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC),
    ]
    return type_onehot + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


def smiles_to_graph(smiles: str):
    """Convert a SMILES string to a PyG `Data` graph, or None if unparseable.

    Node features: x  [num_atoms, ATOM_FEATURE_DIM]
    Edge index   : edge_index [2, 2*num_bonds]  (undirected -> both directions)
    Edge features: edge_attr  [2*num_bonds, BOND_FEATURE_DIM]
    """
    from rdkit import Chem
    import torch
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if len(edge_index) == 0:  # single-atom molecule: add a self loop
        edge_index = [[0, 0]]
        edge_attr = [[0.0] * BOND_FEATURE_DIM]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2):
    """Return an ECFP4-style Morgan fingerprint as a float numpy array, or None."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, arr)
    return arr
