"""Affinity models.

TwoBranchAffinityModel  -- the flagship. Ligand GNN embedding is concatenated
with the ESM-2 protein embedding, then an MLP head regresses pIC50. This is the
ClusterFocusedHybrid two-branch pattern generalised to protein-ligand affinity:
one branch sees the small molecule, one sees the target, fusion sees both.

LigandOnlyModel -- ablation baseline that drops the protein branch. If the
two-branch model does not beat this, the protein information is not helping and
you should say so honestly. (This mirrors the presence-only vs hybrid baseline
comparison in the AMR work.)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dhfr_affinity.models.ligand_gnn import LigandGNN


def _mlp_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, 1),
    )


class TwoBranchAffinityModel(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        ligand_out: int = 128,
        protein_out: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ligand_branch = LigandGNN(
            hidden_dim=gnn_hidden, num_layers=gnn_layers,
            out_dim=ligand_out, dropout=dropout,
        )
        self.protein_branch = nn.Sequential(
            nn.Linear(protein_dim, protein_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = _mlp_head(ligand_out + protein_out, head_hidden, dropout)

    def forward(self, batch):
        lig = self.ligand_branch(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )                                   # [B, ligand_out]
        prot = self.protein_branch(batch.protein)  # [B, protein_out]
        fused = torch.cat([lig, prot], dim=1)
        return self.head(fused).squeeze(-1)        # [B]


class LigandOnlyModel(nn.Module):
    """Ablation: ligand branch only, no protein information."""

    def __init__(
        self,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        ligand_out: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.ligand_branch = LigandGNN(
            hidden_dim=gnn_hidden, num_layers=gnn_layers,
            out_dim=ligand_out, dropout=dropout,
        )
        self.head = _mlp_head(ligand_out, head_hidden, dropout)

    def forward(self, batch):
        lig = self.ligand_branch(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        return self.head(lig).squeeze(-1)
