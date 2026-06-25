"""Ligand branch: a Graph Isomorphism Network (GIN) encoder.

Reads the molecular graph (atoms = nodes, bonds = edges) and produces a single
fixed-length vector per molecule via message passing + global pooling. GIN is a
strong, simple baseline GNN for molecular property prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

from dhfr_affinity.features.ligand import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class LigandGNN(nn.Module):
    """GIN-based molecular encoder.

    Args:
        hidden_dim: width of node embeddings
        num_layers: number of message-passing layers
        out_dim: dimensionality of the pooled molecule embedding
        dropout: dropout applied to the pooled embedding
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_encoder = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.bond_encoder = nn.Linear(BOND_FEATURE_DIM, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # GINEConv uses edge features during message passing
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index, e)
            h = bn(h)
            h = F.relu(h)
        pooled = global_mean_pool(h, batch)  # [num_graphs, hidden_dim]
        pooled = F.dropout(pooled, p=self.dropout, training=self.training)
        return self.proj(pooled)  # [num_graphs, out_dim]
