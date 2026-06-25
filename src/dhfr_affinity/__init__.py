"""dhfr_affinity: structure-aware protein-ligand binding affinity prediction
for antifolate small molecules against dihydrofolate reductase (DHFR).

Two-branch architecture:
  - ligand branch  : a message-passing GNN over the molecular graph (PyTorch Geometric)
  - protein branch : ESM-2 sequence embeddings of the DHFR target (wild-type or variant)
  - fusion         : concatenate the two branch embeddings -> MLP -> predicted pIC50

This mirrors the ClusterFocusedHybrid two-branch design generalised from
AMR resistance-driver discovery to protein-ligand affinity.
"""

__version__ = "0.1.0"
