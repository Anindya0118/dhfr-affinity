"""Build the PyTorch Geometric dataset.

Each example is a molecular graph with the target's ESM-2 embedding attached as
a graph-level attribute (`data.protein`) and the pIC50 as `data.y`. PyG's
DataLoader then batches graphs and stacks the protein vectors automatically.

Target sequences are supplied as a {organism: sequence} mapping (see
`load_sequences`). Download wild-type DHFR sequences from UniProt, e.g.:
    E. coli folA  -> P0ABQ4
    Human DHFR    -> P00374
and add resistant variants by editing the residues you care about.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from dhfr_affinity.features.ligand import smiles_to_graph
from dhfr_affinity.features.protein import ESMEmbedder


def load_sequences(path: str) -> dict[str, str]:
    """Load a simple FASTA file into {header: sequence}."""
    seqs, header, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(buf)
                header, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if header is not None:
        seqs[header] = "".join(buf)
    return seqs


def build_dataset(
    df,
    sequences: dict[str, str],
    embedder: ESMEmbedder,
    organism_col: str = "organism",
    smiles_col: str = "smiles",
    label_col: str = "pic50",
) -> list[Data]:
    """Return a list of PyG `Data` objects ready for a DataLoader.

    Rows whose organism has no sequence, or whose SMILES will not parse, are
    skipped with a count reported at the end.
    """
    # embed each unique target sequence once
    unique_seqs = [sequences[o] for o in df[organism_col].unique() if o in sequences]
    emb_by_seq = embedder.embed_many(unique_seqs)

    data_list, skipped = [], 0
    for _, row in df.iterrows():
        org = row[organism_col]
        if org not in sequences:
            skipped += 1
            continue
        graph = smiles_to_graph(row[smiles_col])
        if graph is None:
            skipped += 1
            continue
        seq = sequences[org]
        prot = torch.tensor(emb_by_seq[seq], dtype=torch.float).unsqueeze(0)  # [1, dim]
        graph.protein = prot
        graph.y = torch.tensor([row[label_col]], dtype=torch.float)
        graph.smiles = row[smiles_col]
        graph.organism = org
        data_list.append(graph)

    print(f"[dataset] built {len(data_list)} examples (skipped {skipped})")
    if data_list:
        print(f"[dataset] protein embedding dim = {data_list[0].protein.shape[1]}")
    return data_list
