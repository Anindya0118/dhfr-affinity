"""End-to-end pipeline: data -> features -> train -> evaluate.

Run from the repo root after `pip install -e .` and installing requirements:

    python scripts/run_pipeline.py --config configs/default.yaml

For a fast smoke run on a small slice:

    python scripts/run_pipeline.py --config configs/default.yaml --max-per-target 300

Outputs a comparison table (ligand-only vs two-branch) and saves the cleaned
dataset + metrics under ./outputs.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import yaml

from dhfr_affinity.data import (
    fetch_dhfr_bioactivities,
    clean_bioactivities,
    scaffold_split,
    random_split,
)
from dhfr_affinity.dataset import build_dataset, load_sequences
from dhfr_affinity.features.protein import ESMEmbedder
from dhfr_affinity.models import TwoBranchAffinityModel, LigandOnlyModel
from dhfr_affinity.train import make_loaders, train_model, evaluate_split
from dhfr_affinity.evaluate import comparison_table
from dhfr_affinity.utils import set_seed, get_device


def load_or_fetch(cfg, cache_csv: str, max_per_target):
    if os.path.exists(cache_csv):
        print(f"[data] loading cached {cache_csv}")
        return pd.read_csv(cache_csv)
    raw = fetch_dhfr_bioactivities(
        targets=cfg["data"]["targets"],
        accepted_types=cfg["data"]["accepted_types"],
        max_per_target=max_per_target,
    )
    clean = clean_bioactivities(raw, pic50_bounds=tuple(cfg["data"]["pic50_bounds"]))
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    clean.to_csv(cache_csv, index=False)
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-per-target", type=int, default=None)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(cfg["split"]["seed"])
    device = get_device()
    print(f"[device] {device}")

    # 1. data
    max_pt = args.max_per_target or cfg["data"]["max_per_target"]
    df = load_or_fetch(cfg, os.path.join(args.outdir, "dhfr_clean.csv"), max_pt)

    # 2. split
    splitter = scaffold_split if cfg["split"]["method"] == "scaffold" else random_split
    parts = splitter(
        df,
        frac_train=cfg["split"]["frac_train"],
        frac_valid=cfg["split"]["frac_valid"],
        seed=cfg["split"]["seed"],
    )

    # 3. features (protein embeddings + ligand graphs)
    sequences = load_sequences(cfg["data"]["sequences_fasta"])
    embedder = ESMEmbedder(cfg["protein"]["esm_model"], cfg["protein"]["cache_dir"], str(device))
    splits = {name: build_dataset(p, sequences, embedder) for name, p in parts.items()}
    protein_dim = splits["train"][0].protein.shape[1]

    loaders = make_loaders(splits, batch_size=cfg["train"]["batch_size"])

    # 4. train both models
    results = {}
    for name, ctor in [
        ("ligand_only", LigandOnlyModel),
        ("two_branch", TwoBranchAffinityModel),
    ]:
        print(f"\n=== training {name} ===")
        model = ctor(protein_dim=protein_dim, **cfg["model"])
        model, _ = train_model(
            model, loaders, device,
            epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"], patience=cfg["train"]["patience"],
        )
        results[name] = evaluate_split(model, loaders["test"], device)

    table = comparison_table(results)
    print("\n=== test-set comparison ===")
    print(table.round(3).to_string())
    table.to_csv(os.path.join(args.outdir, "comparison.csv"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump({k: v["metrics"] for k, v in results.items()}, fh, indent=2)
    print(f"\nsaved results to {args.outdir}/")


if __name__ == "__main__":
    main()
