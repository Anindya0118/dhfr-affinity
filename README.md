# DHFR Antifolate Binding-Affinity Prediction

**Structure-aware protein–ligand binding affinity prediction for antifolate small molecules against dihydrofolate reductase (DHFR).**

A two-branch model: a graph neural network reads the ligand (RDKit → PyTorch Geometric), an ESM-2 protein language model reads the DHFR target, and a fusion head predicts pIC50. It generalises a two-branch architecture I built for antimicrobial-resistance work to the protein–ligand affinity problem.

---

## Why DHFR / antifolates

Trimethoprim — half of the antibiotic TMP-SMX (co-trimoxazole) — is a small-molecule inhibitor of dihydrofolate reductase (DHFR, the *folA* gene product). My prior work modelled *E. coli* resistance to TMP-SMX from genomics. This project asks the complementary, structure-driven question: **given a small molecule and a DHFR target (wild-type or a resistant variant), how tightly do they bind?** That is the binding-affinity prediction task at the centre of small-molecule drug design.

## What's modelled

- **Ligand branch** — a GIN message-passing network over the atom/bond graph (`features/ligand.py`, `models/ligand_gnn.py`).
- **Protein branch** — frozen ESM-2 embeddings of the target sequence, mean-pooled (`features/protein.py`). Variant sequences let the model see how binding-site mutations shift predicted affinity.
- **Fusion** — concatenate both branch embeddings → MLP → pIC50 (`models/hybrid.py`).
- **Baseline** — a ligand-only ablation. If the two-branch model doesn't beat it, the protein signal isn't helping, and the writeup says so.

## Methodological choices that matter

- **Scaffold split, not random** (`data/splits.py`) — random splits leak analogues across train/test and inflate scores. Scaffold splitting forces generalisation to unseen chemotypes — an honest estimate of out-of-domain behaviour.
- **Interpretability** (`interpret.py`) — per-atom gradient attribution showing which parts of a molecule drive a prediction.
- **Companion structure check (Project 2)** — Boltz-2 co-folding as an independent, structure-based sanity check on predicted affinities, with an explicit assessment of where the foundation model is and isn't trustworthy.

---

## Repo layout

```
src/dhfr_affinity/
  data/        ChEMBL fetch (chembl.py), cleaning/pIC50 (clean.py), scaffold split (splits.py)
  features/    RDKit ligand graphs + fingerprints (ligand.py), ESM-2 embeddings (protein.py)
  models/      GIN ligand branch (ligand_gnn.py), two-branch + baseline (hybrid.py)
  dataset.py   bundles graphs + protein embeddings + labels into PyG Data
  train.py     training loop w/ early stopping; evaluate.py  metrics + plots; interpret.py
  utils.py     seeding, device, regression metrics
notebooks/
  01_project1_colab.ipynb   end-to-end runner (Project 1)
  02_project2_boltz2.ipynb  Boltz-2 co-folding reality check (Project 2)
scripts/run_pipeline.py     CLI: data -> features -> train -> evaluate
configs/default.yaml        all knobs in one place
```

---

## Quickstart (Google Colab — recommended)

1. Push this repo to GitHub.
2. Open `notebooks/01_project1_colab.ipynb` in Colab, set **Runtime → GPU**, edit the clone URL, and run top to bottom. It installs deps, auto-fetches DHFR sequences from UniProt, pulls + cleans ChEMBL data, trains the baseline and the two-branch model, and plots results.
3. For the structure check, open `notebooks/02_project2_boltz2.ipynb` on an **A100/L4** runtime (Pro+ or pay-as-you-go) and run a few DHFR–ligand complexes.

## Quickstart (local / CLI)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install torch + torch-geometric matched to your platform/CUDA first if on GPU
pip install -e .

# add real DHFR sequences (UniProt P0ABQ4 for E. coli, P00374 for human) to:
#   data/dhfr_sequences.fasta

# fast first pass on a small data slice:
python scripts/run_pipeline.py --config configs/default.yaml --max-per-target 300
```

Outputs (`outputs/`): `dhfr_clean.csv`, `comparison.csv` (ligand-only vs two-branch), `metrics.json`.

---

## Status / roadmap

- [x] Data, featurisation, two-branch model, scaffold split, training, evaluation, attribution
- [ ] Resistant-variant case study (WT vs mutant DHFR embedding, predicted potency shift)
- [ ] Boltz-2 co-folding comparison + trust assessment
- [ ] FastAPI + Docker serving demo (SMILES + target → predicted pIC50 + attribution)

## Notes & honesty

- pIC50 is pooled across IC50/Ki/Kd and several organisms to have enough data; the *E. coli* slice is the case study, not the whole training set.
- The protein branch here is **sequence-based** (ESM-2), so this is structure-*aware* via the ligand graph and sequence-driven on the target side; Project 2 adds the genuinely 3D structure-based view.
- ESM-2 is used as a frozen feature extractor (not fine-tuned).
