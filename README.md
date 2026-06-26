# dhfr-affinity

Predicting how tightly small-molecule antifolates bind dihydrofolate reductase (DHFR), with an ESM-2 + graph-neural-net model.

I build interpretable ML for *E. coli* resistance to trimethoprim-sulfamethoxazole, and trimethoprim is a small-molecule DHFR inhibitor. That work is about why a target stops responding to a drug. This repo is me working the other side of it: given the target, predict which molecules bind, and how well.

## What it does

A two-branch model:

- a GIN graph network over the ligand (atoms/bonds), built from SMILES with RDKit;
- a frozen ESM-2 embedding of the DHFR protein sequence;
- concatenate the two, MLP head, predict pIC50.

There's also a ligand-only version with no protein branch, as an ablation — if the two-branch model can't beat it, the protein signal isn't doing anything, and the numbers below say how that turned out.

Data is DHFR bioactivity pulled from ChEMBL (E. coli + human), converted to pIC50. Train/test is a **scaffold split** rather than random, so whole chemical series stay on one side — random splits leak close analogues and inflate the score.

## Results so far (the honest version)

Small dataset: ~320 compounds after cleaning and filtering to the organisms I have sequences for, with a ~30-compound test set. On a scaffold split:

| model | RMSE | Pearson | Spearman |
|---|---|---|---|
| ligand-only | 1.72 | -0.43 | -0.41 |
| two-branch  | 1.54 |  0.28 |  0.19 |

Read this for what it is. The two-branch model beats the ligand-only baseline and the correlation goes from negative to positive, so the protein branch is contributing something. But Pearson 0.28 is weak, and a 30-compound test set makes the metric itself noisy. The bottleneck is data volume, not the architecture — the model early-stops in ~20 epochs because the validation set is tiny. Next step is loosening the ChEMBL filters and adding more targets (with their sequences) to get into the low thousands of compounds, then retraining.

I'd rather show the real number and the reason behind it than a tuned figure on 30 points.

## Project 2 — Boltz-2 co-folding (in progress)

Separate question: can you trust a co-folding foundation model's *predicted* affinity? `notebooks/02_project2_boltz2.ipynb` runs Boltz-2 on DHFR-ligand complexes and pulls out its affinity prediction. Trimethoprim against E. coli DHFR runs end to end and produces an affinity value. Still to do: a 3-6 compound panel to check whether Boltz's ranking matches experiment, and a wild-type vs resistant-variant comparison to see if it picks up a binding-site mutation. That last one is the interesting test and it isn't done yet.

## Layout

```
src/dhfr_affinity/
  data/      ChEMBL fetch, cleaning -> pIC50, scaffold split
  features/  RDKit ligand graphs + fingerprints, ESM-2 embeddings
  models/    GIN ligand branch, two-branch model + ligand-only baseline
  dataset.py train.py evaluate.py interpret.py utils.py
notebooks/
  01_project1_colab.ipynb   data -> train -> evaluate (run this one)
  02_project2_boltz2.ipynb  Boltz-2 co-folding (in progress)
```

## Running it

Open `notebooks/01_project1_colab.ipynb` in Colab on a GPU, set the clone URL, run top to bottom. It pulls ChEMBL, fetches the DHFR sequences from UniProt, trains both models, prints the comparison and prediction plots. Locally: `pip install -r requirements.txt && pip install -e .`, then `python scripts/run_pipeline.py --max-per-target 300` for a quick pass.

## Notes

- ESM-2 is used frozen, as a feature extractor — not fine-tuned.
- The protein side is sequence-based, so this is structure-*aware* via the ligand graph; the genuinely 3D structure view is what Project 2 is for.
- pIC50 pools IC50/Ki/Kd across organisms to have enough data; E. coli is the case study, not the whole training set.
