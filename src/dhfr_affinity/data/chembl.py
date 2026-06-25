"""Pull DHFR inhibitor bioactivity data from ChEMBL.

We headline *E. coli* DHFR (the trimethoprim target, directly tied to the
TMP-SMX resistance work) but pull DHFR across several organisms so there is
enough data to train on. Each row keeps its organism so you can slice the
E. coli case study out later.

ChEMBL target IDs (verify on https://www.ebi.ac.uk/chembl/ as they can change):
  CHEMBL202    Human DHFR
  CHEMBL1809   E. coli DHFR
  CHEMBL2366   Mouse DHFR
  CHEMBL3494   P. falciparum DHFR (antifolate-rich)
You can edit DHFR_TARGETS in configs/default.yaml without touching code.
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

# Sensible defaults; override via config. Keys are organism labels.
DHFR_TARGETS = {
    "human": "CHEMBL202",
    "e_coli": "CHEMBL1809",
    "mouse": "CHEMBL2366",
    "p_falciparum": "CHEMBL3494",
}

# Activity types we accept (all are -log-transformable potency measures).
ACCEPTED_TYPES = ("IC50", "Ki", "Kd")


def fetch_dhfr_bioactivities(
    targets: dict[str, str] | None = None,
    accepted_types: Iterable[str] = ACCEPTED_TYPES,
    max_per_target: int | None = None,
) -> pd.DataFrame:
    """Query ChEMBL for activities against the given DHFR targets.

    Returns a tidy DataFrame with columns:
        organism, target_chembl_id, molecule_chembl_id, smiles,
        standard_type, standard_value, standard_units, standard_relation

    Requires `chembl_webresource_client` (pip install chembl_webresource_client)
    and network access. Run this on Colab, not in an offline sandbox.
    """
    from chembl_webresource_client.new_client import new_client

    targets = targets or DHFR_TARGETS
    activity = new_client.activity
    rows = []

    for organism, target_id in targets.items():
        print(f"[chembl] fetching {organism} ({target_id}) ...", flush=True)
        query = activity.filter(
            target_chembl_id=target_id,
            standard_type__in=list(accepted_types),
        ).only(
            [
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_type",
                "standard_value",
                "standard_units",
                "standard_relation",
                "target_chembl_id",
            ]
        )
        count = 0
        for rec in query:
            if rec.get("canonical_smiles") is None:
                continue
            if rec.get("standard_value") is None:
                continue
            rows.append(
                {
                    "organism": organism,
                    "target_chembl_id": rec["target_chembl_id"],
                    "molecule_chembl_id": rec["molecule_chembl_id"],
                    "smiles": rec["canonical_smiles"],
                    "standard_type": rec["standard_type"],
                    "standard_value": rec["standard_value"],
                    "standard_units": rec.get("standard_units"),
                    "standard_relation": rec.get("standard_relation"),
                }
            )
            count += 1
            if max_per_target and count >= max_per_target:
                break
        print(f"[chembl]   -> {count} records", flush=True)
        time.sleep(0.5)  # be polite to the API

    df = pd.DataFrame(rows)
    print(f"[chembl] total raw records: {len(df)}")
    return df
