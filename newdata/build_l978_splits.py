"""
Rebuild the L978 train/val/test files from the existing L978 CSVs without
needing the raw (too large to commit) Xena matrix.

    python newdata/build_l978_splits.py

Reads   newdata/data_RNAseq_RTCGA_L978/{train,test}_df_covariates.csv (old split)
        newdata/clinical.tsv
Writes  newdata/data_RNAseq_RTCGA_L978/{train,val,test}_df_covariates.csv

The old files mixed normals/metastases in and split by sample, so 260 patients
were in both train and test. This pools them, applies the rules in
split_utils.py, and re-splits by patient. Idempotent: running it again on its
own output gives the same result.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import (select_primary_one_per_patient, load_patient_race, add_race,
                         patient_split, summarize)

BASE = Path(__file__).resolve().parent
L978 = BASE / "data_RNAseq_RTCGA_L978"
CLIN = BASE / "clinical.tsv"
META_FRONT = ["full_barcode", "reduced_barcode", "tissue_type", "cancer_type", "cohort", "cancer"]
META_BACK = ["barcode", "gender", "age", "race"]


def main():
    parts = [pd.read_csv(p) for p in sorted(L978.glob("*_df_covariates.csv"))]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("full_barcode")
    df = df.drop(columns=[c for c in ["race", "patient"] if c in df.columns])
    print(f"pooled: {len(df)} samples, {df['full_barcode'].str[:12].nunique()} patients")

    gene_cols = [c for c in df.columns if c.isdigit()]
    assert len(gene_cols) == 978, len(gene_cols)

    df = select_primary_one_per_patient(df)
    df = add_race(df, load_patient_race(CLIN))
    df = df[META_FRONT + gene_cols + META_BACK]

    train, val, test = patient_split(df)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        summarize(name, d)
        d.to_csv(L978 / f"{name}_df_covariates.csv", index=False, float_format="%.6g")
    print("written to", L978)


if __name__ == "__main__":
    main()
