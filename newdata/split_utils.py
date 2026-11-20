"""
Shared sample-filtering, covariate and splitting logic for the TCGA data.

Used by processing.py (full pipeline from the raw Xena matrix) and by
build_l978_splits.py (re-filtering / re-splitting the existing L978 files).

Rules (agreed for this project):
  * keep only primary tumors: TCGA sample-type code 01 (primary solid tumor)
    and 03 (primary blood-derived cancer, i.e. LAML). Normals (10-19),
    recurrent (02), additional primary (05) and metastatic (06) are dropped.
  * one sample per patient (first full barcode in sorted order).
  * race from the Pan-Cancer clinical table; missing / not evaluated / unknown
    are pooled into a single recorded category "unknown".
  * split by patient, 80/10/10 train/val/test, stratified by cancer cohort.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KEEP_SAMPLE_CODES = {1, 3}

RACE_MAP = {
    "WHITE": "white",
    "BLACK OR AFRICAN AMERICAN": "black",
    "ASIAN": "asian",
    "AMERICAN INDIAN OR ALASKA NATIVE": "aian",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "nhpi",
}
RACE_NA = "unknown"  # recorded-but-missing race (not "NA": pandas reads that as NaN); a real category, NOT the CFG null token
RACES = ["white", "black", "asian", "aian", "nhpi", RACE_NA]

SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def sample_type_code(full_barcodes: pd.Series) -> pd.Series:
    """TCGA sample-type code = characters 14-15 of the barcode (e.g. '01')."""
    assert full_barcodes.str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}").all()
    return full_barcodes.str[13:15].astype(int)


def select_primary_one_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Keep primary tumors (codes 01/03) and one sample per patient."""
    code = sample_type_code(df["full_barcode"])
    df = df[code.isin(KEEP_SAMPLE_CODES).to_numpy()].copy()
    df["patient"] = df["full_barcode"].str[:12]
    df = (df.sort_values("full_barcode")
            .drop_duplicates("patient", keep="first")
            .reset_index(drop=True))
    assert df["patient"].is_unique
    df["cancer"] = 1
    return df


def load_patient_race(clinical_path) -> pd.Series:
    """Patient barcode (TCGA-XX-XXXX) -> normalised race label."""
    clin = pd.read_csv(clinical_path, sep="\t", index_col=0)
    clin["patient"] = clin.index.str[:12]
    clin = clin.drop_duplicates("patient").set_index("patient")
    race = clin["race"].map(RACE_MAP).fillna(RACE_NA)
    assert set(race.unique()) <= set(RACES), set(race.unique()) - set(RACES)
    return race


def add_race(df: pd.DataFrame, race: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["race"] = df["full_barcode"].str[:12].map(race)
    n_missing = int(df["race"].isna().sum())
    if n_missing:
        print(f"[add_race] {n_missing} samples have no clinical row -> dropped")
        df = df.dropna(subset=["race"])
    return df


def patient_split(df: pd.DataFrame, ratios=SPLIT_RATIOS, seed: int = SPLIT_SEED):
    """
    Split rows into train/val/test so that no patient appears in more than one
    split. Stratified on cohort. Works at the patient level, so it is correct
    even if a patient were to have several rows.
    """
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1) < 1e-9

    patient = df["full_barcode"].str[:12]
    pat = (pd.DataFrame({"patient": patient, "cohort": df["cohort"].to_numpy()})
             .drop_duplicates("patient"))
    assert pat["patient"].is_unique, "a patient maps to more than one cohort"

    train_p, rest_p = train_test_split(
        pat, test_size=val_r + test_r, random_state=seed, stratify=pat["cohort"])
    val_p, test_p = train_test_split(
        rest_p, test_size=test_r / (val_r + test_r), random_state=seed,
        stratify=rest_p["cohort"])

    out = []
    for p in (train_p, val_p, test_p):
        out.append(df[patient.isin(set(p["patient"])).to_numpy()].reset_index(drop=True))
    check_patient_disjoint(*out)
    return tuple(out)


def check_patient_disjoint(*splits: pd.DataFrame):
    pats = [set(s["full_barcode"].str[:12]) for s in splits]
    for i in range(len(pats)):
        for j in range(i + 1, len(pats)):
            overlap = pats[i] & pats[j]
            assert not overlap, f"splits {i} and {j} share {len(overlap)} patients"


def summarize(name: str, df: pd.DataFrame):
    print(f"--- {name}: {len(df)} samples, {df['full_barcode'].str[:12].nunique()} patients")
    print("  sample codes:", sample_type_code(df["full_barcode"]).value_counts().to_dict())
    print("  sex (1=female):", df["gender"].value_counts().to_dict())
    print("  race:", df["race"].value_counts().to_dict())
    print(f"  age: min {df['age'].min()}, median {df['age'].median()}, max {df['age'].max()}")
    print("  cohorts:", df["cohort"].nunique())
