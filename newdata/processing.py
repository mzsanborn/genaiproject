import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from data.utils_preprocessing import COHORTS, TISSUES, CANCER_TYPES   # repo mappings
from pathlib import Path

BASE = Path(__file__).resolve().parent
XENA = BASE / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
CLIN = BASE / "clinical.tsv"
OUT = BASE / "data_RNAseq_RTCGA"
OUT.mkdir(parents=True, exist_ok=True)          # wherever the repo's PATH points
UNLOG = False                          # variant B: 2**x - 1 ; set False for variant A

# --- expression: genes x samples -> samples x genes
expr = pd.read_csv(XENA, sep="\t", index_col=0).astype("float32").T
print(expr.shape, expr.columns[:3].tolist())
assert all("|" in g for g in expr.columns), "no SYMBOL|ENTREZ ids -> need a symbol->Entrez map"
gene_ids = [g.split("|")[1] for g in expr.columns]
assert len(set(gene_ids)) == len(gene_ids)
expr.columns = gene_ids
expr = expr.fillna(0.0)                              # repo sets NaN -> 0
if UNLOG:
    expr = (2.0 ** expr - 1.0).clip(lower=0)

# --- clinical at patient level (CDR is tumor-only; normals borrow their patient's row)
clin = pd.read_csv(CLIN, sep="\t", index_col=0)
print(list(clin.columns))                            # VERIFY the three names below
STUDY, AGE, GENDER = "cancer type abbreviation", "age_at_initial_pathologic_diagnosis", "gender"
clin["patient"] = clin.index.str[:12]
pat = (clin.drop_duplicates("patient").set_index("patient")[[STUDY, AGE, GENDER]]
           .set_axis(["cohort", "age", "gender"], axis=1))

# --- sample-level metadata
code = expr.index.str[13:15].astype(int)
meta = pd.DataFrame({"full_barcode": expr.index, "patient": expr.index.str[:12],
                     "cancer": (code < 10).astype(int)})
meta = meta.merge(pat, left_on="patient", right_index=True, how="inner")
meta = meta.dropna(subset=["cohort", "age", "gender"])
meta["tissue_type"] = meta["cohort"].map(dict(zip(COHORTS, TISSUES)))
meta["cancer_type"] = meta["cohort"].map(dict(zip(COHORTS, CANCER_TYPES)))
meta.loc[meta["cancer"] == 0, "cancer_type"] = "normal"
meta = meta.dropna(subset=["tissue_type"])           # cohorts the repo doesn't know
meta["age"] = meta["age"].astype(float).astype(int)
meta["reduced_barcode"] = meta["patient"].str.lower()
meta["barcode"] = meta["reduced_barcode"]

# --- drop exact-duplicate expression vectors (repo's only dedupe)
X = expr.loc[meta["full_barcode"]]
dup = X.duplicated().to_numpy()
meta, X = meta[~dup].reset_index(drop=True), X[~dup].reset_index(drop=True)

df = pd.concat([meta[["full_barcode", "reduced_barcode", "tissue_type", "cancer_type",
                      "cohort", "cancer"]], X, meta[["barcode", "gender", "age"]]], axis=1)

print(len(df), "samples")                            # paper: 9749 (won't match exactly)
print(df["tissue_type"].nunique(), "tissues")        # expect 24
print(df["cancer"].value_counts())

# --- split: ~1/6 test; the loader then splits train 80/20 into train/val
train, test = train_test_split(df, test_size=1/6, random_state=42, stratify=df["tissue_type"])
np.save(f"{OUT}/TCGA_rnaseq_RSEM_gene_ids_full.npy", np.array(gene_ids))
train.to_csv(f"{OUT}/train_df_covariates.csv", index=False, float_format="%.6g")
test.to_csv(f"{OUT}/test_df_covariates.csv",  index=False, float_format="%.6g")