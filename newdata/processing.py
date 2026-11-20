import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from data.utils_preprocessing import COHORTS, TISSUES, CANCER_TYPES   # repo mappings


BASE = Path(__file__).resolve().parent
# XENA = BASE / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
# CLIN = BASE / "clinical.tsv"
XENA = "/Users/goelaine314/Downloads/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
CLIN = "/Users/goelaine314/Downloads/clinical.tsv"
OUT = BASE / "data_RNAseq_RTCGA"
OUT.mkdir(parents=True, exist_ok=True)          # wherever the repo's PATH points
UNLOG = True                          # variant B: 2**x - 1 ; set False for variant A

# --- expression: genes x samples -> samples x genes
expr = pd.read_csv(XENA, sep="\t", index_col=0).astype("float32").T
print(expr.shape, expr.columns[:3].tolist())
# assert all("|" in g for g in expr.columns), "no SYMBOL|ENTREZ ids -> need a symbol->Entrez map"
# gene_ids = [g.split("|")[1] for g in expr.columns]
# assert len(set(gene_ids)) == len(gene_ids)
# expr.columns = gene_ids
# expr.columns = expr.columns.astype(str)
# bad = expr.columns[~expr.columns.str.fullmatch(r"\d+")]
# print(len(bad), "non-numeric gene ids")
# print(bad[:20].tolist())
# assert expr.columns.str.fullmatch(r"\d+").all(), "non-numeric gene ids"
# gene_ids = list(expr.columns)
# assert len(set(gene_ids)) == len(gene_ids), "duplicate gene ids"

# import urllib.request

# ---- symbol -> Entrez map from NCBI
GI = BASE / "Homo_sapiens.gene_info.gz"
# if not GI.exists():
#     urllib.request.urlretrieve(
#         "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Homo_sapiens.gene_info.gz", GI)
gi = pd.read_csv(GI, sep="\t", dtype=str, usecols=["GeneID", "Symbol", "Synonyms"])

primary = dict(zip(gi["Symbol"], gi["GeneID"]))
syn = {}
for gid, s in zip(gi["GeneID"], gi["Synonyms"]):
    if s != "-":
        for x in s.split("|"):
            syn.setdefault(x, set()).add(gid)
syn = {k: next(iter(v)) for k, v in syn.items() if len(v) == 1 and k not in primary}

def to_entrez(g):
    return g if g.isdigit() else (primary.get(g) or syn.get(g))

# mapped = [to_entrez(str(g)) for g in expr.columns]
# unmapped = [g for g, m in zip(expr.columns, mapped) if m is None]
# print(f"{len(unmapped)} unmapped symbols:", unmapped[:30])

# # unmapped genes keep their symbol as a placeholder name so the gene count stays 20531
# new_cols = [m if m is not None else str(g) for g, m in zip(expr.columns, mapped)]
# expr.columns = new_cols

# dup = expr.columns.duplicated()
# print("duplicate gene ids (later copies dropped):", int(dup.sum()))
# expr = expr.loc[:, ~dup]

# gene_ids = list(expr.columns)
# print("final gene count:", len(gene_ids))
prim = [str(g) if str(g).isdigit() else primary.get(str(g)) for g in expr.columns]
claimed = {m for m in prim if m}
mapped = []
for g, m in zip(expr.columns, prim):
    if m is None:
        s = syn.get(str(g))
        if s is not None and s not in claimed:
            claimed.add(s)
            m = s
    mapped.append(m)

unmapped = [g for g, m in zip(expr.columns, mapped) if m is None]
print(f"{len(unmapped)} unmapped symbols (kept under their symbol name)")

# expr.columns = [m if m is not None else str(g) for g, m in zip(expr.columns, mapped)]
# assert expr.columns.is_unique, "duplicate gene ids"
orig = [str(g) for g in expr.columns]
final = [m if m is not None else g for g, m in zip(orig, mapped)]

# the two legacy SLC35E2 columns -> NCBI's SLC35E2A (9906) and SLC35E2B (728661)
slc_pos = [i for i, g in enumerate(orig) if g == "SLC35E2"]
assert len(slc_pos) == 2
for pos, gid in zip(slc_pos, ["9906", "728661"]):
    final[pos] = gid
overrides = {"B3GNT1": "11041", "CHP": "11261", "RAGE": "5891",
             "HN1L": "90861", "CD97": "976"}
for i, g in enumerate(orig):
    if g in overrides:
        assert final[i] == g, f"{g} was already mapped to {final[i]}"   # only touch unmapped ones
        final[i] = overrides[g]

print("AGER present as its own column:", "AGER" in orig or "177" in orig)

expr.columns = final
assert expr.columns.is_unique, "duplicate gene ids"
gene_ids = list(expr.columns)
print("final gene count:", len(gene_ids))      # must be 20531

lm_check = pd.read_csv(BASE.parent / "data" / "tcga_union_978landmark_genes.csv", dtype=str)
print("SLC35E2A/B in landmark list:", {"9906", "728661"} & set(lm_check.iloc[:, 0]))



gene_ids = list(expr.columns)
print("final gene count:", len(gene_ids))      # must be 20531

lm = pd.read_csv(BASE.parent / "data" / "tcga_union_978landmark_genes.csv", dtype=str)
print(lm.columns.tolist(), lm.shape)
lm_ids = set(lm.iloc[:, 0])
missing = sorted(lm_ids - set(gene_ids))
print(f"landmarks found: {len(lm_ids) - len(missing)} / {len(lm_ids)}", missing[:30])

# gi_all = gi.set_index("GeneID")
# unm = set(unmapped)
# orig_set = set(orig)
# for gid in missing:
#     row = gi_all.loc[gid]
#     names = {row["Symbol"]}
#     if row["Synonyms"] != "-":
#         names |= set(row["Synonyms"].split("|"))
#     print(gid, row["Symbol"], "| synonyms:", row["Synonyms"],
#           "| unmapped Xena cols matching:", sorted(names & unm),
#           "| any Xena col with those names:", sorted(names & orig_set))
# import sys; sys.exit()    # temporary
assert not missing, "fix landmark ids before continuing"

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
assert expr.index.str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}").all()
code = expr.index.str[13:15].astype(int)
meta = pd.DataFrame({"full_barcode": expr.index, "patient": expr.index.str[:12],
                     "cancer": (code < 10).astype(int)})
meta = meta.merge(pat, left_on="patient", right_index=True, how="inner")
meta = meta.dropna(subset=["cohort", "age", "gender"])

print(meta["gender"].value_counts(dropna=False))      # see what values exist
meta["gender"] = meta["gender"].astype(str).str.strip().str.lower().map({"male": 0.0, "female": 1.0})
meta = meta.dropna(subset=["gender"])                 # drops "[Not Available]" etc.


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


# --- verification block
gene_cols = X.columns.astype(str).tolist()

assert len(df) == len(X)
assert df.columns.is_unique
assert gene_cols == list(map(str, gene_ids))
assert df[gene_cols].shape[1] == len(gene_ids)

assert df[gene_cols].apply(lambda c: pd.api.types.is_numeric_dtype(c)).all()
assert np.isfinite(df[gene_cols].to_numpy()).all()
assert not df[gene_cols].isna().any().any()

assert set(df["cancer"].unique()) <= {0, 1}
assert set(df["gender"].unique()) <= {0.0, 1.0}
assert df["age"].between(0, 120).all()
assert df["tissue_type"].notna().all()
assert df["cohort"].isin(COHORTS).all()

print("Samples:", len(df))
print("Genes:", len(gene_cols))
print("Tissues:", df["tissue_type"].nunique())
print("Cohorts:", df["cohort"].nunique())
print(df["cancer"].value_counts(normalize=True))
print(df["tissue_type"].value_counts().sort_index())

# --- split: ~1/6 test; the loader then splits train 80/20 into train/val
# train, test = train_test_split(df, test_size=1/6, random_state=42, stratify=df["tissue_type"])

from data.utils_preprocessing import train_test_split as repo_split
np.random.seed(42)
train, test = repo_split(df, split_ratios=[0.8, 0.2], shuffle=True)
print(len(train), len(test))
np.save(f"{OUT}/TCGA_rnaseq_RSEM_gene_ids_full.npy", np.array(gene_ids))
train.to_csv(f"{OUT}/train_df_covariates.csv", index=False, float_format="%.6g")
test.to_csv(f"{OUT}/test_df_covariates.csv",  index=False, float_format="%.6g")

# --- verification block
assert list(train.columns) == list(test.columns)
assert set(train["full_barcode"]).isdisjoint(test["full_barcode"])
# assert set(train["reduced_barcode"]).isdisjoint(test["reduced_barcode"])
assert train["tissue_type"].nunique() == test["tissue_type"].nunique()





# after building df
# landmarks = pd.read_csv(
#     "data/tcga_union_978landmark_genes.csv",
#     dtype=str
# )["genes_ids"].tolist()

# missing = sorted(set(landmarks) - set(gene_cols))
# print("Landmarks available:", len(landmarks) - len(missing), "/ 978")
# assert not missing, f"Missing landmark IDs: {missing[:10]}"