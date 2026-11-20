"""
Covariate encoding shared by the CFG data loader, model and sampler.

A conditioning tensor y has shape (batch, 4), dtype float32, columns:

    0  cancer   integer index into CANCER_COHORTS          (null = N_CANCER)
    1  sex      0 = male, 1 = female                       (null = N_SEX)
    2  race     integer index into RACES (incl. "unknown")       (null = N_RACE)
    3  age      (age - AGE_MEAN) / AGE_SCALE, continuous    (null = NaN)

"Null" is the classifier-free-guidance token: "this covariate is not given".
It is different from race == "unknown", which is a recorded category in the data.
"""
import math
import torch

# 33 TCGA cohorts present in the data (STES is a combined cohort with no own samples)
CANCER_COHORTS = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC",
                  "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV",
                  "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM",
                  "UCEC", "UCS", "UVM"]
SEXES = ["male", "female"]
RACES = ["white", "black", "asian", "aian", "nhpi", "unknown"]  # must match newdata/split_utils.py

N_CANCER, N_SEX, N_RACE = len(CANCER_COHORTS), len(SEXES), len(RACES)

# Fixed (not data-fitted) age normalisation, so no scaler state has to be saved.
# TCGA ages are 14-90 with median ~61.
AGE_MEAN, AGE_SCALE = 60.0, 15.0

COLS = {"cancer": 0, "sex": 1, "race": 2, "age": 3}
NULL_VALUE = {"cancer": float(N_CANCER), "sex": float(N_SEX), "race": float(N_RACE),
              "age": float("nan")}
N_COV = len(COLS)


def encode_covariates(df):
    """DataFrame with cohort / gender (0/1) / race / age columns -> float tensor (n, 4)."""
    cancer_idx = df["cohort"].map({c: i for i, c in enumerate(CANCER_COHORTS)})
    race_idx = df["race"].map({r: i for i, r in enumerate(RACES)})
    assert cancer_idx.notna().all(), set(df["cohort"][cancer_idx.isna()])
    assert race_idx.notna().all(), set(df["race"][race_idx.isna()])
    assert df["gender"].isin([0, 1]).all()
    y = torch.stack([
        torch.tensor(cancer_idx.to_numpy(), dtype=torch.float32),
        torch.tensor(df["gender"].to_numpy(), dtype=torch.float32),
        torch.tensor(race_idx.to_numpy(), dtype=torch.float32),
        torch.tensor(((df["age"].astype(float) - AGE_MEAN) / AGE_SCALE).to_numpy(), dtype=torch.float32),
    ], dim=1)
    return y


def make_y(n, cancer=None, sex=None, race=None, age=None, device=None):
    """
    Build a conditioning tensor for sampling. Pass names/values; None = null.
        make_y(100, cancer="BRCA", sex="female", race="black", age=55)
    """
    y = torch.empty(n, N_COV)
    y[:, 0] = NULL_VALUE["cancer"] if cancer is None else CANCER_COHORTS.index(cancer)
    y[:, 1] = NULL_VALUE["sex"] if sex is None else SEXES.index(sex)
    y[:, 2] = NULL_VALUE["race"] if race is None else RACES.index(race)
    y[:, 3] = NULL_VALUE["age"] if age is None else (float(age) - AGE_MEAN) / AGE_SCALE
    return y.to(device) if device is not None else y


def null_out(y, names):
    """Return a copy of y with the given covariates replaced by the null token."""
    y = y.clone()
    for name in names:
        y[:, COLS[name]] = NULL_VALUE[name]
    return y


def drop_conditions(y, p_each: float, p_all: float):
    """
    CFG training-time condition dropout.
    With probability p_all a row loses every covariate (trains the fully
    unconditional model); otherwise each covariate is independently nulled with
    probability p_each (trains every partial-conditioning combination, so that
    at sampling time we can guide on e.g. race alone while keeping cancer type).
    """
    if p_each <= 0 and p_all <= 0:
        return y
    y = y.clone()
    n = y.shape[0]
    drop_all = torch.rand(n, device=y.device) < p_all
    for name, col in COLS.items():
        drop = drop_all | (torch.rand(n, device=y.device) < p_each)
        y[drop, col] = NULL_VALUE[name]
    return y
