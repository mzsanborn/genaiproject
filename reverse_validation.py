"""
Reverse validation ("train on synthetic, test on real") for the DDIM baseline and CFG models.

One multi-task MLP predicts cancer type, sex, race and age from the 978 landmark genes
(shared trunk, four heads; the targets are correlated, so they are learned jointly).
Three settings, each repeated over several seeds:

  TRTR  train on REAL train patients       -> test on REAL test patients   (reference)
  TSTR  train on SYNTHETIC samples         -> test on REAL test patients   (reverse validation)
  TRTS  train on REAL train patients       -> test on SYNTHETIC samples    (do samples carry label signal?)

Labels of synthetic samples: the generator made sample i from training patient i's
conditioning (tissue for the baseline, cancer type/sex/race/age for CFG), so sample i gets
patient i's cancer type, sex, race and age. This is checked against samples_label.csv.
For the tissue-only baseline, sex/race/age signal can only come through tissue.

All data are compared in the generator's own scaled space (scaler refit on real training
data with the run's scaler_type, which reproduces the training transform exactly); the
classifier then standardises its inputs on its own training set.

Metrics (real test split unless TRTS):
  cancer  accuracy, balanced accuracy, macro-F1      sex  accuracy
  race    accuracy and balanced accuracy over races with >= MIN_RACE_N test patients,
          'unknown' excluded (recorded-but-missing race is not evaluated)
  age     mean absolute error (years)
  plus cancer-type accuracy and age MAE within race / sex / age-group subgroups.

Usage on Colab (after a run has finished; see colab/run_ddim.ipynb):
  python reverse_validation.py --run /content/runs/cfg                       # local run folder
  python reverse_validation.py --model cfg --from_backup genaiproject_backups_standard
  add --runs 5 (seeds), --device cuda, --smoke (2 seeds, few epochs: quick check)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from src.generation.ddim.datasets.utils import get_tcga_datasets, load_tcga  # noqa: E402
from src.generation.ddim.covariates import CANCER_COHORTS, RACES, encode_covariates  # noqa: E402

MIN_RACE_N = 5
EVAL_RACES = [r for r in RACES if r != "unknown"]
AGE_BINS = [0, 39, 59, 74, 200]
AGE_LABELS = ["<40", "40-59", "60-74", "75+"]


# ----------------------------------------------------------------------------- data
def targets(meta):
    return {
        "cancer": meta["cohort"].map({c: i for i, c in enumerate(CANCER_COHORTS)}).to_numpy(np.int64),
        "sex": meta["gender"].to_numpy(np.int64),                       # 1 = female
        "race": meta["race"].map({r: i for i, r in enumerate(RACES)}).to_numpy(np.int64),
        "age": meta["age"].to_numpy(np.float32),
    }


def load_run(run_dir):
    cfg = yaml.unsafe_load(open(os.path.join(run_dir, "logs", "tcga", "config.yml")))
    img = os.path.join(run_dir, "image_samples", "images")
    X = pd.read_csv(os.path.join(img, "samples.csv"), header=None).to_numpy(np.float32)
    lab = pd.read_csv(os.path.join(img, "samples_label.csv"), header=None).to_numpy()
    return cfg, X, lab


def fetch_from_backup(model, backup_root, dest):
    from src.generation.ddim.remote_backup import make_backend
    be = make_backend("drive", f"{backup_root}/{model}")
    for rel in ["logs/tcga/config.yml", "image_samples/images/samples.csv",
                "image_samples/images/samples_label.csv"]:
        ok = be.download(rel, os.path.join(dest, rel))
        if not ok:
            raise FileNotFoundError(f"{rel} not found in My Drive/{backup_root}/{model}")
    print(f"downloaded samples + config from My Drive/{backup_root}/{model}")
    return be


def check_sample_order(cfg, lab, meta_train, y_tissue_train):
    cond = getattr(cfg.data, "conditioning", "tissue")
    if cond == "tissue":
        ok = lab.shape == y_tissue_train.shape and np.array_equal(lab, y_tissue_train)
    else:
        ref = encode_covariates(meta_train).numpy()
        ok = lab.shape == ref.shape and np.allclose(lab, ref, equal_nan=True)
    if not ok:
        raise RuntimeError("samples_label.csv does not match the training patients in order; "
                           "cannot assign demographics to synthetic samples")
    return cond


# ----------------------------------------------------------------------------- model
class MultiTaskMLP(nn.Module):
    def __init__(self, d_in, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.cancer = nn.Linear(d, len(CANCER_COHORTS))
        self.sex = nn.Linear(d, 1)
        self.race = nn.Linear(d, len(RACES))
        self.age = nn.Linear(d, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.cancer(h), self.sex(h).squeeze(1), self.race(h), self.age(h).squeeze(1)


def class_weights(y, n):
    c = np.bincount(y, minlength=n).astype(np.float32)
    w = np.where(c > 0, c.sum() / (n * np.maximum(c, 1)), 0.0)
    return torch.tensor(w / w[c > 0].mean())


def train_classifier(Xtr, Ttr, Xva, Tva, device, seed, max_epochs=200, patience=15, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr)
    age_mu, age_sd = Ttr["age"].mean(), Ttr["age"].std() + 1e-6
    to_t = lambda X: torch.tensor(scaler.transform(X), dtype=torch.float32)
    xtr, xva = to_t(Xtr), to_t(Xva)
    ttr = {k: torch.tensor(v) for k, v in Ttr.items()}
    tva = {k: torch.tensor(v) for k, v in Tva.items()}
    w_c = class_weights(Ttr["cancer"], len(CANCER_COHORTS)).to(device)
    w_r = class_weights(Ttr["race"], len(RACES)).to(device)
    model = MultiTaskMLP(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss_fn(out, t):
        c, s, r, a = out
        return (F.cross_entropy(c, t["cancer"], weight=w_c)
                + F.binary_cross_entropy_with_logits(s, t["sex"].float())
                + F.cross_entropy(r, t["race"], weight=w_r)
                + F.mse_loss(a, (t["age"] - age_mu) / age_sd))

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            t = {k: v[idx].to(device) for k, v in ttr.items()}
            opt.zero_grad()
            loss_fn(model(xtr[idx].to(device)), t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xva.to(device)), {k: v.to(device) for k, v in tva.items()}).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    @torch.no_grad()
    def predict(X):
        c, s, r, a = model(to_t(X).to(device))
        return {"cancer": c.argmax(1).cpu().numpy(), "sex": (s > 0).long().cpu().numpy(),
                "race": r.argmax(1).cpu().numpy(), "age": (a * age_sd + age_mu).cpu().numpy()}
    return predict


# ----------------------------------------------------------------------------- metrics
def metrics(pred, T, meta):
    out = {
        "cancer_acc": accuracy_score(T["cancer"], pred["cancer"]),
        "cancer_bal_acc": balanced_accuracy_score(T["cancer"], pred["cancer"]),
        "cancer_macro_f1": f1_score(T["cancer"], pred["cancer"], average="macro"),
        "sex_acc": accuracy_score(T["sex"], pred["sex"]),
        "age_mae": float(np.abs(T["age"] - pred["age"]).mean()),
    }
    # race: drop 'unknown' and races with too few patients in this evaluation set
    keep_races = [RACES.index(r) for r in EVAL_RACES if (T["race"] == RACES.index(r)).sum() >= MIN_RACE_N]
    m = np.isin(T["race"], keep_races)
    out["race_acc"] = accuracy_score(T["race"][m], pred["race"][m])
    out["race_bal_acc"] = balanced_accuracy_score(T["race"][m], pred["race"][m])
    for r in keep_races:
        mr = T["race"] == r
        out[f"race_recall_{RACES[r]}"] = float((pred["race"][mr] == r).mean())
    # subgroups: cancer-type accuracy and age MAE within each demographic group
    groups = {f"race={r}": (meta["race"] == r).to_numpy() for r in EVAL_RACES}
    groups.update({"sex=female": (meta["gender"] == 1).to_numpy(), "sex=male": (meta["gender"] == 0).to_numpy()})
    bins = pd.cut(meta["age"], AGE_BINS, labels=AGE_LABELS)
    groups.update({f"age={b}": (bins == b).to_numpy() for b in AGE_LABELS})
    for g, mg in groups.items():
        if mg.sum() >= MIN_RACE_N:
            out[f"cancer_acc|{g}"] = accuracy_score(T["cancer"][mg], pred["cancer"][mg])
            out[f"age_mae|{g}"] = float(np.abs(T["age"][mg] - pred["age"][mg]).mean())
            out[f"n|{g}"] = int(mg.sum())
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="local run folder, e.g. /content/runs/cfg")
    ap.add_argument("--model", choices=["baseline", "cfg"], help="with --from_backup")
    ap.add_argument("--from_backup", help="Drive folder under My Drive holding <model>/ backups")
    ap.add_argument("--runs", type=int, default=5, help="seeds per setting")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="results CSV (default: <run>/reverse_validation.csv)")
    ap.add_argument("--smoke", action="store_true", help="2 seeds, 5 epochs: check the pipeline quickly")
    args = ap.parse_args()

    backend = None
    if args.from_backup:
        assert args.model, "--model is required with --from_backup"
        args.run = args.run or f"/content/reverse_validation/{args.model}"
        backend = fetch_from_backup(args.model, args.from_backup, args.run)
    assert args.run, "give --run, or --model with --from_backup"
    name = args.model or os.path.basename(os.path.normpath(args.run))

    cfg, X_syn, lab = load_run(args.run)
    scaler_type = cfg.data.scaler_type
    # real data in the generator's scaled space (scaler fit on real train, as in training)
    tr, va = get_tcga_datasets(scaler_type, conditioning="tissue", eval_split="val")
    _, te = get_tcga_datasets(scaler_type, conditioning="tissue", eval_split="test")
    X_tr, X_va, X_te = (d.tensors[0].numpy() for d in (tr, va, te))
    m_tr, m_va, m_te = (load_tcga(split=s) for s in ("train", "val", "test"))
    cond = check_sample_order(cfg, lab, m_tr, tr.tensors[1].numpy())
    T_tr, T_va, T_te = targets(m_tr), targets(m_va), targets(m_te)
    T_syn, m_syn = T_tr, m_tr                                   # sample i <- training patient i
    print(f"model '{name}' ({cond}-conditioned, {scaler_type} scaling): {len(X_syn)} synthetic samples; "
          f"real train {len(X_tr)}, val {len(X_va)}, test {len(X_te)}")

    n_runs, epochs = (2, 5) if args.smoke else (args.runs, 200)
    rows = []
    t0 = time.time()
    for seed in range(n_runs):
        # TRTR / TRTS: classifier trained on real train, early-stopped on real val
        pred_real = train_classifier(X_tr, T_tr, X_va, T_va, args.device, seed, max_epochs=epochs)
        res = {"TRTR": metrics(pred_real(X_te), T_te, m_te),
               "TRTS": metrics(pred_real(X_syn), T_syn, m_syn)}
        # TSTR: classifier trained on synthetic (80%), early-stopped on held-out synthetic (20%)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X_syn)); cut = int(0.8 * len(perm))
        i_fit, i_es = perm[:cut], perm[cut:]
        sub = lambda T, i: {k: v[i] for k, v in T.items()}
        pred_syn = train_classifier(X_syn[i_fit], sub(T_syn, i_fit), X_syn[i_es], sub(T_syn, i_es),
                                    args.device, seed, max_epochs=epochs)
        res["TSTR"] = metrics(pred_syn(X_te), T_te, m_te)
        for setting, met in res.items():
            for k, v in met.items():
                rows.append({"model": name, "setting": setting, "seed": seed, "metric": k, "value": v})
        print(f"seed {seed}: TRTR cancer acc {res['TRTR']['cancer_acc']:.3f} | "
              f"TSTR cancer acc {res['TSTR']['cancer_acc']:.3f}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(args.run, "reverse_validation.csv")
    df.to_csv(out, index=False)
    summ = (df[~df.metric.str.startswith("n|")].groupby(["metric", "setting"])["value"]
              .agg(["mean", "std"]).unstack("setting"))
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    headline = ["cancer_acc", "cancer_bal_acc", "cancer_macro_f1", "sex_acc", "race_acc", "race_bal_acc", "age_mae"]
    fmt = lambda s: f"{s['mean']:.3f} ± {s['std']:.3f}"
    print(f"\n=== {name}: mean ± std over {n_runs} seeds ===")
    print(f"{'metric':<20}" + "".join(f"{s:>18}" for s in ["TRTR (real)", "TSTR (synthetic)", "TRTS"]))
    for k in headline + sorted(m for m in summ.index if m not in headline):
        if k in summ.index:
            vals = [fmt({"mean": summ.loc[k, ("mean", s)], "std": summ.loc[k, ("std", s)]})
                    if ("mean", s) in summ.columns and not np.isnan(summ.loc[k, ("mean", s)]) else "-"
                    for s in ["TRTR", "TSTR", "TRTS"]]
            print(f"{k:<20}" + "".join(f"{v:>18}" for v in vals))
    print(f"\nper-seed results: {out}")
    if backend is not None:
        backend.upload(out, "reverse_validation.csv")
        print(f"uploaded to My Drive/{args.from_backup}/{args.model}/reverse_validation.csv")


if __name__ == "__main__":
    main()
