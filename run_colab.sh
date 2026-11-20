#!/bin/bash
# Baseline + CFG DDIM on Colab (fits a T4: ~63M params, fp16 autocast).
# Normally launched from colab/run_ddim.ipynb. By hand, in a Colab cell:
#   from google.colab import auth; auth.authenticate_user()     # Drive API access
#   !git clone -b cfg-demographics <repo-url> /content/genaiproject
#   %cd /content/genaiproject
#   !pip install -q pytorch_warmup
#   !bash run_colab.sh baseline      # or: cfg
#
# Storage model (the standard pattern for ephemeral GPU machines):
#   * Training reads/writes ONLY Colab's local disk (EXP_ROOT, default /content/runs).
#   * Every ~20 min the latest checkpoint is uploaded to Google Drive through the
#     Drive API (My Drive/$BACKUP_ROOT/<model>/...). The upload only counts as done
#     when Drive has stored the file. Only the latest copy is kept; older ones are
#     permanently deleted, so each model uses ~1 GB of Drive (2 GB during an upload).
#   * Samples, metrics and logs are uploaded when the model finishes.
#   * We deliberately do NOT write into a mounted Drive folder: that mount uploads
#     asynchronously and silently stopped uploading on 2026-11-20.
#
# Resuming: re-running the same command resumes from the local checkpoint, or, on a
# fresh Colab machine, from the Drive backup; with neither it starts from scratch.
#   RESUME=0 bash run_colab.sh cfg   -> DELETE the local run and start over
#                                       (the Drive backup is replaced at the next upload)
#
# Data: newdata/data_RNAseq_RTCGA_L978/{train,val,test}_df_covariates.csv
# (primary tumors only, one sample per patient, 80/10/10 patient-level split).
set -e
cd "$(dirname "$0")"

WHICH=${1:-baseline}
EXP_ROOT=${EXP_ROOT:-/content/runs}
BACKUP_ROOT=${BACKUP_ROOT:-genaiproject_backups}   # folder under My Drive
BACKUP_BACKEND=${BACKUP_BACKEND:-drive}
BACKUP_EVERY_MIN=${BACKUP_EVERY_MIN:-20}

case "$WHICH" in
  baseline) CONFIG=tcga_landmark_ddim_baseline.yml ;;  # tissue-conditioned replication
  cfg)      CONFIG=tcga_landmark_ddim_cfg.yml ;;       # cancer type + sex + race + age, CFG
  *) echo "usage: $0 baseline|cfg"; exit 1 ;;
esac

RESUME_FLAG="--resume_training"   # main_ddim.py falls back to a fresh start if no checkpoint exists
if [ "${RESUME:-1}" = "0" ]; then
  RESUME_FLAG=""
  echo "RESUME=0: starting $WHICH from scratch"
fi

# --with_best_params y : Lacan et al. tuned hyperparameters (4096x2 layers, quad schedule, lr 1.4e-3)
# --timesteps 1000     : DDIM sampling steps (lower, e.g. 100, for quicker sampling)
python main_ddim.py --config "$CONFIG" --with_best_params y \
    --exp "$EXP_ROOT/$WHICH" --doc tcga --sample --fid \
    --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:0" $RESUME_FLAG \
    --backup "$BACKUP_ROOT/$WHICH" --backup_backend "$BACKUP_BACKEND" --backup_every_min "$BACKUP_EVERY_MIN"
