#!/bin/bash
# Baseline + CFG DDIM on Colab (fits a T4: ~30M params, fp16 autocast).
#
# In a Colab cell:
#   from google.colab import drive; drive.mount('/content/drive')
#   !git clone -b cfg-demographics <repo-url> /content/genaiproject
#   %cd /content/genaiproject
#   !pip install -q pytorch_warmup
#   !bash run_colab.sh baseline      # or: cfg
#
# Checkpoints go to Google Drive (EXP_ROOT) every ~2000 steps, at the end of an
# epoch. Re-running the same command RESUMES from the last checkpoint if there is
# one (and starts fresh if there isn't). To throw a run away and start over:
#   !RESUME=0 bash run_colab.sh cfg
# WARNING: with RESUME=0 the existing run folder for that model is DELETED.
#
# Data: newdata/data_RNAseq_RTCGA_L978/{train,val,test}_df_covariates.csv
# (primary tumors only, one sample per patient, 80/10/10 patient-level split).
set -e
cd "$(dirname "$0")"

WHICH=${1:-baseline}
EXP_ROOT=${EXP_ROOT:-/content/drive/MyDrive/genaiproject_runs}

case "$WHICH" in
  baseline) CONFIG=tcga_landmark_ddim_baseline.yml ;;  # tissue-conditioned replication
  cfg)      CONFIG=tcga_landmark_ddim_cfg.yml ;;       # cancer type + sex + race + age, CFG
  *) echo "usage: $0 baseline|cfg"; exit 1 ;;
esac

RESUME_FLAG=""
if [ "${RESUME:-1}" = "1" ] && [ -f "$EXP_ROOT/$WHICH/logs/tcga/ckpt.pth" ]; then
  RESUME_FLAG="--resume_training"
  echo "Resuming $WHICH from $EXP_ROOT/$WHICH/logs/tcga/ckpt.pth"
else
  echo "Starting $WHICH from scratch"
fi

# --with_best_params y : Lacan et al. tuned hyperparameters (4096x2 layers, quad schedule, lr 1.4e-3)
# --timesteps 1000     : DDIM sampling steps (lower, e.g. 100, for quicker sampling)
python main_ddim.py --config "$CONFIG" --with_best_params y \
    --exp "$EXP_ROOT/$WHICH" --doc tcga --sample --fid \
    --timesteps 1000 --eta 0 --ni --device "[0]" --device2 "cuda:0" $RESUME_FLAG
