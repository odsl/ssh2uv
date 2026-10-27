#!/bin/bash
#SBATCH --job-name=ssh2uv_random_split_eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
# Leakage analysis (Section 4.1): evaluate the random-split models on their validation
# set, and the Grad+All configuration (random split and temporal retrain) on the
# blocks of the temporal split. From the repository root:
#   VENV_PATH=/path/to/venv sbatch slurm/random_split_eval.slurm
source slurm/common.sh
python scripts/random_split/evaluate.py --results_dir runs/random_split --outdir outputs/random_split_eval
python scripts/random_split/evaluate_temporal.py \
    --split data/temporal_split/global-final-64_master_metadata.json \
    --model temporal=runs/random_split/temporal_retrain/grad_ab_sst_wind_bc192/checkpoint_best.pt \
    --model random=runs/random_split/grad_bc192/grad_ab_sst_wind_bc192/checkpoint_best.pt \
    --out outputs/temporal_retrain_metrics.json
