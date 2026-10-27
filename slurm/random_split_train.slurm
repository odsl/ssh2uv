#!/bin/bash
#SBATCH --job-name=ssh2uv_random_split
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
# Train one model of the earlier random-split version (leakage analysis, Section 4.1).
#   VENV_PATH=/path/to/venv CONFIG=configs/random_split/grad_bc192/grad_ab_sst_wind_bc192.json \
#       sbatch slurm/random_split_train.slurm
# The output directory is set in the config (runs/random_split/...).
source slurm/common.sh
python scripts/random_split/train.py --config "${CONFIG:?set CONFIG}" ${EXTRA:-}
