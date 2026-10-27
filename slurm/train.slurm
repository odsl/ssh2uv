#!/bin/bash
#SBATCH --job-name=ssh2uv_train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
# Train one temporal-split experiment (one seed). From the repository root:
#   VENV_PATH=/path/to/venv CONFIG=configs/temporal/C3_bc128_phys_long.json SEED=0 sbatch slurm/train.slurm
# Writes runs/temporal/<name>/seed<SEED>/{checkpoint_best.pt,results.json}.
source slurm/common.sh
python scripts/train.py --config "${CONFIG:?set CONFIG}" --seed "${SEED:-0}" ${EXTRA:-}
