#!/bin/bash
#SBATCH --job-name=ssh2uv_predict
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
# Cache test-block predictions of the final (C3) and baseline (E0) models, seed 0,
# for Figures 4-6. From the repository root:
#   VENV_PATH=/path/to/venv sbatch slurm/predict.slurm
source slurm/common.sh
python scripts/predict.py --block test --out outputs/test_predictions \
    --model c3=runs/temporal/C3_bc128_phys_long/seed0/checkpoint_best.pt \
    --model e0=runs/temporal/E0_baseline/seed0/checkpoint_best.pt
