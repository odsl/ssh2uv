#!/bin/bash
#SBATCH --job-name=ssh2uv_figures
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# All paper figures (CPU only). Figures 4-6 need slurm/predict.slurm, Figure 8 needs
# slurm/random_split_eval.slurm; Figures 3 and 7 use the shipped results/ only.
#   VENV_PATH=/path/to/venv sbatch slurm/figures.slurm
source slurm/common.sh
python scripts/figures/fig_snapshot.py             --out figures/snapshot
python scripts/figures/fig_unet.py                 --out figures/unet_architecture
python scripts/figures/fig_leakage.py              --out figures/leakage
python scripts/figures/fig_progression.py          --out figures/progression
python scripts/figures/fig_kinematics_map.py       --out figures/kinematics_map
python scripts/figures/fig_predictions.py          --out figures/predictions
python scripts/figures/fig_vorticity_strain_pdf.py --out figures/vorticity_strain_pdf
python scripts/figures/fig_spectral_flux.py        --out figures/spectral_flux
echo "Figures written to figures/"
