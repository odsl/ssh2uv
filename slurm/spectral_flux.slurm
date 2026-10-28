#!/bin/bash
#SBATCH --job-name=ssh2uv_flux
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
# Spectral KE flux (Section 6) of the true, geostrophic, baseline and final velocities on the
# test period, at 53 m and at the surface. From the repository root:
#   VENV_PATH=/path/to/venv sbatch slurm/spectral_flux.slurm
source slurm/common.sh
for depth in 53m surface; do
    if [ "$depth" = 53m ]; then F=global-final-64_stacked_DEEP.npy; P=""; else F=global-final-64_stacked_FULL.npy; P="S_"; fi
    M=""
    for s in 0 1 2; do
        M="$M --model c3=runs/temporal/${P}C3_bc128_phys_long/seed$s/checkpoint_best.pt"
        M="$M --model e0=runs/temporal/${P}E0_baseline/seed$s/checkpoint_best.pt"
    done
    python scripts/spectral_flux.py --block test --data_file "$F" --out "outputs/spectral_flux_$depth" $M
done
