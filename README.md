# ssh2uv: from sea surface height to velocity

Code for the paper **"From Sea Surface Height to Velocity: A Machine Learning Approach"**
(Y. R. Yaradoddi, J. Wang, E. Serghine).

A U-Net predicts the horizontal velocity at 53 m depth (LLC4320 model level 16), and at the surface (0.5 m), from
sea surface height (SSH), sea surface temperature, and wind stress. It is trained on
hourly fields from the pre-SWOT LLC4320 Cape Basin region (11.8–15.8°E, 41–45°S) and
evaluated on a held-out period.

## Main result

Held-out test period, 21 August – 12 November 2012. Mean ± standard deviation over 3 seeds:

| model | RMSE u (m/s) | RMSE v (m/s) | RMSE speed (m/s) | r(ζ) | r(δ) |
|---|---|---|---|---|---|
| f-plane geostrophy from SSH | 0.0645 | 0.0655 | 0.0687 | 0.749 | 0 |
| final U-Net (`C3_bc128_phys_long`) | 0.0460 ± 0.0003 | 0.0464 ± 0.0003 | 0.0486 ± 0.0007 | 0.931 | 0.661 |
| *surface (0.5 m):* geostrophy | 0.1389 | 0.1114 | 0.1249 | 0.762 | 0 |
| *surface (0.5 m):* final U-Net (`S_C3_bc128_phys_long`) | 0.0488 | 0.0494 | 0.0511 ± 0.0009 | 0.973 | 0.874 |

Here ζ is relative vorticity and δ horizontal divergence. The final U-Net also reproduces the
spectral kinetic-energy flux (inverse cascade above 20–30 km, forward cascade below), which
geostrophy misrepresents (paper Section 6). The tables of the paper can
be regenerated from the shipped per-run results in `results/` (see below).

## Repository layout

```
configs/
  temporal/              one JSON per temporal-split experiment (main results)
  random_split/          earlier random-split experiments (leakage analysis)
notebooks/
  01_download_llc4320.ipynb   download the Cape Basin data from NASA PO.DAAC
src/ssh2uv/              Python package
  pipeline/              temporal-split data loading, augmentation, inputs/targets
  models/unet.py         U-Net
  kinematics.py          constants (2 km grid, f0), vorticity, divergence, strain, geostrophy
  spectral.py            isotropic KE spectrum and spectral KE flux
  evaluation.py          checkpoint loading and metrics
  losses.py, metrics.py, config.py, data/   used by the random-split pipeline
  plotting.py            figure style
scripts/
  preprocess.py          regrid to 2 km and cut 64x64 patches
  make_temporal_split.py train / validation / test blocks
  train.py               train and evaluate one temporal-split experiment
  predict.py             cache test predictions (and metrics) for the figures
  spectral_flux.py       KE spectra and spectral fluxes on the whole domain
  summarize_runs.py      results table over all experiments
  summarize_ablation.py  ablation tables
  random_split/          earlier pipeline: train, evaluate, evaluate on the temporal split
  figures/               one script per paper figure
slurm/                   job scripts (see below)
tests/                   unit tests (pytest)
results/                 per-run test/validation metrics of all paper experiments
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or: pip install -r requirements.txt && pip install -e .
pytest                           # unit tests, CPU only
```

The paper runs used Python 3.12, PyTorch 2.10, and one NVIDIA H200 GPU per run.

## Reproducing the paper

All commands run from the repository root. On a SLURM cluster, the scripts in
`slurm/` wrap these steps. Set `VENV_PATH` (and optionally `MODULES`) as described in
`slurm/common.sh`.

**1. Data.** Create a NASA Earthdata login, export `EARTHDATA_TOKEN`, and run
`notebooks/01_download_llc4320.ipynb`. It streams the 429 daily granules of
[`MITgcm_LLC4320_Pre-SWOT_JPL_L4_CapeBasin_v1.0`](https://doi.org/10.5067/PRESW-CBJ10)
and writes per-variable arrays to `cape_processed_data/`. Then:

```bash
python scripts/preprocess.py           # -> data/global-final-64_stacked_{DEEP,DEEP_GRAD,FULL_GRAD}.npy
python scripts/make_temporal_split.py  # -> data/temporal_split/
```

**2. Main experiments** (Section 4). Each config is trained with seeds 0, 1, 2:

```bash
python scripts/train.py --config configs/temporal/C3_bc128_phys_long.json --seed 0
# or: CONFIG=configs/temporal/C3_bc128_phys_long.json SEED=0 sbatch slurm/train.slurm
```

Each run writes `runs/temporal/<name>/seed<k>/checkpoint_best.pt` and `results.json`.
Training stops early on the validation block; the test block is evaluated once, with
the best checkpoint.

Surface-velocity experiments use the `configs/temporal/S_*.json` configs (target file
`global-final-64_stacked_FULL.npy`).

**Spectral energy flux** (Section 6): `sbatch slurm/spectral_flux.slurm`, or
`scripts/spectral_flux.py` with the C3 and E0 checkpoints of each depth.

**3. Random versus temporal split** (Appendix B). Train the random-split models with
`scripts/random_split/train.py` and the configs in `configs/random_split/`. Then run
`slurm/random_split_eval.slurm`.

**4. Tables and figures.**

```bash
python scripts/summarize_runs.py       # all experiments  -> results/summary.md
python scripts/summarize_ablation.py   # Tables 3-5        -> results/ablation.md
python scripts/predict.py --model c3=runs/temporal/C3_bc128_phys_long/seed0/checkpoint_best.pt \
                          --model e0=runs/temporal/E0_baseline/seed0/checkpoint_best.pt
sbatch slurm/figures.slurm             # or run the scripts in scripts/figures/
```

| figure | script | needs |
|---|---|---|
| 1 snapshot | `fig_snapshot.py` | `data/` |
| 2 architecture | `fig_unet.py` | – |
| 3 progression | `fig_progression.py` | `results/` (shipped) |
| 4 predictions | `fig_predictions.py` | `outputs/test_predictions/` (`predict.py`); run `fig_kinematics_map.py` first |
| 5 vorticity/divergence | `fig_kinematics_map.py` | `outputs/test_predictions/` |
| 6 vorticity–strain PDF | `fig_vorticity_strain_pdf.py` | `outputs/test_predictions/` |
| 7 spectral energy flux | `fig_spectral_flux.py` | `results/spectral_flux/` (shipped) |
| 8 random vs temporal split (App. B) | `fig_leakage.py` | `outputs/random_split_eval/` (step 3) |

The summary scripts and Figures 3 and 7 read `results/` by default, which contains the
metrics of all paper runs. They therefore work without retraining. Pass
`--runs runs/temporal` to use your own runs.

## Data split

The 10,196 hourly time steps (each with six 64×64 patches of 2 km cells) are split
into consecutive blocks separated by 48-hour gaps:

| block | period |
|---|---|
| training | 15 Sep 2011 – 8 Jul 2012 |
| validation | 10 Jul – 19 Aug 2012 |
| test | 21 Aug – 12 Nov 2012 |

Normalization statistics are computed from training samples only.

If `data/` contains no split file, `scripts/random_split/train.py` falls back to a
random 80/20 split of patches, which is compared with the temporal split in Appendix B
of the paper. Do not use it to report skill.

## License

MIT; see `LICENSE`. To cite this work, see `CITATION.cff`.
