# Experiment configs

## `temporal/`: main experiments (Sections 4.2 and 4.3)

One JSON file per experiment, read by `scripts/train.py`. Keys missing from a file take
the defaults in `DEFAULTS` at the top of `scripts/train.py`:

| key | default | meaning |
|---|---|---|
| `inputs` | `["grad", "sst", "wind"]` | input groups: `eta` (raw SSH), `grad` (SSH gradients), `ug` (geostrophic velocity), `sst`, `wind` |
| `target` | `"uv"` | `uv` (velocity) or `residual` (departure from f-plane geostrophy) |
| `base_channels` | 64 | U-Net base width B |
| `rotate`, `crops` | false | random 90° rotations; random-position crops |
| `loss` | `"mse"` | `mse` or `mae` |
| `vort_weight`, `div_weight` | 0 | weights of the vorticity and divergence penalties |
| `ug_smooth_px` | 0 | Gaussian smoothing (grid cells) of SSH before computing geostrophy |
| `lr`, `weight_decay`, `batch_size` | 3e-4, 1e-4, 128 | optimizer settings |
| `max_epochs`, `patience` | 60, 10 | training length and early-stopping patience |

Experiment names follow the paper (Table 2 and Appendix A):
- `E*`: baseline and augmentation steps;
- `R*`: round 2;
- `C*`: round 3 (`C3_bc128_phys_long` is the final model);
- `A_*`: ablations around C3 (Tables 3–5).

## `random_split/`: earlier version (Section 4.1, leakage analysis)

Configs of the earlier random-split experiments, read by `scripts/random_split/train.py`:

- `ssha_followup/`: raw SSH with each covariate set, B=192.
- `grad_bc192/`, `grad_bc384/`: SSH gradients with each covariate set, B=192 and B=384. `_mse` and `_l1vd` are the loss variants.
- `temporal_retrain/`: the Grad+All configuration retrained on the temporal split.

Set `"wandb": {"enabled": false}` or `WANDB_MODE=disabled` to train without Weights & Biases.
