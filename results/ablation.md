Geostrophic reference (no training): test RMSE speed 0.0687 m/s, dir. err 18.4°, r(ζ) 0.749, r(δ) 0; val RMSE speed 0.0768 m/s.

## 1. SSH representation × covariates (width 128, MSE + physics penalties)

| covariates | raw SSH: test RMSE speed | SSH gradients: test RMSE speed | change | raw SSH: test r(ζ) / r(δ) | gradients: test r(ζ) / r(δ) |
|---|---|---|---|---|---|
| none | 0.0597 ± 0.0002 | 0.0551 ± 0.0015 | -7.6 % | 0.801 / 0.249 | 0.868 / 0.417 |
| SST | 0.0592 ± 0.0018 | 0.0581 ± 0.0020 | -1.9 % | 0.830 / 0.334 | 0.881 / 0.478 |
| wind | 0.0487 ± 0.0005 | 0.0489 ± 0.0005 | +0.3 % | 0.892 / 0.659 | 0.927 / 0.644 |
| SST + wind | 0.0510 ± 0.0002 | 0.0486 ± 0.0007 | -4.6 % | 0.901 / 0.684 | 0.931 / 0.661 |

Same runs with validation scores:

| model | seeds | test RMSE speed (m/s) | test dir. err (°) | test r(ζ) | test r(δ) | val RMSE speed (m/s) |
|---|---|---|---|---|---|---|
| raw SSH | 3 | 0.0597 ± 0.0002 | 17.5 ± 0.1 | 0.801 ± 0.007 | 0.249 ± 0.017 | 0.0650 ± 0.0012 |
| raw SSH + SST | 3 | 0.0592 ± 0.0018 | 17.0 ± 0.3 | 0.830 ± 0.003 | 0.334 ± 0.022 | 0.0597 ± 0.0002 |
| raw SSH + wind | 3 | 0.0487 ± 0.0005 | 14.5 ± 0.2 | 0.892 ± 0.002 | 0.659 ± 0.010 | 0.0527 ± 0.0006 |
| raw SSH + SST + wind | 3 | 0.0510 ± 0.0002 | 14.5 ± 0.1 | 0.901 ± 0.003 | 0.684 ± 0.007 | 0.0508 ± 0.0002 |
| gradients | 3 | 0.0551 ± 0.0015 | 16.1 ± 0.3 | 0.868 ± 0.002 | 0.417 ± 0.025 | 0.0606 ± 0.0007 |
| gradients + SST | 3 | 0.0581 ± 0.0020 | 16.3 ± 0.4 | 0.881 ± 0.001 | 0.478 ± 0.010 | 0.0562 ± 0.0003 |
| gradients + wind | 3 | 0.0489 ± 0.0005 | 14.5 ± 0.3 | 0.927 ± 0.002 | 0.644 ± 0.014 | 0.0521 ± 0.0005 |
| gradients + SST + wind (C3) | 3 | 0.0486 ± 0.0007 | 14.0 ± 0.1 | 0.931 ± 0.003 | 0.661 ± 0.020 | 0.0490 ± 0.0005 |

## 2. Capacity (gradients + SST + wind, MSE + physics penalties)

| model | seeds | test RMSE speed (m/s) | test dir. err (°) | test r(ζ) | test r(δ) | val RMSE speed (m/s) |
|---|---|---|---|---|---|---|
| width 64 (31M) | 3 | 0.0511 ± 0.0009 | 14.4 ± 0.2 | 0.921 ± 0.002 | 0.614 ± 0.016 | 0.0512 ± 0.0006 |
| width 128 (124M, C3) | 3 | 0.0486 ± 0.0007 | 14.0 ± 0.1 | 0.931 ± 0.003 | 0.661 ± 0.020 | 0.0490 ± 0.0005 |
| width 192 (279M) | 3 | 0.0490 ± 0.0020 | 13.9 ± 0.2 | 0.936 ± 0.003 | 0.686 ± 0.018 | 0.0486 ± 0.0007 |

## 3. Training objective (gradients + SST + wind, width 128)

| model | seeds | test RMSE speed (m/s) | test dir. err (°) | test r(ζ) | test r(δ) | val RMSE speed (m/s) |
|---|---|---|---|---|---|---|
| MSE | 3 | 0.0533 ± 0.0009 | 14.4 ± 0.2 | 0.918 ± 0.006 | 0.545 ± 0.044 | 0.0520 ± 0.0003 |
| MAE | 3 | 0.0526 ± 0.0009 | 14.4 ± 0.3 | 0.920 ± 0.005 | 0.562 ± 0.035 | 0.0520 ± 0.0006 |
| MAE + vorticity/divergence penalties | 3 | 0.0512 ± 0.0007 | 14.2 ± 0.3 | 0.934 ± 0.005 | 0.674 ± 0.018 | 0.0494 ± 0.0004 |
| MSE + vorticity/divergence penalties (C3) | 3 | 0.0486 ± 0.0007 | 14.0 ± 0.1 | 0.931 ± 0.003 | 0.661 ± 0.020 | 0.0490 ± 0.0005 |
