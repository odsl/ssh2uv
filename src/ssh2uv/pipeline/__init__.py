"""Main experiment pipeline: temporal split, training-only statistics, augmentation.

Raw channels are always ``[eta, tau_x, tau_y, theta, u, v]`` on the 2 km grid with
row 0 at the southern edge. Model inputs (SSH, SSH gradients, geostrophic velocity,
SST, wind stress) are derived from these on the GPU, after augmentation.
"""
