#!/usr/bin/env python3
"""Evaluate random-split-pipeline checkpoints on the blocks of the temporal split.

Used for the leakage analysis (Appendix B, Fig. 8b) and for the geostrophic reference
on the validation and test blocks.

For each checkpoint, reports velocity and kinematic metrics on the validation and
held-out test blocks, the test error as a function of time since the end of the
training block, and the f-plane geostrophic baseline on the same samples.

Usage:
    python scripts/random_split/evaluate_temporal.py --split data/temporal_split/global-final-64_master_metadata.json \
        --model temporal=runs/random_split/temporal_retrain/grad_ab_sst_wind_bc192/checkpoint_best.pt \
        --model random=runs/random_split/grad_bc192/grad_ab_sst_wind_bc192/checkpoint_best.pt \
        --out outputs/temporal_retrain_metrics.json
"""

import argparse
import json
import os

import numpy as np
import torch

from ssh2uv.data.dataset import StackedNumpyDataset
from ssh2uv.evaluation import (
    DEEP_CHANNELS, DEEP_GRAD_CHANNELS, OUTPUT_VARS, kinematic_metrics, load_checkpoint, predict,
    velocity_metrics,
)
from ssh2uv.kinematics import N_PATCHES, geostrophic_velocity

INPUT_VARS = ["a", "b", "theta", "windx", "windy"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model", action="append", required=True, help="name=path/to/checkpoint_best.pt")
    p.add_argument("--out", default="outputs/temporal_retrain_metrics.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def score(pred, truth, mask):
    return {**velocity_metrics(pred, truth, mask), **kinematic_metrics(pred, truth, mask)}


def by_lead_time(pred, truth, idx, t_train_end, edges_days=(0, 10, 20, 40, 60, 90)):
    """RMSE of speed binned by days since the end of the training block."""
    days = (np.asarray(idx) // N_PATCHES - t_train_end) / 24.0
    out = {}
    for lo, hi in zip(edges_days[:-1], edges_days[1:]):
        m = (days >= lo) & (days < hi)
        if m.any():
            sp = np.hypot(pred[m, 0], pred[m, 1]) - np.hypot(truth[m, 0], truth[m, 1])
            out[f"{lo}-{hi} d"] = float(np.sqrt(np.mean(sp ** 2)))
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    meta = json.load(open(args.split))
    blocks = {"val": meta["test_indices"], "test": meta["heldout_test_indices"]}
    t_train_end = meta["split"]["hours"]["train"][1]
    results = {"split": meta["split"]}

    raw = np.load(os.path.join(args.data_dir, "global-final-64_stacked_DEEP.npy"), mmap_mode="r")
    for name, idx in blocks.items():
        eta = np.stack([raw[i, 0] for i in idx]).astype(np.float64)
        truth = np.stack([raw[i, 4:6] for i in idx]).astype(np.float32)
        ug, vg = geostrophic_velocity(torch.from_numpy(eta))
        geo = np.stack([ug.float().numpy(), vg.float().numpy()], axis=1)
        results.setdefault("geostrophic", {})[name] = score(geo, truth, np.ones(eta.shape, bool))

    for spec in args.model:
        label, path = spec.split("=", 1)
        base = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["model_state"]["enc1.0.weight"].shape[0]
        model, stats = load_checkpoint(path, len(INPUT_VARS), base, device)
        results[label] = {"checkpoint": path, "base_channels": int(base)}
        for name, idx in blocks.items():
            ds = StackedNumpyDataset(data_dir=args.data_dir, data_file="global-final-64_stacked_DEEP_GRAD.npy",
                                     channel_order=DEEP_GRAD_CHANNELS, input_vars=INPUT_VARS,
                                     output_vars=OUTPUT_VARS, normalize=False,
                                     subset_indices=np.asarray(idx))
            ds.stats, ds.normalize = stats, True
            pred, truth, mask = predict(model, ds, stats, device, batch_size=256)
            results[label][name] = score(pred, truth, mask)
            if name == "test":
                results[label]["test_by_days_after_training"] = by_lead_time(pred, truth, idx, t_train_end)
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    keys = ["rmse_speed", "rmse_u", "mean_dir_error_deg", "r_zeta", "rmse_zeta_over_f", "r_div"]
    for label, r in results.items():
        if label == "split":
            continue
        for name in ("val", "test"):
            print(f"{label:12s} {name:5s} " + "  ".join(f"{k}={r[name][k]:.4f}" for k in keys))
        if "test_by_days_after_training" in r:
            print(f"{'':12s} test RMSE speed by days after training: {r['test_by_days_after_training']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
