#!/usr/bin/env python3
"""Evaluate the random-split models on their random-split validation set.

Used for the comparison in Appendix B (Fig. 8a). For every model in
``ssh2uv.evaluation.PAPER_MODELS`` and for an f-plane
geostrophic baseline, this computes velocity RMSEs and the RMSE / Pearson r of
zeta/f and (div u)/f on the validation split, using the 2 km analysis grid and
f0 at the mean latitude of the domain (see ``ssh2uv.kinematics``).

Outputs (in --outdir):
    metrics.json              all metrics
    predictions/<key>.npy     validation predictions, (N, 2, 64, 64) float32 [m/s]
    truth.npy, eta.npy        validation targets (N, 2, 64, 64) and SSH (N, 64, 64)
    val_indices.npy           sample indices of the validation split

scripts/figures/fig_leakage.py reads these files.

Usage:
    python scripts/random_split/evaluate.py --data_dir data --results_dir runs/random_split \
        --outdir outputs/random_split_eval
"""

import argparse
import json
import os

import numpy as np
import torch

from ssh2uv.evaluation import (
    DEEP_CHANNELS,
    PAPER_MODELS,
    kinematic_metrics,
    load_checkpoint,
    predict,
    results_path,
    validation_dataset,
    velocity_metrics,
)
from ssh2uv.kinematics import DX_M, F0, geostrophic_velocity


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=os.environ.get("SSH2UV_DATA", "data"))
    p.add_argument("--results_dir", default=os.environ.get("SSH2UV_RESULTS", "runs/random_split"))
    p.add_argument("--outdir", default="outputs/random_split_eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--models", nargs="*", default=None, help="Subset of model keys to evaluate")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.join(args.outdir, "predictions"), exist_ok=True)
    print(f"device={device}  dx={DX_M} m  f0={F0:.4e} s^-1")

    metrics = {"_constants": {"dx_m": DX_M, "f0": F0,
                              "note": "kinematic RMSEs are of zeta/f and div/f with signed f0"}}
    truth_saved = False

    models = [m for m in PAPER_MODELS if args.models is None or m.key in args.models]
    for m in models:
        ckpt = results_path(args.results_dir, m)
        if not os.path.exists(ckpt):
            print(f"[skip] {m.label}: {ckpt} not found")
            continue
        print(f"\n== {m.label} ==")
        model, stats = load_checkpoint(ckpt, len(m.input_vars), m.base_channels, device)
        val = validation_dataset(args.data_dir, m.data_file, m.channel_order, m.input_vars, stats)
        pred, truth, mask = predict(model, val, stats, device, args.batch_size)
        del model
        torch.cuda.empty_cache()

        entry = {"label": m.label, **velocity_metrics(pred, truth, mask),
                 **kinematic_metrics(pred, truth, mask)}
        metrics[m.key] = entry
        print(json.dumps(entry, indent=2))
        np.save(os.path.join(args.outdir, "predictions", f"{m.key}.npy"), pred.astype(np.float32))
        if not truth_saved:
            np.save(os.path.join(args.outdir, "truth.npy"), truth.astype(np.float32))
            np.save(os.path.join(args.outdir, "mask.npy"), mask)
            np.save(os.path.join(args.outdir, "val_indices.npy"), np.asarray(val.indices))
            truth_saved = True

    # Geostrophic baseline from SSH on the same validation samples
    print("\n== Geostrophic baseline ==")
    val_eta = validation_dataset(args.data_dir, "global-final-64_stacked_DEEP.npy", DEEP_CHANNELS,
                                 ["Eta"])
    idx = np.asarray(val_eta.indices)
    raw = np.load(os.path.join(args.data_dir, "global-final-64_stacked_DEEP.npy"), mmap_mode="r")
    eta = np.stack([raw[i, 0] for i in idx]).astype(np.float32)
    truth = np.stack([raw[i, 4:6] for i in idx]).astype(np.float32)
    mask = np.ones(eta.shape, dtype=bool)
    ug, vg = geostrophic_velocity(torch.from_numpy(eta).double())
    geo = np.stack([ug.float().numpy(), vg.float().numpy()], axis=1)
    entry = {"label": "Geostrophic", **velocity_metrics(geo, truth, mask),
             **kinematic_metrics(geo, truth, mask)}
    metrics["geostrophic"] = entry
    print(json.dumps(entry, indent=2))
    np.save(os.path.join(args.outdir, "eta.npy"), eta)
    np.save(os.path.join(args.outdir, "predictions", "geostrophic.npy"), geo.astype(np.float32))
    if not truth_saved:
        np.save(os.path.join(args.outdir, "truth.npy"), truth)
        np.save(os.path.join(args.outdir, "mask.npy"), mask)
        np.save(os.path.join(args.outdir, "val_indices.npy"), idx)
    else:
        same = np.array_equal(np.load(os.path.join(args.outdir, "val_indices.npy")), idx)
        print(f"validation indices identical across data files: {same}")
        metrics["_constants"]["val_indices_consistent"] = bool(same)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {os.path.join(args.outdir, 'metrics.json')}")


if __name__ == "__main__":
    main()
