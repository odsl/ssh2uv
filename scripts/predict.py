#!/usr/bin/env python3
"""Cache predictions of trained checkpoints on one block of the temporal split.

Writes, in the layout read by scripts/figures/*:
    <out>/truth.npy, <out>/eta.npy            (N, 2, 64, 64), (N, 64, 64)
    <out>/val_indices.npy                     global sample indices (t * 6 + p) of the N tiles
    <out>/predictions/<key>.npy               (N, 2, 64, 64) for each checkpoint, plus geostrophic.npy
    <out>/metrics.json                        velocity and kinematic metrics of every prediction

Usage:
    python scripts/predict.py --block test --out outputs/test_predictions \
        --model c3=runs/temporal/C3_bc128_phys_long/seed0/checkpoint_best.pt \
        --model e0=runs/temporal/E0_baseline/seed0/checkpoint_best.pt

    # surface velocity (0.5 m) instead of 53 m:
    python scripts/predict.py --data_file global-final-64_stacked_FULL.npy --out outputs/test_predictions_surface \
        --model c3=runs/temporal/S_C3_bc128_phys_long/seed0/checkpoint_best.pt
"""

import argparse
import json
import os

import numpy as np
import torch

from ssh2uv.evaluation import kinematic_metrics, velocity_metrics
from ssh2uv.kinematics import N_PATCHES
from ssh2uv.models import UNet
from ssh2uv.pipeline.data import DATA_FILE, eval_batches, load_domains, load_split_hours
from ssh2uv.pipeline.features import (
    Standardizer, build_inputs, geostrophic_from_raw, n_input_channels, to_velocity,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", default="test", choices=["train", "val", "test"])
    p.add_argument("--split", default="data/temporal_split/global-final-64_master_metadata.json")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--data_file", default=DATA_FILE, help="stacked file; selects the target depth")
    p.add_argument("--model", action="append", default=[], help="key=path/to/checkpoint_best.pt")
    p.add_argument("--out", default="outputs/test_predictions")
    return p.parse_args()


def scores(pred, truth):
    mask = np.ones(truth[:, 0].shape, bool)
    return {**velocity_metrics(pred, truth, mask), **kinematic_metrics(pred, truth, mask)}


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, b = load_split_hours(args.split)[args.block]
    dom = load_domains(args.data_dir, (a, b), args.data_file).to(device)
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)

    truth, eta, geo = [], [], []
    for raw in eval_batches(dom, 512):
        truth.append(raw[:, 4:6].cpu().numpy())
        eta.append(raw[:, 0].cpu().numpy())
        geo.append(geostrophic_from_raw(raw.double()).float().cpu().numpy())
    truth, geo = np.concatenate(truth), np.concatenate(geo)
    np.save(os.path.join(args.out, "truth.npy"), truth)
    np.save(os.path.join(args.out, "eta.npy"), np.concatenate(eta))
    np.save(os.path.join(args.out, "predictions", "geostrophic.npy"), geo)
    np.save(os.path.join(args.out, "val_indices.npy"), np.arange(a * N_PATCHES, b * N_PATCHES))
    metrics = {"block": args.block, "data_file": args.data_file, "geostrophic": scores(geo, truth)}

    for spec in args.model:
        key, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        trained_on = cfg.get("data_file", DATA_FILE)
        if trained_on != args.data_file:
            raise ValueError(f"{key} was trained on {trained_on}, but --data_file is {args.data_file}")
        model = UNet(n_input_channels(cfg["inputs"]), 2, cfg["base_channels"])
        model.load_state_dict(ck["model_state"])
        model.to(device).eval()
        x_std = Standardizer(torch.tensor(ck["x_std"]["mean"]), torch.tensor(ck["x_std"]["std"])).to(device)
        y_std = Standardizer(torch.tensor(ck["y_std"]["mean"]), torch.tensor(ck["y_std"]["std"])).to(device)
        sm = cfg.get("ug_smooth_px", 0.0)
        preds = []
        for raw in eval_batches(dom, 256):
            out = model(x_std.norm(build_inputs(raw, cfg["inputs"], sm)))
            preds.append(to_velocity(y_std.denorm(out.float()), raw, cfg["target"], sm).cpu().numpy())
        pred = np.concatenate(preds).astype(np.float32)
        np.save(os.path.join(args.out, "predictions", f"{key}.npy"), pred)
        metrics[key] = {"config": cfg["name"], "seed": ck["seed"], "epoch": ck["epoch"], **scores(pred, truth)}
        print(f"saved {key} ({cfg['name']}, seed {ck['seed']}, epoch {ck['epoch']}): "
              f"RMSE speed {metrics[key]['rmse_speed']:.4f} m/s", flush=True)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"geostrophic: RMSE speed {metrics['geostrophic']['rmse_speed']:.4f} m/s; wrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
