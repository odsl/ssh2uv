#!/usr/bin/env python3
"""Cache predictions of trained checkpoints on one block of the temporal split.

Writes, in the layout read by scripts/figures/*:
    <out>/truth.npy, <out>/eta.npy            (N, 2, 64, 64), (N, 64, 64)
    <out>/val_indices.npy                     global sample indices (t * 6 + p) of the N tiles
    <out>/predictions/<key>.npy               (N, 2, 64, 64) for each checkpoint, plus geostrophic.npy

Usage:
    python scripts/predict.py --block test --out outputs/test_predictions \
        --model c3=runs/temporal/C3_bc128_phys_long/seed0/checkpoint_best.pt \
        --model e0=runs/temporal/E0_baseline/seed0/checkpoint_best.pt
"""

import argparse
import os

import numpy as np
import torch

from ssh2uv.kinematics import N_PATCHES
from ssh2uv.models import UNet
from ssh2uv.pipeline.data import eval_batches, load_domains, load_split_hours
from ssh2uv.pipeline.features import (
    Standardizer, build_inputs, geostrophic_from_raw, n_input_channels, to_velocity,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", default="test", choices=["train", "val", "test"])
    p.add_argument("--split", default="data/temporal_split/global-final-64_master_metadata.json")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model", action="append", default=[], help="key=path/to/checkpoint_best.pt")
    p.add_argument("--out", default="outputs/test_predictions")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, b = load_split_hours(args.split)[args.block]
    dom = load_domains(args.data_dir, (a, b)).to(device)
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)

    truth, eta, geo = [], [], []
    for raw in eval_batches(dom, 512):
        truth.append(raw[:, 4:6].cpu().numpy())
        eta.append(raw[:, 0].cpu().numpy())
        geo.append(geostrophic_from_raw(raw.double()).float().cpu().numpy())
    np.save(os.path.join(args.out, "truth.npy"), np.concatenate(truth))
    np.save(os.path.join(args.out, "eta.npy"), np.concatenate(eta))
    np.save(os.path.join(args.out, "predictions", "geostrophic.npy"), np.concatenate(geo))
    np.save(os.path.join(args.out, "val_indices.npy"), np.arange(a * N_PATCHES, b * N_PATCHES))

    for spec in args.model:
        key, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
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
        np.save(os.path.join(args.out, "predictions", f"{key}.npy"), np.concatenate(preds).astype(np.float32))
        print(f"saved {key} ({cfg['name']}, seed {ck['seed']}, epoch {ck['epoch']})", flush=True)


if __name__ == "__main__":
    main()
