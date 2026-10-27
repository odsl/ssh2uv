#!/usr/bin/env python3
"""Train and evaluate one experiment on the temporal split (main pipeline of the paper).

Protocol: train on the training block; early-stop on the validation block (RMSE of
speed); evaluate the best checkpoint once on the held-out test block. Normalisation
statistics come from training samples only.

Usage:
    python scripts/train.py --config configs/temporal/C3_bc128_phys_long.json --seed 0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from ssh2uv.evaluation import kinematic_metrics, velocity_metrics
from ssh2uv.kinematics import F0, divergence, vorticity
from ssh2uv.models import UNet
from ssh2uv.pipeline.data import TrainSampler, eval_batches, fixed_tiles, load_domains, load_split_hours
from ssh2uv.pipeline.features import (
    Standardizer, build_inputs, build_target, n_input_channels, random_rotate, to_velocity,
)

DEFAULTS = {
    "inputs": ["grad", "sst", "wind"], "target": "uv", "base_channels": 64,
    "rotate": False, "crops": False, "loss": "mse",
    "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 128, "max_epochs": 60,
    "patience": 10, "warmup_epochs": 2, "ug_smooth_px": 0.0, "vort_weight": 0.0, "div_weight": 0.0,
    "data_dir": "data", "split": "data/temporal_split/global-final-64_master_metadata.json",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=None)
    p.add_argument("--wandb_project", default="ssh2uv-temporal")
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--steps_per_epoch", type=int, default=None, help="debug: limit steps")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def predict_velocity(model, domains, cfg, x_std, y_std, device, batch_size=256):
    model.eval()
    preds, truths = [], []
    for raw in eval_batches(domains, batch_size):
        raw = raw.to(device)
        sm = cfg["ug_smooth_px"]
        x = x_std.norm(build_inputs(raw, cfg["inputs"], sm))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x)
        uv = to_velocity(y_std.denorm(out.float()), raw, cfg["target"], sm)
        preds.append(uv.cpu().numpy())
        truths.append(raw[:, 4:6].cpu().numpy())
    return np.concatenate(preds), np.concatenate(truths)


def main():
    args = parse_args()
    cfg = {**DEFAULTS, **json.load(open(args.config))}
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    name = cfg["name"]
    outdir = args.outdir or os.path.join("runs", "temporal", name, f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    t0 = time.time()
    hours = load_split_hours(cfg["split"])
    dom = {k: load_domains(cfg["data_dir"], hours[k]).to(device) for k in ("train", "val", "test")}
    print(f"[data] loaded {[tuple(d.shape) for d in dom.values()]} in {time.time() - t0:.0f}s", flush=True)

    # statistics from (augmented) training samples only
    g = torch.Generator().manual_seed(1234)
    n_hours = dom["train"].shape[0]
    t = torch.randint(0, n_hours, (4096,), generator=g)
    p = torch.randint(0, 6, (4096,), generator=g)
    sample = fixed_tiles(dom["train"], t, p)
    if cfg["rotate"]:
        sample = random_rotate(sample, g)
    sm = cfg["ug_smooth_px"]
    x_std = Standardizer.fit(build_inputs(sample, cfg["inputs"], sm)).to(device)
    y_std = Standardizer.fit(build_target(sample, cfg["target"], sm)).to(device)
    # scales for the optional vorticity/divergence penalties (zeta/f, div/f of training samples)
    kin_scale = (float((vorticity(sample[:, 4].double(), sample[:, 5].double()) / F0).std()),
                 float((divergence(sample[:, 4].double(), sample[:, 5].double()) / F0).std()))
    del sample

    # ---------------- model ----------------
    model = UNet(n_input_channels(cfg["inputs"]), 2, cfg["base_channels"]).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sampler = TrainSampler(dom["train"], cfg["batch_size"], cfg["crops"], seed=args.seed)
    steps = args.steps_per_epoch or sampler.steps
    total, warm = cfg["max_epochs"] * steps, cfg["warmup_epochs"] * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    loss_fn = torch.nn.MSELoss() if cfg["loss"] == "mse" else torch.nn.L1Loss()
    aug_g = torch.Generator().manual_seed(args.seed + 1)

    run = None
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, group=name, name=f"{name}-s{args.seed}",
                         config={**cfg, "seed": args.seed, "n_params": n_params}, dir=outdir, reinit=True)
    except Exception as e:  # logging is optional
        print(f"[wandb] disabled: {e}")
    print(f"[model] {n_params:,} parameters | config {json.dumps(cfg)}", flush=True)

    # ---------------- train ----------------
    best, best_epoch, bad, history = float("inf"), 0, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t_ep, run_loss = time.time(), 0.0
        for i, raw in enumerate(sampler):
            if i >= steps:
                break
            if cfg["rotate"]:
                raw = random_rotate(raw, aug_g)
            x = x_std.norm(build_inputs(raw, cfg["inputs"], sm))
            y = y_std.norm(build_target(raw, cfg["target"], sm))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
            loss = loss_fn(out.float(), y)
            if cfg["vort_weight"] or cfg["div_weight"]:
                uv_hat = to_velocity(y_std.denorm(out.float()), raw, cfg["target"], sm)
                uh, vh, ut, vt = uv_hat[:, 0], uv_hat[:, 1], raw[:, 4], raw[:, 5]
                z = lambda u, v: vorticity(u, v) / F0 / kin_scale[0]
                d = lambda u, v: divergence(u, v) / F0 / kin_scale[1]
                loss = loss + cfg["vort_weight"] * loss_fn(z(uh, vh), z(ut, vt)) \
                    + cfg["div_weight"] * loss_fn(d(uh, vh), d(ut, vt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
        pred, truth = predict_velocity(model, dom["val"], cfg, x_std, y_std, device)
        vm = velocity_metrics(pred, truth, np.ones(truth[:, 0].shape, bool))
        rec = {"epoch": epoch, "train_loss": run_loss / steps, "val_rmse_speed": vm["rmse_speed"],
               "val_rmse_u": vm["rmse_u"], "val_dir_err": vm["mean_dir_error_deg"],
               "lr": sched.get_last_lr()[0], "time_s": time.time() - t_ep}
        history.append(rec)
        print(" | ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        if run:
            run.log(rec, step=epoch)
        if vm["rmse_speed"] < best:
            best, best_epoch, bad = vm["rmse_speed"], epoch, 0
            torch.save({"model_state": model.state_dict(), "x_std": x_std.state(), "y_std": y_std.state(),
                        "config": cfg, "seed": args.seed, "epoch": epoch}, os.path.join(outdir, "checkpoint_best.pt"))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[early stop] no val improvement for {bad} epochs", flush=True)
                break

    # ---------------- test (once, best checkpoint) ----------------
    ck = torch.load(os.path.join(outdir, "checkpoint_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    results = {"name": name, "seed": args.seed, "n_params": n_params, "best_epoch": best_epoch,
               "epochs_run": len(history), "config": cfg}
    for split in ("val", "test"):
        pred, truth = predict_velocity(model, dom[split], cfg, x_std, y_std, device)
        mask = np.ones(truth[:, 0].shape, bool)
        results[split] = {**velocity_metrics(pred, truth, mask), **kinematic_metrics(pred, truth, mask)}
    results["history"] = history
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    summary = {f"test/{k}": v for k, v in results["test"].items()}
    summary.update({f"final_val/{k}": v for k, v in results["val"].items()})
    summary["best_epoch"] = best_epoch
    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in summary.items()}), flush=True)
    if run:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
