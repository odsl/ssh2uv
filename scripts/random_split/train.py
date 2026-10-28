#!/usr/bin/env python3
"""Training script for the random-split comparison.

Used in the paper only for Appendix B: the random-split models
and the same configuration retrained on the temporal split. Without a split file in the
data directory, the loader falls back to a random 80/20 split of patches (seed 42).

Usage:
    python scripts/random_split/train.py --config configs/random_split/grad_bc192/grad_ab_sst_wind_bc192.json

    # Or with CLI overrides:
    python scripts/random_split/train.py \
        --config configs/random_split/grad_bc192/grad_ab_bc192.json \
        --outdir runs/random_split/my_run --epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from ssh2uv.data.dataset import (
    MMapDataset,
    StackedNumpyDataset,
    create_train_val_datasets,
    create_stacked_train_val_datasets,
    create_datasets_auto,
    detect_data_format,
    load_metadata,
    load_norm_stats,
    NormStats,
    NormStatsTorch,
    CHANNEL_DISPLAY_NAMES,
    CHANNEL_UNITS,
)
from ssh2uv.config import load_config, ExperimentConfig
from ssh2uv.losses import VelocityPhysicsLoss, direction_loss_cosine
from ssh2uv.metrics import evaluate_on_loader
from ssh2uv.models import UNet

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed. Install with: pip install wandb")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not installed. Visualizations will be skipped.")


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    """Save dictionary to JSON with numpy type conversion."""
    def convert_numpy(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, dict):
            return {k: convert_numpy(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert_numpy(i) for i in o]
        return o
    
    with open(path, "w") as f:
        json.dump(convert_numpy(obj), f, indent=2)


def compute_dataset_metadata_fingerprint(metadata: Dict) -> str:
    """Create a deterministic fingerprint for dataset metadata provenance."""
    metadata_json = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()


def define_wandb_metric_semantics() -> None:
    """Define epoch-based metric semantics for cleaner W&B dashboards."""
    if not WANDB_AVAILABLE:
        return

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("best_epoch", step_metric="epoch", summary="last")
    wandb.define_metric("best_val_loss", step_metric="epoch", summary="min")
    wandb.define_metric("timing/*", step_metric="epoch")
    wandb.define_metric("throughput/*", step_metric="epoch")
    wandb.define_metric("efficiency/*", step_metric="epoch")
    wandb.define_metric("gpu/*", step_metric="epoch")
    wandb.define_metric("final/*", step_metric="epoch")


# =============================================================================
# Visualization
# =============================================================================

def create_prediction_visualization(
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    input_vars: List[str],
    output_vars: List[str],
    num_samples: int = 4,
    sample_indices: Optional[List[int]] = None,
) -> Optional[plt.Figure]:
    """Create visualization comparing predictions vs ground truth in physical units.
    
    Supports any number of input and output channels. The visualization layout is:
    - Row per sample
    - Columns: [input_0, input_1, ..., output_0_true, output_0_pred, output_1_true, output_1_pred, ...]
    
    For exactly 2 outputs (U/V), uses the legacy 5-column layout for compatibility.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    num_samples = min(num_samples, x.shape[0])
    C_in = len(input_vars)
    C_out = len(output_vars)
    
    # Comprehensive unit labels including new format
    unit_labels = {
        'SSH': 'm', 'TAUX': 'N/m²', 'TAUY': 'N/m²', 'SST': '°C',
        'u_sfc': 'm/s', 'v_sfc': 'm/s', 'u_k1': 'm/s', 'v_k1': 'm/s',
        'uE_sfc': 'm/s', 'vN_sfc': 'm/s', 'uE_k1': 'm/s', 'vN_k1': 'm/s',
        # New stacked format variables
        'Eta': 'm', 'windx': 'N/m²', 'windy': 'N/m²', 'theta': '°C',
        'u0': 'm/s', 'v0': 'm/s', 'u1': 'm/s', 'v1': 'm/s',
    }
    # Merge with imported channel units
    unit_labels.update(CHANNEL_UNITS)
    
    # Display names for cleaner labels
    display_names = {
        'Eta': 'SSH (η)', 'windx': 'Wind τx', 'windy': 'Wind τy', 
        'theta': 'SST (θ)', 'u0': 'U surface', 'v0': 'V surface',
        'u1': 'U deep', 'v1': 'V deep',
    }
    display_names.update(CHANNEL_DISPLAY_NAMES)
    
    # Determine layout: 1 input col + 2 cols per output (true, pred)
    n_cols = 1 + 2 * C_out
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(4.5 * n_cols, 3.5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        sample_mask = mask[i, 0].cpu().numpy() <= 0.5
        col = 0

        # Input (show first input channel)
        x_plot = np.ma.masked_where(sample_mask, x[i, 0].cpu().numpy())
        im0 = axes[i, col].imshow(x_plot, cmap='viridis')
        input_name = display_names.get(input_vars[0], input_vars[0])
        input_unit = unit_labels.get(input_vars[0], '')
        title = f"{input_name}"
        if C_in > 1:
            title += f" (+{C_in-1} more)"
        if input_unit:
            title += f" [{input_unit}]"
        axes[i, col].set_title(title)
        axes[i, col].axis('off')
        plt.colorbar(im0, ax=axes[i, col], fraction=0.046, pad=0.04)
        col += 1
        
        # Output channels (true and pred for each)
        for c in range(C_out):
            out_true = np.ma.masked_where(sample_mask, y_true[i, c].cpu().numpy())
            out_pred = np.ma.masked_where(sample_mask, y_pred[i, c].cpu().numpy())
            
            # Determine colormap: use diverging for velocity-like vars
            var_name = output_vars[c]
            is_velocity = any(v in var_name.lower() for v in ['u', 'v', 'vel'])
            cmap = 'RdBu_r' if is_velocity else 'viridis'
            
            if is_velocity:
                vmax = max(abs(out_true).max(), abs(out_pred).max(), 1e-8)
                vmin = -vmax
            else:
                vmin = min(out_true.min(), out_pred.min())
                vmax = max(out_true.max(), out_pred.max())
            
            out_display = display_names.get(var_name, var_name)
            out_unit = unit_labels.get(var_name, '')
            unit_str = f" [{out_unit}]" if out_unit else ""
            
            # True
            im_true = axes[i, col].imshow(out_true, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i, col].set_title(f"{out_display} true{unit_str}")
            axes[i, col].axis('off')
            plt.colorbar(im_true, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1
            
            # Pred
            im_pred = axes[i, col].imshow(out_pred, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i, col].set_title(f"{out_display} pred{unit_str}")
            axes[i, col].axis('off')
            plt.colorbar(im_pred, ax=axes[i, col], fraction=0.046, pad=0.04)
            col += 1

        sample_label = f"idx {sample_indices[i]}" if sample_indices is not None and i < len(sample_indices) else f"sample {i}"
        axes[i, 0].set_ylabel(sample_label, rotation=90, fontsize=10)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Training Loop
# =============================================================================

def train(
    mmap_dir: str,
    outdir: str,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    dir_weight: float = 0.05,
    speed_eps: float = 0.01,
    val_fraction: float = 0.1,
    temporal_split: bool = True,  # Default: temporal split for ocean data
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    base_channels: int = 32,
    amp: bool = True,
    grad_clip_norm: float = 1.0,
    early_stop_patience: int = 10,
    warmup_epochs: int = 3,
    seed: int = 42,
    exclude_zeros: bool = False,  # DEPRECATED - no longer used
    # Physics-informed loss settings
    loss_type: str = 'mse',          # 'mse' (L2) or 'mae' (L1)
    vort_weight: float = 0.0,        # a1: weight on vorticity residual
    div_weight: float = 0.0,         # a2: weight on divergence residual
    # Resume from checkpoint
    resume_from: Optional[str] = None,
    # Wandb settings
    wandb_enabled: bool = True,
    wandb_project: str = "ssh2uv",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    wandb_offline: bool = False,
    wandb_log_model: bool = False,
    # Config reference
    config: Optional[ExperimentConfig] = None,
    # Data format options
    input_vars_override: Optional[List[str]] = None,
    output_vars_override: Optional[List[str]] = None,
    data_format: str = 'auto',  # 'auto', 'mmap', or 'stacked'
    data_file: Optional[str] = None,
    channel_order: Optional[List[str]] = None,
) -> Dict:
    """Main training function.
    
    Supports both legacy mmap format and new stacked numpy format.
    The data format is auto-detected by default.
    
    Returns:
        Dictionary with training history and best metrics
    """
    print("=" * 70)
    print("SSH2UV Training - Memory-Mapped Data")
    print("=" * 70)
    
    # Setup
    total_run_start = time.time()
    ensure_dir(outdir)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Detect data format
    detected_format = detect_data_format(mmap_dir)
    if data_format == 'auto':
        data_format = detected_format
    print(f"Data format: {data_format}")
    
    # Create datasets based on format
    print("\nCreating datasets...")
    
    if data_format == 'stacked':
        # Use new stacked format dataset
        train_dataset, val_dataset = create_stacked_train_val_datasets(
            data_dir=mmap_dir,
            data_file=data_file,
            channel_order=channel_order,
            input_vars=input_vars_override,
            output_vars=output_vars_override,
            val_fraction=val_fraction,
            normalize=True,
            seed=seed,
        )
        input_vars = train_dataset.input_vars
        output_vars = train_dataset.output_vars
        C_in = len(input_vars)
        C_out = len(output_vars)
        total_samples = train_dataset.total_samples
        data_is_prenormalized = train_dataset.data_is_prenormalized
        # Create metadata dict for stacked format (for wandb logging)
        metadata = {
            'format': 'stacked',
            'data_file': data_file,
            'channel_order': channel_order,
            'input_vars': input_vars,
            'output_vars': output_vars,
            'total_samples': total_samples,
            'normalized': data_is_prenormalized,
        }
    else:
        # Use legacy mmap format
        metadata = load_metadata(mmap_dir)
        input_vars = input_vars_override or metadata['input_vars']
        output_vars = output_vars_override or metadata['output_vars']
        C_in = len(input_vars)
        C_out = len(output_vars)
        total_samples = metadata.get('total_samples', 0)
        data_is_prenormalized = metadata.get('normalized', False)
        
        train_dataset, val_dataset = create_train_val_datasets(
            mmap_dir=mmap_dir,
            val_fraction=val_fraction,
            normalize=True,
            exclude_zeros=exclude_zeros,
            seed=seed,
            temporal_split=temporal_split,
        )
    
    print(f"Input vars:  {input_vars}")
    print(f"Output vars: {output_vars}")
    print(f"Data dir:    {mmap_dir}")
    print(f"Data pre-normalized: {data_is_prenormalized}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")
    
    # Explicit normalization state check - loud and clear
    print(f"\n{'='*50}")
    print(f"NORMALIZATION STATE CHECK:")
    print(f"  Data pre-normalized during preprocessing: {train_dataset.data_is_prenormalized}")
    print(f"  Runtime normalization will be applied:    {train_dataset.normalize}")
    if train_dataset.data_is_prenormalized and not train_dataset.normalize:
        print(f"  [OK] Pre-normalized data detected, runtime normalization DISABLED")
    elif not train_dataset.data_is_prenormalized and train_dataset.normalize:
        print(f"  [OK] Raw data detected, runtime normalization ENABLED")
    else:
        print(f"  [OK] Normalization disabled by user request")
    print(f"{'='*50}\n")
    
    # Get normalization stats
    stats = train_dataset.get_stats()
    if stats is None:
        raise ValueError("Normalization stats not found in mmap directory")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    print("DataLoader config:")
    print(f"  Workers: {num_workers}, Prefetch: {prefetch_factor}, Pin memory: {pin_memory and device.type == 'cuda'}")
    print(f"  Total prefetched batches: {num_workers * prefetch_factor if num_workers > 0 else 0}")
    if device.type == "cuda" and not pin_memory:
        print("  WARNING: pin_memory disabled - CPU->GPU transfer may be slower")
    
    # Create model
    model = UNet(in_channels=C_in, out_channels=C_out, base=base_channels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Physics-informed loss
    physics_loss_fn = VelocityPhysicsLoss(
        loss_type=loss_type,
        vort_weight=vort_weight,
        div_weight=div_weight,
    )
    print(f"Loss: {loss_type.upper()} + vort_weight={vort_weight} + div_weight={div_weight}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # AMP scaler
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Provenance for wandb config/dashboard inspection
    pin_memory_enabled = pin_memory and device.type == "cuda"
    persistent_workers = num_workers > 0
    split_strategy = "temporal" if temporal_split else "random"
    dataset_metadata_fingerprint = compute_dataset_metadata_fingerprint(metadata)
    effective_batch_size = batch_size
    effective_lr = lr
    metadata_path = os.path.abspath(os.path.join(mmap_dir, "metadata.json"))
    wandb_config = {
        "mmap_dir": mmap_dir,
        "mmap_dir_abs": os.path.abspath(mmap_dir),
        "outdir": outdir,
        "outdir_abs": os.path.abspath(outdir),
        "metadata_path": metadata_path,
        "dataset_metadata_fingerprint": dataset_metadata_fingerprint,
        "dataset_metadata_fingerprint_algo": "sha256",
        "dataset_format": metadata.get("format"),
        "dataset_zarr_source": metadata.get("zarr_source"),
        "dataset_total_samples": total_samples,
        "dataset_num_timesteps": metadata.get("num_timesteps"),
        "dataset_temporal_data": metadata.get("temporal_data"),
        "dataset_temporal_split_recommended": metadata.get("temporal_split_recommended"),
        "dataset_patch_size": metadata.get("patch_size"),
        "dataset_input_channels": metadata.get("input_channels"),
        "dataset_output_channels": metadata.get("output_channels"),
        "dataset_input_shape": metadata.get("input_shape"),
        "dataset_output_shape": metadata.get("output_shape"),
        "data_prenormalized": train_dataset.data_is_prenormalized,
        "runtime_normalization_enabled": train_dataset.normalize,
        "input_vars": input_vars,
        "output_vars": output_vars,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_steps_per_epoch": len(train_loader),
        "val_steps_per_epoch": len(val_loader),
        "split_strategy": split_strategy,
        "temporal_split": temporal_split,
        "val_fraction": val_fraction,
        "split_seed": seed,
        "seed": seed,
        "seed_strategy": "fixed",
        "epochs": epochs,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "lr": lr,
        "effective_lr": effective_lr,
        "weight_decay": weight_decay,
        "dir_weight": dir_weight,
        "loss_type": loss_type,
        "vort_weight": vort_weight,
        "div_weight": div_weight,
        "speed_eps": speed_eps,
        "amp_requested": amp,
        "amp_enabled": use_amp,
        "grad_clip_norm": grad_clip_norm,
        "grad_clip_enabled": grad_clip_norm > 0,
        "warmup_epochs": warmup_epochs,
        "warmup_enabled": warmup_epochs > 0,
        "early_stop_patience": early_stop_patience,
        "early_stopping_enabled": early_stop_patience > 0,
        "world_size": 1,
        "model_name": "UNet",
        "base_channels": base_channels,
        "model_params": total_params,
        "optimizer_name": optimizer.__class__.__name__,
        "optimizer_betas": list(optimizer.defaults.get("betas", ())),
        "optimizer_eps": optimizer.defaults.get("eps"),
        "optimizer_weight_decay": optimizer.defaults.get("weight_decay"),
        "scheduler_name": scheduler.__class__.__name__,
        "scheduler_mode": getattr(scheduler, "mode", None),
        "scheduler_factor": getattr(scheduler, "factor", None),
        "scheduler_patience": getattr(scheduler, "patience", None),
        "scheduler_monitor": "val/loss",
        "dataloader_num_workers": num_workers,
        "dataloader_prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "dataloader_total_prefetched_batches": num_workers * prefetch_factor if num_workers > 0 else 0,
        "dataloader_pin_memory_requested": pin_memory,
        "dataloader_pin_memory_enabled": pin_memory_enabled,
        "dataloader_persistent_workers": persistent_workers,
        "dataloader_train_shuffle": True,
        "dataloader_val_shuffle": False,
        "dataloader_train_drop_last": True,
        "dataloader_val_drop_last": False,
    }
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_val = float("inf")
    best_epoch = -1
    history = {
        "train_loss": [], "train_mse": [], "train_dir": [],
            "train_vort": [], "train_div": [],
            "val_loss": [], "val_mse": [], "val_dir": [],
            "val_vort": [], "val_div": [],
        "wandb_run_id": None,
    }
    
    resume_ckpt_exists = resume_from is not None and os.path.exists(resume_from)

    if resume_ckpt_exists:
        print(f"\n[RESUME] Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        loaded_history = ckpt.get('history', {})
        
        # Validate checkpoint matches current variable configuration
        ckpt_input_vars = loaded_history.get('input_vars', [])
        ckpt_output_vars = loaded_history.get('output_vars', [])
        
        if ckpt_input_vars and ckpt_input_vars != input_vars:
            raise ValueError(
                f"[RESUME ERROR] Checkpoint input_vars mismatch!\n"
                f"  Checkpoint: {ckpt_input_vars}\n"
                f"  Current:    {input_vars}\n"
                f"Cannot resume training with different input variables."
            )
        if ckpt_output_vars and ckpt_output_vars != output_vars:
            raise ValueError(
                f"[RESUME ERROR] Checkpoint output_vars mismatch!\n"
                f"  Checkpoint: {ckpt_output_vars}\n"
                f"  Current:    {output_vars}\n"
                f"Cannot resume training with different output variables."
            )
        
        # Safe to use loaded history
        history = loaded_history
        # Ensure current vars are recorded (for old checkpoints without vars)
        history['input_vars'] = input_vars
        history['output_vars'] = output_vars
        
        # Restore scheduler state if available
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        
        # Restore scaler state if available (for AMP)
        if 'scaler_state' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler_state'])
        
        # Find best val from history
        if history.get('val_loss'):
            best_val = min(history['val_loss'])
            best_epoch = history['val_loss'].index(best_val) + 1
        
        print(f"[RESUME] Resuming from epoch {start_epoch}, best_val={best_val:.6f}")
    
    # Normalization tensors
    y_mean_t = torch.from_numpy(stats.y_mean.reshape(1, C_out, 1, 1)).to(device, dtype=torch.float32)
    y_std_t = torch.from_numpy(stats.y_std.reshape(1, C_out, 1, 1)).to(device, dtype=torch.float32)
    x_mean_t = torch.from_numpy(stats.x_mean.reshape(1, C_in, 1, 1)).to(device, dtype=torch.float32)
    x_std_t = torch.from_numpy(stats.x_std.reshape(1, C_in, 1, 1)).to(device, dtype=torch.float32)
    
    # Initialize wandb
    use_wandb = wandb_enabled and WANDB_AVAILABLE
    if wandb_enabled and not WANDB_AVAILABLE:
        print("[WARN] wandb requested but not available")
    
    if use_wandb:
        if wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        wandb_resume_id = history.get("wandb_run_id") if resume_ckpt_exists else None
        if resume_ckpt_exists and not wandb_resume_id:
            print("[WARN] Resume checkpoint has no wandb_run_id; starting a new wandb run")

        if not wandb_run_name and not wandb_resume_id:
            import datetime
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            wandb_run_name = f"{os.path.basename(outdir)}_{timestamp}"

        wandb_init_kwargs = {
            "project": wandb_project,
            "entity": wandb_entity,
            "tags": wandb_tags or [],
            "config": wandb_config,
            "dir": outdir,
        }
        if wandb_run_name:
            wandb_init_kwargs["name"] = wandb_run_name
        if wandb_resume_id:
            wandb_init_kwargs["id"] = wandb_resume_id
            wandb_init_kwargs["resume"] = "allow"

        wandb.init(**wandb_init_kwargs)
        wandb.config.update(wandb_config, allow_val_change=True)
        define_wandb_metric_semantics()
        history["wandb_run_id"] = wandb.run.id
        wandb.watch(model, log="gradients", log_freq=100)
        print(f"[INFO] Wandb run: {wandb.run.name}")
    
    patience_left = early_stop_patience
    
    # Training loop
    print("\n" + "=" * 70)
    print(f"Starting training from epoch {start_epoch}...")
    print("=" * 70)
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        
        running_loss, running_mse, running_dir = 0.0, 0.0, 0.0
        running_vort, running_div = 0.0, 0.0
        seen = 0
        data_time = 0.0
        compute_time = 0.0
        batch_start = time.time()
        
        # Learning rate warmup - only apply on fresh starts, not on resume
        # Otherwise warmup could re-inflate an already-decayed LR from scheduler
        if warmup_epochs > 0 and epoch <= warmup_epochs and start_epoch == 1:
            warmup_lr = lr * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for xb, yb, maskb in pbar:
            data_time += time.time() - batch_start
            compute_start = time.time()

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            maskb = maskb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(xb)
                total_loss, base_loss, vort_l, div_l = physics_loss_fn(
                    pred, yb, maskb, y_mean_t, y_std_t)
                dloss = direction_loss_cosine(pred, yb, maskb, y_mean_t, y_std_t, speed_eps)
                loss = total_loss + dir_weight * dloss

            scaler.scale(loss).backward()

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_mse += base_loss.item() * bs
            running_dir += dloss.item() * bs
            running_vort += vort_l.item() * bs
            running_div += div_l.item() * bs
            seen += bs
            compute_time += time.time() - compute_start
            batch_start = time.time()
            
            pbar.set_postfix({"loss": running_loss / max(seen, 1)})

        train_loss = running_loss / max(seen, 1)
        train_mse = running_mse / max(seen, 1)
        train_dir = running_dir / max(seen, 1)
        train_vort = running_vort / max(seen, 1)
        train_div = running_div / max(seen, 1)
        train_time = time.time() - epoch_start

        # Validation
        val_start = time.time()
        model.eval()
        val_running_loss, val_running_mse, val_running_dir = 0.0, 0.0, 0.0
        val_running_vort, val_running_div = 0.0, 0.0
        val_seen = 0
        first_batch = None
        first_batch_indices = None

        with torch.no_grad():
            for batch_idx, (xb, yb, maskb) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                maskb = maskb.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(xb)
                    total_loss, base_loss, vort_l, div_l = physics_loss_fn(
                        pred, yb, maskb, y_mean_t, y_std_t)
                    dloss = direction_loss_cosine(pred, yb, maskb, y_mean_t, y_std_t, speed_eps)
                    loss = total_loss + dir_weight * dloss

                if batch_idx == 0 and epoch % 5 == 0:
                    first_batch = (xb[:4].clone(), yb[:4].clone(), pred[:4].clone(), maskb[:4].clone())
                    current_bs = xb.size(0)
                    batch_start_idx = batch_idx * batch_size
                    first_batch_indices = val_dataset.indices[batch_start_idx:batch_start_idx + min(4, current_bs)].tolist()

                bs = xb.size(0)
                val_running_loss += loss.item() * bs
                val_running_mse += base_loss.item() * bs
                val_running_dir += dloss.item() * bs
                val_running_vort += vort_l.item() * bs
                val_running_div += div_l.item() * bs
                val_seen += bs

        val_loss = val_running_loss / max(val_seen, 1)
        val_mse = val_running_mse / max(val_seen, 1)
        val_dir = val_running_dir / max(val_seen, 1)
        val_vort = val_running_vort / max(val_seen, 1)
        val_div = val_running_div / max(val_seen, 1)
        val_time = time.time() - val_start
        
        scheduler.step(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_dir"].append(train_dir)
        history.setdefault("train_vort", []).append(train_vort)
        history.setdefault("train_div", []).append(train_div)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_dir"].append(val_dir)
        history.setdefault("val_vort", []).append(val_vort)
        history.setdefault("val_div", []).append(val_div)

        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "history": history,
            "norm_stats": {
                "x_mean": stats.x_mean,
                "x_std": stats.x_std,
                "y_mean": stats.y_mean,
                "y_std": stats.y_std,
            },
        }
        torch.save(ckpt, os.path.join(outdir, "checkpoint_last.pt"))

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            torch.save(ckpt, os.path.join(outdir, "checkpoint_best.pt"))
            if early_stop_patience > 0:
                patience_left = early_stop_patience
        else:
            if early_stop_patience > 0:
                patience_left -= 1

        # Log
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        
        phys_str = ""
        if vort_weight > 0 or div_weight > 0:
            phys_str = f" vort {val_vort:.5f} div {val_div:.5f} |"
        print(
            f"Epoch {epoch:03d} | lr {lr_now:.2e} | "
            f"train {train_loss:.6f} (base {train_mse:.6f}, dir {train_dir:.6f}) | "
            f"val {val_loss:.6f} (base {val_mse:.6f}, dir {val_dir:.6f}){phys_str} | "
            f"best {best_val:.6f} @ {best_epoch} | "
            f"time: {epoch_time:.1f}s"
        )

        if epoch == 1:
            data_pct = 100 * data_time / train_time if train_time > 0 else 0.0
            compute_pct = 100 * compute_time / train_time if train_time > 0 else 0.0
            print("\n[DIAGNOSTIC] First epoch timing breakdown:")
            print(f"  Data loading: {data_time:.1f}s ({data_pct:.1f}%)")
            print(f"  Compute:      {compute_time:.1f}s ({compute_pct:.1f}%)")
            print(f"  Throughput:   {seen / train_time:.1f} samples/sec")
            if device.type == "cuda":
                total_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
                max_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"  GPU memory:   {max_mem_gb:.2f} GB / {total_mem_gb:.2f} GB")
                print(f"  GPU util:     {100 * max_mem_gb / total_mem_gb:.1f}%")
            if data_time > compute_time:
                print("  BOTTLENECK: data loading is slower than model compute")
                print("    Suggestions:")
                print("      - Increase --num_workers")
                print("      - Increase --prefetch_factor")
                print("      - Enable --pin_memory")
                print("      - Increase --batch_size if GPU memory allows")
            print("")

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/base_loss": train_mse,
                "train/dir_loss": train_dir,
                "train/vort_loss": train_vort,
                "train/div_loss": train_div,
                "val/loss": val_loss,
                "val/base_loss": val_mse,
                "val/dir_loss": val_dir,
                "val/vort_loss": val_vort,
                "val/div_loss": val_div,
                "learning_rate": lr_now,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "timing/epoch_time": epoch_time,
                "timing/train_time": train_time,
                "timing/val_time": val_time,
                "throughput/samples_per_sec_train": seen / train_time if train_time > 0 else 0.0,
                "throughput/samples_per_sec_val": val_seen / val_time if val_time > 0 else 0.0,
                "efficiency/data_loading_time": data_time,
                "efficiency/compute_time": compute_time,
                "efficiency/data_to_compute_ratio": data_time / compute_time if compute_time > 0 else 0.0,
                "efficiency/data_loading_pct": 100 * data_time / train_time if train_time > 0 else 0.0,
                "efficiency/compute_pct": 100 * compute_time / train_time if train_time > 0 else 0.0,
            }
            if device.type == "cuda":
                total_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
                mem_alloc_gb = torch.cuda.memory_allocated(device) / 1e9
                log_dict["gpu/memory_allocated_gb"] = mem_alloc_gb
                log_dict["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
                log_dict["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                log_dict["gpu/memory_utilization_pct"] = 100 * mem_alloc_gb / total_mem_gb if total_mem_gb > 0 else 0.0
            
            if first_batch is not None and MATPLOTLIB_AVAILABLE:
                try:
                    xb_vis, yb_vis, pred_vis, mask_vis = first_batch
                    x_phys = xb_vis * x_std_t + x_mean_t
                    y_phys = yb_vis * y_std_t + y_mean_t
                    pred_phys = pred_vis * y_std_t + y_mean_t
                    
                    fig = create_prediction_visualization(
                        x_phys, y_phys, pred_phys, mask_vis,
                        input_vars, output_vars, num_samples=4,
                        sample_indices=first_batch_indices,
                    )
                    if fig is not None:
                        log_dict["predictions"] = wandb.Image(fig)
                        if first_batch_indices is not None:
                            log_dict["predictions/sample_indices"] = first_batch_indices
                        plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Visualization failed: {e}")
            
            wandb.log(log_dict)
            wandb.summary["best_epoch"] = best_epoch
            wandb.summary["best_val_loss"] = best_val
            wandb.summary["last_completed_epoch"] = epoch

        
        print(f"  {data_time:.1f}s loading vs {compute_time:.1f}s compute ")

        save_json(history, os.path.join(outdir, "history.json"))

        # Early stopping
        if early_stop_patience > 0 and patience_left <= 0:
            print(f"Early stopping at epoch {epoch}. Best: {best_epoch}")
            break

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    best_path = os.path.join(outdir, "checkpoint_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    
    final_batch_vis = None
    final_batch_indices = None
    if MATPLOTLIB_AVAILABLE:
        model.eval()
        with torch.no_grad():
            for batch_idx, (xb, yb, maskb) in enumerate(val_loader):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                maskb = maskb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    predb = model(xb)
                current_bs = xb.shape[0]
                final_batch_vis = (
                    (xb[:4] * x_std_t + x_mean_t).detach().cpu(),
                    (yb[:4] * y_std_t + y_mean_t).detach().cpu(),
                    (predb[:4] * y_std_t + y_mean_t).detach().cpu(),
                    maskb[:4].detach().cpu(),
                )
                batch_start_idx = batch_idx * val_loader.batch_size
                final_batch_indices = val_dataset.indices[batch_start_idx:batch_start_idx + min(4, current_bs)].tolist()
                break

    final_metrics = evaluate_on_loader(model, val_loader, device, y_mean_t, y_std_t, speed_eps)
    total_training_time = time.time() - total_run_start
    
    print(f"RMSE U: {final_metrics['rmse_u']:.6f} m/s")
    print(f"RMSE V: {final_metrics['rmse_v']:.6f} m/s")
    print(f"RMSE Speed: {final_metrics['rmse_speed']:.6f} m/s")
    print(f"Mean Direction Error: {final_metrics['mean_dir_error_deg']:.2f}°")
    print(f"Total training time: {total_training_time / 60.0:.2f} min ({total_training_time:.1f} s)")
    
    # Save final metrics
    history["final_metrics"] = final_metrics
    history["total_training_time_sec"] = total_training_time
    save_json(history, os.path.join(outdir, "history.json"))
    
    if use_wandb:
        final_eval_epoch = best_epoch if best_epoch > 0 else len(history.get("val_loss", []))
        log_dict = {f"final/{k}": v for k, v in final_metrics.items()}
        log_dict["epoch"] = final_eval_epoch
        log_dict["best_epoch"] = best_epoch
        log_dict["best_val_loss"] = best_val
        log_dict["timing/total_training_time_sec"] = total_training_time
        log_dict["timing/total_training_time_min"] = total_training_time / 60.0
        if final_batch_vis is not None and MATPLOTLIB_AVAILABLE:
            try:
                x_vis, y_vis, pred_vis, mask_vis = final_batch_vis
                fig = create_prediction_visualization(
                    x_vis, y_vis, pred_vis, mask_vis,
                    input_vars, output_vars, num_samples=min(4, x_vis.shape[0]),
                    sample_indices=final_batch_indices,
                )
                if fig is not None:
                    final_pred_path = os.path.join(outdir, "final_predictions.png")
                    fig.savefig(final_pred_path, dpi=150, bbox_inches='tight')
                    log_dict["final/predictions"] = wandb.Image(fig)
                    if final_batch_indices is not None:
                        log_dict["final/predictions_sample_indices"] = final_batch_indices
                    plt.close(fig)
            except Exception as e:
                print(f"[WARN] Final visualization failed: {e}")
        wandb.log(log_dict)
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_val_loss"] = best_val
        wandb.summary["final_checkpoint_epoch"] = final_eval_epoch
        wandb.summary["final_rmse_u"] = final_metrics["rmse_u"]
        wandb.summary["final_rmse_v"] = final_metrics["rmse_v"]
        wandb.summary["final_rmse_speed"] = final_metrics["rmse_speed"]
        wandb.summary["final_mean_dir_error_deg"] = final_metrics["mean_dir_error_deg"]
        wandb.summary["total_training_time_sec"] = total_training_time
        
        if wandb_log_model:
            artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)
        
        wandb.finish()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best model saved to: {best_path}")
    print(f"Total wall time: {total_training_time / 60.0:.2f} min")
    print("=" * 70)
    
    return history


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SSH2UV model using memory-mapped data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    
    # Data
    parser.add_argument("--mmap_dir", type=str, default=None,
                        help="Directory with mmap data files")
    parser.add_argument("--outdir", type=str, default="runs/random_split/run",
                        help="Output directory")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--temporal_split", action=argparse.BooleanOptionalAction, default=True,
                        help="Use temporal split (last N%% timesteps for val). "
                             "IMPORTANT: Default True for ocean data to prevent temporal leakage. "
                             "Use --no-temporal_split for random split (NOT recommended for temporal data).")
    parser.add_argument("--exclude_zeros", action=argparse.BooleanOptionalAction, default=False,
                        help="DEPRECATED - no longer used. Was buggy: incorrectly masked "
                             "valid ocean pixels with zero values (calm water conditions).")
    
    # Data format options (NEW)
    parser.add_argument("--data_format", type=str, default="auto",
                        choices=["auto", "mmap", "stacked"],
                        help="Data format: 'auto' (detect), 'mmap' (legacy), 'stacked' (new)")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Stacked data file name (e.g., 'global-final-64_stacked_DEEP.npy')")
    parser.add_argument("--input_vars", type=str, nargs='+', default=None,
                        help="Input variable names (e.g., 'Eta windx windy')")
    parser.add_argument("--output_vars", type=str, nargs='+', default=None,
                        help="Output variable names (e.g., 'u1 v1')")
    parser.add_argument("--channel_order", type=str, nargs='+', default=None,
                        help="Channel order in stacked file (e.g., 'Eta windx windy theta u1 v1')")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dir_weight", type=float, default=0.05)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mae"],
                        help="Base loss type: 'mse' (L2) or 'mae' (L1)")
    parser.add_argument("--vort_weight", type=float, default=0.0,
                        help="Weight a1 for vorticity residual loss (0 = disabled)")
    parser.add_argument("--div_weight", type=float, default=0.0,
                        help="Weight a2 for divergence residual loss (0 = disabled)")
    parser.add_argument("--speed_eps", type=float, default=0.01)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model
    parser.add_argument("--base_channels", type=int, default=32)
    
    # Infrastructure
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Pin memory for faster GPU transfer")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    
    # Wandb
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="ssh2uv")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs='*', default=None)
    parser.add_argument("--wandb_offline", action="store_true", default=False)
    parser.add_argument("--wandb_log_model", action="store_true", default=False)
    
    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override with config values (CLI takes precedence)
        if args.mmap_dir is None:
            args.mmap_dir = config.mmap_dir
        if args.outdir == parser.get_default("outdir"):
            args.outdir = config.output_dir
        # Apply other config values as defaults
        # Config attribute names that differ from the CLI flag names
        config_values = {
            'lr': config.learning_rate,
            'base_channels': config.model_params.get('base_channels'),
        }
        for key in ['epochs', 'batch_size', 'lr', 'weight_decay', 'dir_weight',
                    'loss_type', 'vort_weight', 'div_weight',
                    'speed_eps', 'early_stop_patience', 'warmup_epochs', 'seed',
                    'base_channels', 'num_workers', 'prefetch_factor', 'val_fraction',
                    'grad_clip_norm', 'amp']:
            if getattr(args, key) == parser.get_default(key):
                config_val = config_values.get(key, getattr(config, key, None))
                if config_val is not None:
                    setattr(args, key, config_val)
        
        # Data format settings from config
        if args.data_format == "auto" and config.data_format:
            args.data_format = config.data_format
        if args.data_file is None and config.data_file:
            args.data_file = config.data_file
        if args.input_vars is None:
            args.input_vars = config.input_vars
        if args.output_vars is None:
            args.output_vars = config.output_vars
        if args.channel_order is None and config.channel_order:
            args.channel_order = config.channel_order
        
        # Wandb settings from config
        if not args.wandb and config.wandb_enabled:
            args.wandb = True
        if args.wandb_project == parser.get_default("wandb_project"):
            args.wandb_project = config.wandb_project
        if args.wandb_entity == parser.get_default("wandb_entity") and config.wandb_entity is not None:
            args.wandb_entity = config.wandb_entity
        if args.wandb_run_name == parser.get_default("wandb_run_name") and config.wandb_run_name is not None:
            args.wandb_run_name = config.wandb_run_name
        if args.wandb_tags == parser.get_default("wandb_tags"):
            args.wandb_tags = config.wandb_tags
        if args.wandb_offline == parser.get_default("wandb_offline"):
            args.wandb_offline = config.wandb_offline
        if args.wandb_log_model == parser.get_default("wandb_log_model"):
            args.wandb_log_model = config.wandb_log_model
    
    if args.mmap_dir is None:
        parser.error("--mmap_dir is required (or provide --config)")
    
    return args


def main():
    args = parse_args()
    
    train(
        mmap_dir=args.mmap_dir,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dir_weight=args.dir_weight,
        speed_eps=args.speed_eps,
        val_fraction=args.val_fraction,
        temporal_split=args.temporal_split,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        base_channels=args.base_channels,
        amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        exclude_zeros=args.exclude_zeros,
        loss_type=args.loss_type,
        vort_weight=args.vort_weight,
        div_weight=args.div_weight,
        resume_from=args.resume_from,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_offline=args.wandb_offline,
        wandb_log_model=args.wandb_log_model,
        # New data format options
        input_vars_override=args.input_vars,
        output_vars_override=args.output_vars,
        data_format=args.data_format,
        data_file=args.data_file,
        channel_order=args.channel_order,
    )


if __name__ == "__main__":
    main()
