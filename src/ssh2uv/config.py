"""Experiment configuration loader and validator.

Loads JSON experiment configs and provides validation and default values.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from copy import deepcopy


# Available variables in zarr data
AVAILABLE_INPUT_VARS = ['SSH', 'SST', 'SSS', 'TAUX', 'TAUY']
AVAILABLE_OUTPUT_VARS = ['u_sfc', 'v_sfc', 'uE_sfc', 'vN_sfc', 'u_k1', 'v_k1', 'uE_k1', 'vN_k1']

# Default configuration values
DEFAULT_CONFIG = {
    "experiment": {
        "name": "unnamed_experiment",
        "description": "",
        "tags": [],
        "version": "1.0",
        "output_dir": "runs/random_split/unnamed"
    },
    "data": {
        "zarr_source": None,
        "mmap_dir": None,
        "input_vars": ["SSH"],
        "output_vars": ["u_sfc", "v_sfc"],
        "normalize": True,
        "val_fraction": 0.1,
        "temporal_split": True,  # IMPORTANT: Default True for ocean data to prevent temporal leakage
        "exclude_zeros": True,
        "ignore_nonfinite": True,
        "max_timesteps": None
    },
    "model": {
        "architecture": "unet",
        "params": {
            "base_channels": 32
        }
    },
    "training": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "dir_weight": 0.05,
        "loss_type": "mse",
        "vort_weight": 0.0,
        "div_weight": 0.0,
        "speed_eps": 0.01,
        "early_stop_patience": 10,
        "target_loss": 0.0,
        "amp": True,
        "grad_clip_norm": 1.0,
        "warmup_epochs": 3,
        "seed": 42
    },
    "infrastructure": {
        "num_gpus": 1,
        "training_script": "scripts/random_split/train.py",
        "num_workers": 8,
        "prefetch_factor": 4,
        "pin_memory": True
    },
    "wandb": {
        "enabled": True,
        "project": "ssh2uv",
        "entity": None,
        "run_name": None,
        "tags": [],
        "offline": False,
        "log_model": False
    },
    "slurm": {
        "enabled": True,
        "partition": "medium",
        "time": "06:00:00",
        "gres": "gpu:1",
        "mem": "64G",
        "cpus_per_task": 16,
        "nodes": 1
    }
}

# Channel order definitions for stacked data format
STACKED_CHANNEL_ORDERS = {
    'DEEP':      ['Eta', 'windx', 'windy', 'theta', 'u1', 'v1'],
    'FULL':      ['Eta', 'windx', 'windy', 'theta', 'u0', 'v0'],
    'DEEP_GRAD': ['a', 'b', 'windx', 'windy', 'theta', 'u1', 'v1'],
    'FULL_GRAD': ['a', 'b', 'windx', 'windy', 'theta', 'u0', 'v0'],
}


@dataclass
class ExperimentConfig:
    """Structured experiment configuration."""
    
    # Experiment metadata
    name: str
    description: str
    tags: List[str]
    version: str
    output_dir: str
    
    # Data configuration
    zarr_source: Optional[str]
    mmap_dir: str
    input_vars: List[str]
    output_vars: List[str]
    normalize: bool
    val_fraction: float
    temporal_split: bool  # IMPORTANT: Should be True for ocean data
    exclude_zeros: bool
    ignore_nonfinite: bool
    max_timesteps: Optional[int]
    # New fields for stacked data format
    data_format: str  # 'auto', 'stacked', or 'mmap'
    data_file: Optional[str]  # For stacked format: the .npy file name
    channel_order: Optional[List[str]]  # For stacked format: channel names in order
    
    # Model configuration
    architecture: str
    model_params: Dict[str, Any]
    
    # Training configuration
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dir_weight: float
    loss_type: str          # 'mse' (L2) or 'mae' (L1)
    vort_weight: float      # a1: vorticity residual weight
    div_weight: float       # a2: divergence residual weight
    speed_eps: float
    early_stop_patience: int
    target_loss: float
    amp: bool
    grad_clip_norm: float
    warmup_epochs: int
    seed: int
    
    # Infrastructure configuration
    num_gpus: int
    training_script: str
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    
    # Wandb configuration
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_tags: List[str]
    wandb_offline: bool
    wandb_log_model: bool
    
    # SLURM configuration
    slurm_enabled: bool
    slurm_partition: str
    slurm_time: str
    slurm_gres: str
    slurm_mem: str
    slurm_cpus_per_task: int
    slurm_nodes: int
    
    # Raw config for reference
    raw_config: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict) -> 'ExperimentConfig':
        """Create ExperimentConfig from dictionary."""
        return cls(
            # Experiment
            name=config['experiment']['name'],
            description=config['experiment'].get('description', ''),
            tags=config['experiment'].get('tags', []),
            version=config['experiment'].get('version', '1.0'),
            output_dir=config['experiment']['output_dir'],
            
            # Data
            zarr_source=config['data'].get('zarr_source'),
            mmap_dir=config['data']['mmap_dir'],
            input_vars=config['data']['input_vars'],
            output_vars=config['data']['output_vars'],
            normalize=config['data'].get('normalize', True),
            val_fraction=config['data'].get('val_fraction', 0.1),
            temporal_split=config['data'].get('temporal_split', True),  # Default True for ocean data
            exclude_zeros=config['data'].get('exclude_zeros', True),
            ignore_nonfinite=config['data'].get('ignore_nonfinite', True),
            max_timesteps=config['data'].get('max_timesteps'),
            # New stacked data format fields
            data_format=config['data'].get('data_format', 'auto'),
            data_file=config['data'].get('data_file'),
            channel_order=config['data'].get('channel_order'),
            
            # Model
            architecture=config['model']['architecture'],
            model_params=config['model'].get('params', {}),
            
            # Training
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4),
            dir_weight=config['training'].get('dir_weight', 0.05),
            loss_type=config['training'].get('loss_type', 'mse'),
            vort_weight=config['training'].get('vort_weight', 0.0),
            div_weight=config['training'].get('div_weight', 0.0),
            speed_eps=config['training'].get('speed_eps', 0.01),
            early_stop_patience=config['training'].get('early_stop_patience', 10),
            target_loss=config['training'].get('target_loss', 0.0),
            amp=config['training'].get('amp', True),
            grad_clip_norm=config['training'].get('grad_clip_norm', 1.0),
            warmup_epochs=config['training'].get('warmup_epochs', 3),
            seed=config['training'].get('seed', 42),
            
            # Infrastructure
            num_gpus=config['infrastructure'].get('num_gpus', 1),
            training_script=config['infrastructure'].get("training_script", "scripts/random_split/train.py"),
            num_workers=config['infrastructure'].get('num_workers', 8),
            prefetch_factor=config['infrastructure'].get('prefetch_factor', 4),
            pin_memory=config['infrastructure'].get('pin_memory', True),
            
            # Wandb
            wandb_enabled=config['wandb'].get('enabled', True),
            wandb_project=config['wandb'].get('project', 'ssh2uv'),
            wandb_entity=config['wandb'].get('entity'),
            wandb_run_name=config['wandb'].get('run_name'),
            wandb_tags=config['wandb'].get('tags', []),
            wandb_offline=config['wandb'].get('offline', False),
            wandb_log_model=config['wandb'].get('log_model', False),
            
            # SLURM
            slurm_enabled=config['slurm'].get('enabled', True),
            slurm_partition=config['slurm'].get('partition', 'medium'),
            slurm_time=config['slurm'].get('time', '06:00:00'),
            slurm_gres=config['slurm'].get('gres', 'gpu:1'),
            slurm_mem=config['slurm'].get('mem', '64G'),
            slurm_cpus_per_task=config['slurm'].get('cpus_per_task', 16),
            slurm_nodes=config['slurm'].get('nodes', 1),
            
            raw_config=config
        )
    
    def to_dict(self) -> Dict:
        """Convert back to dictionary."""
        return self.raw_config
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size (batch_size * num_gpus for DDP)."""
        return self.batch_size * self.num_gpus


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str, validate: bool = True) -> ExperimentConfig:
    """Load experiment configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        validate: Whether to validate the config
    
    Returns:
        ExperimentConfig object
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        user_config = json.load(f)
    
    # Merge with defaults
    config = deep_merge(DEFAULT_CONFIG, user_config)
    
    if validate:
        validate_config(config)
    
    return ExperimentConfig.from_dict(config)


def validate_config(config: Dict) -> None:
    """Validate experiment configuration.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    # Required fields
    if not config.get('experiment', {}).get('name'):
        errors.append("experiment.name is required")
    
    if not config.get('experiment', {}).get('output_dir'):
        errors.append("experiment.output_dir is required")
    
    if not config.get('data', {}).get('mmap_dir'):
        errors.append("data.mmap_dir is required")
    
    # Validate variables
    input_vars = config.get('data', {}).get('input_vars', [])
    output_vars = config.get('data', {}).get('output_vars', [])
    
    if not input_vars:
        errors.append("data.input_vars cannot be empty")
    
    if not output_vars:
        errors.append("data.output_vars cannot be empty")
    
    # Validate numeric ranges
    training = config.get('training', {})
    
    if training.get('epochs', 0) <= 0:
        errors.append("training.epochs must be positive")
    
    if training.get('batch_size', 0) <= 0:
        errors.append("training.batch_size must be positive")
    
    if training.get('learning_rate', 0) <= 0:
        errors.append("training.learning_rate must be positive")
    
    val_fraction = config.get('data', {}).get('val_fraction', 0)
    if not (0 < val_fraction < 1):
        errors.append("data.val_fraction must be between 0 and 1")
    
    # Validate infrastructure
    infra = config.get('infrastructure', {})
    num_gpus = infra.get('num_gpus', 1)
    training_script = infra.get('training_script', '')
    
    if num_gpus > 1 and 'ddp' not in training_script.lower():
        errors.append(f"For num_gpus > 1, training_script should be a DDP script (got: {training_script})")
    
    # Raise all errors
    if errors:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))


def save_config(config: ExperimentConfig, output_path: str) -> None:
    """Save experiment configuration to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def print_config_summary(config: ExperimentConfig) -> None:
    """Print a summary of the experiment configuration."""
    print("=" * 70)
    print(f"Experiment: {config.name}")
    print("=" * 70)
    print(f"Description: {config.description}")
    print(f"Output dir:  {config.output_dir}")
    print()
    print("Data Configuration:")
    print(f"  Input vars:  {config.input_vars}")
    print(f"  Output vars: {config.output_vars}")
    print(f"  Mmap dir:    {config.mmap_dir}")
    print(f"  Normalize:   {config.normalize}")
    print()
    print("Model Configuration:")
    print(f"  Architecture: {config.architecture}")
    print(f"  Params:       {config.model_params}")
    print()
    print("Training Configuration:")
    print(f"  Epochs:       {config.epochs}")
    print(f"  Batch size:   {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  AMP:          {config.amp}")
    print()
    print("Infrastructure:")
    print(f"  GPUs:         {config.num_gpus}")
    print(f"  Script:       {config.training_script}")
    print(f"  Workers:      {config.num_workers}")
    print()
    print("Wandb:")
    print(f"  Enabled:      {config.wandb_enabled}")
    print(f"  Project:      {config.wandb_project}")
    print("=" * 70)


if __name__ == "__main__":
    # Test loading a config
    import sys
    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
        print_config_summary(config)
    else:
        print("Usage: python config_loader.py <config_path>")
