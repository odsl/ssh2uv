"""Memory-mapped dataset for high-performance GPU training.

This module provides PyTorch Datasets that read from pre-processed
memory-mapped numpy arrays for maximum I/O efficiency during training.

Supported formats:
1. Legacy MMap format: Separate train_input.npy, train_output.npy, train_mask.npy
2. Stacked format: Single file with all channels [Eta, windx, windy, theta, u, v]
   - global-final-64_stacked_DEEP.npy: (N, 6, 64, 64) [Eta, windx, windy, theta, u1, v1]
   - global-final-64_stacked_FULL.npy: (N, 6, 64, 64) [Eta, windx, windy, theta, u0, v0]
"""

import os
import json
import warnings
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# Channel mappings for stacked data format
# =============================================================================

# Default channel order in stacked files
STACKED_DEEP_CHANNELS = ['Eta', 'windx', 'windy', 'theta', 'u1', 'v1']
STACKED_FULL_CHANNELS = ['Eta', 'windx', 'windy', 'theta', 'u0', 'v0']

# New GRAD format: SSH replaced by x- and y-gradients of SSH
STACKED_DEEP_GRAD_CHANNELS = ['a', 'b', 'windx', 'windy', 'theta', 'u1', 'v1']
STACKED_FULL_GRAD_CHANNELS = ['a', 'b', 'windx', 'windy', 'theta', 'u0', 'v0']

# Human-readable names for visualization
CHANNEL_DISPLAY_NAMES = {
    'Eta': 'SSH (η)',
    'a': 'Δη along x (SSH gradient)',
    'b': 'Δη along y (SSH gradient)',
    'windx': 'Wind τx',
    'windy': 'Wind τy',
    'theta': 'SST (θ)',
    'u0': 'u (0.5 m)',
    'v0': 'v (0.5 m)',
    'u1': 'u (53 m)',
    'v1': 'v (53 m)',
}

# Units for visualization
CHANNEL_UNITS = {
    'Eta': 'm',
    'a': 'm per 2 km cell',
    'b': 'm per 2 km cell',
    'windx': 'N/m²',
    'windy': 'N/m²',
    'theta': '°C',
    'u0': 'm/s',
    'v0': 'm/s',
    'u1': 'm/s',
    'v1': 'm/s',
}


@dataclass
class NormStats:
    """Normalization statistics for input and output variables."""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    
    def to_torch(self, device: torch.device = None) -> 'NormStatsTorch':
        """Convert to torch tensors."""
        return NormStatsTorch(
            x_mean=torch.from_numpy(self.x_mean).float(),
            x_std=torch.from_numpy(self.x_std).float(),
            y_mean=torch.from_numpy(self.y_mean).float(),
            y_std=torch.from_numpy(self.y_std).float(),
            device=device
        )


@dataclass  
class NormStatsTorch:
    """Normalization statistics as torch tensors."""
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    device: Optional[torch.device] = None
    
    def to(self, device: torch.device) -> 'NormStatsTorch':
        """Move tensors to device."""
        return NormStatsTorch(
            x_mean=self.x_mean.to(device),
            x_std=self.x_std.to(device),
            y_mean=self.y_mean.to(device),
            y_std=self.y_std.to(device),
            device=device
        )
    
    def get_y_tensors(self, C_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get y_mean and y_std reshaped for broadcasting (1, C, 1, 1)."""
        y_mean = self.y_mean.reshape(1, C_out, 1, 1)
        y_std = self.y_std.reshape(1, C_out, 1, 1)
        return y_mean, y_std
    
    def get_x_tensors(self, C_in: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get x_mean and x_std reshaped for broadcasting (1, C, 1, 1)."""
        x_mean = self.x_mean.reshape(1, C_in, 1, 1)
        x_std = self.x_std.reshape(1, C_in, 1, 1)
        return x_mean, x_std


def load_norm_stats(mmap_dir: str) -> NormStats:
    """Load normalization stats from mmap directory.
    
    Args:
        mmap_dir: Directory containing norm_stats.npz or metadata.json
    
    Returns:
        NormStats object
    """
    # Try loading from norm_stats.npz
    stats_path = os.path.join(mmap_dir, 'norm_stats.npz')
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        return NormStats(
            x_mean=data['x_mean'],
            x_std=data['x_std'],
            y_mean=data['y_mean'],
            y_std=data['y_std']
        )
    
    # Fall back to metadata.json
    metadata_path = os.path.join(mmap_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'norm_stats' in metadata:
            ns = metadata['norm_stats']
            return NormStats(
                x_mean=np.array(ns['x_mean'], dtype=np.float32),
                x_std=np.array(ns['x_std'], dtype=np.float32),
                y_mean=np.array(ns['y_mean'], dtype=np.float32),
                y_std=np.array(ns['y_std'], dtype=np.float32)
            )
    
    raise FileNotFoundError(f"No norm stats found in {mmap_dir}")


def load_metadata(mmap_dir: str) -> Dict[str, Any]:
    """Load metadata from mmap directory.
    
    Args:
        mmap_dir: Directory containing metadata.json
    
    Returns:
        Metadata dictionary
    """
    metadata_path = os.path.join(mmap_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {mmap_dir}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


class MMapDataset(Dataset):
    """Memory-mapped dataset for efficient GPU training.
    
    Loads pre-processed data from memory-mapped numpy arrays.
    Data is stored in format: (N, C, H, W) for inputs/outputs.
    
    IMPORTANT CONTRACT FOR INVALID PIXELS:
    - Invalid pixels (NaN in original data) are stored as 0.0 in mmap files
    - The mask indicates validity: 1.0 = valid, 0.0 = invalid
    - When normalizing (either at preprocess or runtime), invalid pixels
      remain 0.0 (not normalized)
    - Training loss MUST use the mask to ignore invalid pixels
    - If data was pre-normalized during preprocessing (metadata['normalized']=True),
      runtime normalization is automatically skipped to prevent double-normalization
    
    Args:
        mmap_dir: Directory containing mmap files and metadata
        normalize: Whether to normalize data using stored stats
        exclude_zeros: DEPRECATED - no longer used. The mask already correctly
                      identifies invalid pixels. Checking for zeros after 
                      normalization was either redundant (invalid pixels already
                      in mask) or incorrect (masking valid ocean pixels with
                      genuine zero values like calm water conditions).
        shuffle_indices: Whether to use pre-shuffled indices
        subset_indices: Optional subset of indices to use (for train/val split)
    
    Files expected in mmap_dir:
        - train_input.npy: (N, C_in, H, W) float32
        - train_output.npy: (N, C_out, H, W) float32
        - train_mask.npy: (N, H, W) float32 (1.0=valid, 0.0=invalid)
        - metadata.json: Contains shapes, variables, stats, normalized flag
        - shuffle_indices.npy (optional): Pre-shuffled indices
        - norm_stats.npz (optional): Normalization statistics
    """
    
    def __init__(
        self,
        mmap_dir: str,
        normalize: bool = True,
        exclude_zeros: bool = True,  # DEPRECATED - kept for API compatibility
        shuffle_indices: bool = False,
        subset_indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.mmap_dir = mmap_dir
        
        # DEPRECATION WARNING: exclude_zeros is no longer used
        if exclude_zeros:
            import warnings
            warnings.warn(
                "[DEPRECATED] exclude_zeros=True is no longer used and will be ignored. "
                "The mask already correctly identifies invalid pixels. The previous "
                "exclude_zeros logic was buggy: after normalization, it was either "
                "redundant (invalid pixels already in mask) or incorrect (masking "
                "valid ocean pixels with genuine zero values like calm water).",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Load metadata
        self.metadata = load_metadata(mmap_dir)
        self.total_samples = self.metadata['total_samples']
        self.input_shape = self.metadata['input_shape']
        self.output_shape = self.metadata['output_shape']
        self.input_vars = self.metadata['input_vars']
        self.output_vars = self.metadata['output_vars']
        self.C_in = len(self.input_vars)
        self.C_out = len(self.output_vars)
        self.H = self.input_shape[2]
        self.W = self.input_shape[3]
        
        # Check if data was pre-normalized during preprocessing
        self.data_is_prenormalized = self.metadata.get('normalized', False)
        
        # Determine actual normalize behavior - PREVENT DOUBLE NORMALIZATION
        if self.data_is_prenormalized and normalize:
            import warnings
            warnings.warn(
                f"[DOUBLE-NORMALIZATION PREVENTED] Data in {mmap_dir} was already normalized "
                f"during preprocessing (metadata['normalized']=True). Skipping runtime normalization "
                f"to avoid corrupting the data. If you see unexpected results, check your preprocessing flags.",
                UserWarning
            )
            self.normalize = False  # Data already normalized, don't normalize again
        elif not self.data_is_prenormalized and normalize:
            self.normalize = True  # Need to normalize at runtime
        else:
            self.normalize = False  # User doesn't want normalization
        
        # Memory-map the arrays (lazy loading)
        self.input_mmap = np.load(
            os.path.join(mmap_dir, 'train_input.npy'),
            mmap_mode='r'
        )
        self.output_mmap = np.load(
            os.path.join(mmap_dir, 'train_output.npy'),
            mmap_mode='r'
        )
        self.mask_mmap = np.load(
            os.path.join(mmap_dir, 'train_mask.npy'),
            mmap_mode='r'
        )
        
        # Always load normalization stats (needed for denormalization during evaluation)
        self.stats = None
        try:
            self.stats = load_norm_stats(mmap_dir)
        except FileNotFoundError:
            if self.normalize:
                print(f"[WARN] No norm stats found, data will not be normalized")
                self.normalize = False
            # If data is pre-normalized but no stats file, try to get from metadata
            elif self.data_is_prenormalized and 'norm_stats' in self.metadata:
                ns = self.metadata['norm_stats']
                self.stats = NormStats(
                    x_mean=np.array(ns['x_mean'], dtype=np.float32),
                    x_std=np.array(ns['x_std'], dtype=np.float32),
                    y_mean=np.array(ns['y_mean'], dtype=np.float32),
                    y_std=np.array(ns['y_std'], dtype=np.float32)
                )
        
        # Handle shuffling/subsetting
        if shuffle_indices:
            shuffle_path = os.path.join(mmap_dir, 'shuffle_indices.npy')
            if os.path.exists(shuffle_path):
                self.indices = np.load(shuffle_path)
            else:
                print("[WARN] shuffle_indices.npy not found, using sequential indices")
                self.indices = np.arange(self.total_samples)
        else:
            self.indices = np.arange(self.total_samples)
        
        # Apply subset if provided
        if subset_indices is not None:
            self.indices = self.indices[subset_indices]
        
        self.N = len(self.indices)
    
    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            x: (C_in, H, W) input tensor
            y: (C_out, H, W) output tensor
            mask: (1, H, W) validity mask
        """
        # Map to actual index
        actual_idx = self.indices[idx]
        
        # Read from mmap
        x = self.input_mmap[actual_idx].copy()  # (C_in, H, W)
        y = self.output_mmap[actual_idx].copy()  # (C_out, H, W)
        mask = self.mask_mmap[actual_idx].copy()  # (H, W)
        
        # Normalize if requested - USE MASK to keep invalid pixels as 0.0
        # This ensures consistent sentinel value regardless of normalize timing
        if self.normalize and self.stats is not None:
            # Create broadcastable mask (C, H, W) for element-wise ops
            mask_x = mask[None, :, :]  # (1, H, W) broadcasts to (C_in, H, W)
            mask_y = mask[None, :, :]  # (1, H, W) broadcasts to (C_out, H, W)
            
            # Normalize only where mask is valid (1.0), keep 0.0 where invalid
            x_norm = (x - self.stats.x_mean[:, None, None]) / self.stats.x_std[:, None, None]
            y_norm = (y - self.stats.y_mean[:, None, None]) / self.stats.y_std[:, None, None]
            
            x = np.where(mask_x > 0.5, x_norm, 0.0)
            y = np.where(mask_y > 0.5, y_norm, 0.0)
        
        # NOTE: exclude_zeros logic was removed (see deprecation warning in __init__).
        # The mask already correctly identifies invalid pixels from preprocessing.
        # Checking for zeros after normalization was either:
        # - Redundant: invalid pixels (0.0) already caught by mask
        # - Wrong: valid ocean pixels with genuine zero values (calm water) got masked
        
        # Convert to tensors
        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float()
        mask_t = torch.from_numpy(mask[None, :, :]).float()  # Add channel dim
        
        return x_t, y_t, mask_t
    
    def get_stats(self) -> Optional[NormStats]:
        """Get normalization statistics."""
        return self.stats
    
    def get_stats_torch(self, device: torch.device = None) -> Optional[NormStatsTorch]:
        """Get normalization statistics as torch tensors."""
        if self.stats is None:
            return None
        return self.stats.to_torch(device)


def create_train_val_datasets(
    mmap_dir: str,
    val_fraction: float = 0.1,
    normalize: bool = True,
    exclude_zeros: bool = False,  # DEPRECATED - no longer used, kept for API compatibility
    seed: int = 42,
    temporal_split: bool = True,  # DEFAULT: temporal split for ocean data
    verbose: bool = True,  # Set False in DDP for non-rank-0 processes
) -> Tuple[MMapDataset, MMapDataset]:
    """Create train and validation datasets from mmap directory.
    
    IMPORTANT FOR OCEANOGRAPHIC DATA:
    ---------------------------------
    Ocean data (like LLC4320) is temporally correlated. Using random index splits
    causes validation samples to come from the same timesteps as training samples,
    leading to data leakage and overly optimistic validation metrics.
    
    By default, this function uses TEMPORAL SPLIT where validation comes from
    the last N% of timesteps (chronologically), ensuring no temporal leakage.
    
    Args:
        mmap_dir: Directory containing mmap files
        val_fraction: Fraction of data for validation (by timestep if temporal_split=True)
        normalize: Whether to normalize data
        exclude_zeros: DEPRECATED - no longer used. Kept for API compatibility.
        seed: Random seed (only used for random split)
        temporal_split: If True (default), split by timestep to prevent temporal leakage.
                       If False, random index split (WARNING: causes data leakage on temporal data)
        verbose: If True, print split information. Set False in DDP for non-rank-0 processes.
    
    Returns:
        (train_dataset, val_dataset)
    """
    import warnings
    
    # Load metadata
    metadata = load_metadata(mmap_dir)
    total_samples = metadata['total_samples']
    num_timesteps = metadata.get('num_timesteps', 1)
    is_temporal_data = metadata.get('temporal_data', True)  # Assume temporal by default
    
    if temporal_split:
        # TEMPORAL SPLIT: Last N% of timesteps go to validation
        # Use timestep_ids.npy as the source of truth (sample-level mapping)
        timestep_ids_path = os.path.join(mmap_dir, 'timestep_ids.npy')
        
        if os.path.exists(timestep_ids_path):
            # PRIMARY PATH: Use timestep_ids.npy for accurate sample-level split
            timestep_ids = np.load(timestep_ids_path)
            
            # Validate consistency
            if len(timestep_ids) != total_samples:
                raise ValueError(
                    f"timestep_ids.npy has {len(timestep_ids)} entries but metadata says "
                    f"{total_samples} samples. Data may be corrupted - re-run preprocessing."
                )
            
            # Get unique timesteps and their counts
            unique_timesteps = np.unique(timestep_ids)
            num_timesteps = len(unique_timesteps)
            
            # Calculate timestep split point
            n_val_timesteps = max(1, int(num_timesteps * val_fraction))
            n_train_timesteps = num_timesteps - n_val_timesteps
            
            # Split by timestep: train gets timesteps [0, n_train_timesteps)
            # val gets timesteps [n_train_timesteps, num_timesteps)
            train_timesteps = set(unique_timesteps[:n_train_timesteps])
            val_timesteps = set(unique_timesteps[n_train_timesteps:])
            
            # Get sample indices for each split
            train_indices = np.where(np.isin(timestep_ids, list(train_timesteps)))[0]
            val_indices = np.where(np.isin(timestep_ids, list(val_timesteps)))[0]
            
            if verbose:
                print(f"[TEMPORAL SPLIT] Using timestep_ids.npy (sample-level mapping)")
                print(f"[TEMPORAL SPLIT] Train: timesteps 0-{n_train_timesteps-1} ({len(train_indices):,} samples)")
                print(f"[TEMPORAL SPLIT] Val:   timesteps {n_train_timesteps}-{num_timesteps-1} ({len(val_indices):,} samples)")
            
        else:
            # FALLBACK PATH: Use samples_per_timestep from metadata
            warnings.warn(
                f"timestep_ids.npy not found in {mmap_dir}. Falling back to samples_per_timestep "
                f"from metadata. Re-run preprocessing for accurate sample-level temporal splits.",
                UserWarning
            )
            
            samples_per_timestep = metadata.get('samples_per_timestep', None)
            
            if samples_per_timestep is None:
                # Double fallback: assume equal samples per timestep
                warnings.warn(
                    f"samples_per_timestep also not found in metadata. Assuming equal distribution. "
                    f"Re-run preprocessing to get accurate temporal splits.",
                    UserWarning
                )
                base_samples = total_samples // num_timesteps
                remainder = total_samples % num_timesteps
                samples_per_timestep = [base_samples] * num_timesteps
                for i in range(remainder):
                    samples_per_timestep[num_timesteps - 1 - i] += 1
                
                assert sum(samples_per_timestep) == total_samples, \
                    f"Fallback samples_per_timestep sum {sum(samples_per_timestep)} != total_samples {total_samples}"
            
            # Calculate timestep split point
            n_val_timesteps = max(1, int(num_timesteps * val_fraction))
            n_train_timesteps = num_timesteps - n_val_timesteps
            
            # Compute sample indices for train (early timesteps) and val (late timesteps)
            train_end_idx = sum(samples_per_timestep[:n_train_timesteps])
            train_indices = np.arange(0, train_end_idx)
            val_indices = np.arange(train_end_idx, total_samples)
            
            if verbose:
                print(f"[TEMPORAL SPLIT] Using samples_per_timestep fallback")
                print(f"[TEMPORAL SPLIT] Train: timesteps 0-{n_train_timesteps-1} ({len(train_indices):,} samples)")
                print(f"[TEMPORAL SPLIT] Val:   timesteps {n_train_timesteps}-{num_timesteps-1} ({len(val_indices):,} samples)")
        
    else:
        # RANDOM SPLIT - WARN about temporal leakage
        if is_temporal_data:
            warnings.warn(
                f"\\n{'='*70}\\n"
                f"WARNING: RANDOM SPLIT ON TEMPORAL DATA DETECTED!\\n"
                f"{'='*70}\\n"
                f"This dataset appears to be temporal (ocean) data. Using random index splits\\n"
                f"causes train and validation samples to come from the SAME timesteps,\\n"
                f"leading to DATA LEAKAGE and OVERLY OPTIMISTIC validation metrics.\\n"
                f"\\n"
                f"The model will appear to generalize but is actually memorizing temporal structure.\\n"
                f"\\n"
                f"RECOMMENDATION: Use temporal_split=True (the default) for oceanographic data.\\n"
                f"{'='*70}",
                UserWarning
            )
        
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(total_samples)
        
        n_val = int(total_samples * val_fraction)
        n_train = total_samples - n_val
        
        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train:]
        
        if verbose:
            print(f"[RANDOM SPLIT] Train: {len(train_indices):,} samples (random)")
            print(f"[RANDOM SPLIT] Val:   {len(val_indices):,} samples (random)")
    
    # Create datasets
    train_dataset = MMapDataset(
        mmap_dir=mmap_dir,
        normalize=normalize,
        exclude_zeros=exclude_zeros,
        shuffle_indices=False,
        subset_indices=None,
    )
    train_dataset.indices = train_indices
    train_dataset.N = len(train_indices)
    
    val_dataset = MMapDataset(
        mmap_dir=mmap_dir,
        normalize=normalize,
        exclude_zeros=exclude_zeros,
        shuffle_indices=False,
        subset_indices=None,
    )
    val_dataset.indices = val_indices
    val_dataset.N = len(val_indices)
    
    return train_dataset, val_dataset


# =============================================================================
# Stacked Numpy Dataset (New Format)
# =============================================================================

class StackedNumpyDataset(Dataset):
    """Dataset for stacked numpy format with configurable input/output channels.
    
    This dataset handles the new data format where all variables are stacked
    in a single numpy array with shape (N, C, H, W).
    
    New data format:
    - global-final-64_stacked_DEEP.npy: (N, 6, 64, 64) [Eta, windx, windy, theta, u1, v1]
    - global-final-64_stacked_FULL.npy: (N, 6, 64, 64) [Eta, windx, windy, theta, u0, v0]
    - Per-variable files: Eta.npy, windx.npy, etc. each (N, 64, 64)
    - Metadata with train_indices and test_indices for temporal split
    
    Configuration-based variable selection:
    - input_vars: List of channel names to use as input (e.g., ['Eta', 'windx', 'windy'])
    - output_vars: List of channel names to use as output (e.g., ['u1', 'v1'])
    
    Args:
        data_dir: Directory containing the stacked numpy files and metadata
        data_file: Name of the stacked data file (default: auto-detect)
        channel_order: List of channel names in order they appear in data
        input_vars: List of variable names to use as inputs
        output_vars: List of variable names to use as outputs
        normalize: Whether to normalize data (computes stats if needed)
        subset_indices: Optional indices to use (for train/val/test split)
    """
    
    def __init__(
        self,
        data_dir: str,
        data_file: Optional[str] = None,
        channel_order: Optional[List[str]] = None,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        normalize: bool = True,
        subset_indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.normalize = normalize
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Detect data file and channel order
        self.data_file, self.channel_order = self._detect_data_format(data_file, channel_order)
        
        # Set input/output variables
        self.input_vars = input_vars or self.metadata.get('input_vars', self.channel_order[:4])
        self.output_vars = output_vars or self.metadata.get('output_vars', self.channel_order[4:])
        
        # Validate variables exist in channel order
        all_vars = set(self.channel_order)
        for var in self.input_vars + self.output_vars:
            if var not in all_vars:
                raise ValueError(
                    f"Variable '{var}' not found in channel order {self.channel_order}. "
                    f"Available: {all_vars}"
                )
        
        # Get channel indices
        self.input_indices = [self.channel_order.index(v) for v in self.input_vars]
        self.output_indices = [self.channel_order.index(v) for v in self.output_vars]
        
        # Memory-map the data file
        data_path = os.path.join(data_dir, self.data_file)
        self.data_mmap = np.load(data_path, mmap_mode='r')
        
        # Get dimensions
        self.total_samples = self.data_mmap.shape[0]
        self.C_total = self.data_mmap.shape[1]
        self.H = self.data_mmap.shape[2]
        self.W = self.data_mmap.shape[3]
        self.C_in = len(self.input_vars)
        self.C_out = len(self.output_vars)
        
        # Store shapes for compatibility with existing code
        self.input_shape = (self.total_samples, self.C_in, self.H, self.W)
        self.output_shape = (self.total_samples, self.C_out, self.H, self.W)
        
        # Handle indices (for train/val/test split)
        if subset_indices is not None:
            self.indices = subset_indices
        else:
            self.indices = np.arange(self.total_samples)
        self.N = len(self.indices)
        
        # Load or compute normalization stats
        self.stats = self._load_or_compute_stats() if normalize else None
        
        # Data is not pre-normalized in new format
        self.data_is_prenormalized = self.metadata.get('normalized', False)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from data directory."""
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _detect_data_format(
        self, 
        data_file: Optional[str],
        channel_order: Optional[List[str]]
    ) -> Tuple[str, List[str]]:
        """Auto-detect data file and channel order."""
        if data_file and channel_order:
            return data_file, channel_order
        
        # Try to find stacked files
        deep_file = 'global-final-64_stacked_DEEP.npy'
        full_file = 'global-final-64_stacked_FULL.npy'
        deep_grad_file = 'global-final-64_stacked_DEEP_GRAD.npy'
        full_grad_file = 'global-final-64_stacked_FULL_GRAD.npy'
        
        deep_exists = os.path.exists(os.path.join(self.data_dir, deep_file))
        full_exists = os.path.exists(os.path.join(self.data_dir, full_file))
        deep_grad_exists = os.path.exists(os.path.join(self.data_dir, deep_grad_file))
        full_grad_exists = os.path.exists(os.path.join(self.data_dir, full_grad_file))
        
        if data_file:
            if 'DEEP_GRAD' in data_file.upper() or 'DEEP-GRAD' in data_file.upper():
                return data_file, channel_order or STACKED_DEEP_GRAD_CHANNELS
            elif 'FULL_GRAD' in data_file.upper() or 'FULL-GRAD' in data_file.upper():
                return data_file, channel_order or STACKED_FULL_GRAD_CHANNELS
            elif 'DEEP' in data_file.upper():
                return data_file, channel_order or STACKED_DEEP_CHANNELS
            elif 'FULL' in data_file.upper():
                return data_file, channel_order or STACKED_FULL_CHANNELS
            else:
                return data_file, channel_order or self.metadata.get('channel_order', STACKED_DEEP_CHANNELS)
        
        # Auto-detect: prefer GRAD files if present
        if deep_grad_exists:
            return deep_grad_file, STACKED_DEEP_GRAD_CHANNELS
        elif full_grad_exists:
            return full_grad_file, STACKED_FULL_GRAD_CHANNELS
        elif deep_exists:
            return deep_file, STACKED_DEEP_CHANNELS
        elif full_exists:
            return full_file, STACKED_FULL_CHANNELS
        else:
            # Fall back to any .npy file that looks like stacked data
            npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npy') and 'stacked' in f.lower()]
            if npy_files:
                return npy_files[0], channel_order or self.metadata.get('channel_order', STACKED_DEEP_CHANNELS)
            
            raise FileNotFoundError(
                f"No stacked data file found in {self.data_dir}. "
                f"Expected: {deep_file}, {full_file}, {deep_grad_file}, or {full_grad_file}"
            )
    
    def _load_or_compute_stats(self) -> NormStats:
        """Load normalization stats or compute from data."""
        # Try loading pre-computed stats
        stats_path = os.path.join(self.data_dir, 'norm_stats.npz')
        if os.path.exists(stats_path):
            data = np.load(stats_path)
            # Check if stats match our input/output variables
            if 'channel_names' in data and 'channel_means' in data:
                return self._extract_stats_for_vars(data)
            elif 'x_mean' in data and 'y_mean' in data:
                # Legacy format - assume matches current config
                return NormStats(
                    x_mean=data['x_mean'].astype(np.float32),
                    x_std=data['x_std'].astype(np.float32),
                    y_mean=data['y_mean'].astype(np.float32),
                    y_std=data['y_std'].astype(np.float32)
                )
        
        # Try per-channel stats from metadata
        if 'norm_stats_per_channel' in self.metadata:
            return self._extract_stats_from_metadata()
        
        # Compute stats from data
        print("[INFO] Computing normalization statistics from data...")
        return self._compute_stats()
    
    def _extract_stats_from_metadata(self) -> NormStats:
        """Extract stats for selected variables from metadata."""
        per_channel = self.metadata['norm_stats_per_channel']
        
        x_mean = np.array([per_channel[v]['mean'] for v in self.input_vars], dtype=np.float32)
        x_std = np.array([per_channel[v]['std'] for v in self.input_vars], dtype=np.float32)
        y_mean = np.array([per_channel[v]['mean'] for v in self.output_vars], dtype=np.float32)
        y_std = np.array([per_channel[v]['std'] for v in self.output_vars], dtype=np.float32)
        
        # Ensure std is not zero
        x_std = np.maximum(x_std, 1e-8)
        y_std = np.maximum(y_std, 1e-8)
        
        return NormStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    
    def _extract_stats_for_vars(self, data: np.lib.npyio.NpzFile) -> NormStats:
        """Extract stats for selected variables from loaded npz file."""
        channel_names = list(data['channel_names'])
        channel_means = data['channel_means']
        channel_stds = data['channel_stds']
        
        x_mean = np.array([channel_means[channel_names.index(v)] for v in self.input_vars], dtype=np.float32)
        x_std = np.array([channel_stds[channel_names.index(v)] for v in self.input_vars], dtype=np.float32)
        y_mean = np.array([channel_means[channel_names.index(v)] for v in self.output_vars], dtype=np.float32)
        y_std = np.array([channel_stds[channel_names.index(v)] for v in self.output_vars], dtype=np.float32)
        
        x_std = np.maximum(x_std, 1e-8)
        y_std = np.maximum(y_std, 1e-8)
        
        return NormStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    
    def _compute_stats(self) -> NormStats:
        """Compute normalization stats from data."""
        # Use a subset for efficiency
        n_samples = min(10000, self.total_samples)
        sample_indices = np.random.choice(self.total_samples, n_samples, replace=False)
        
        # Compute per-channel stats for all channels
        channel_means = []
        channel_stds = []
        
        for c in range(self.C_total):
            values = []
            for idx in sample_indices:
                patch = self.data_mmap[idx, c].flatten()
                # Exclude zeros (likely NaN-replaced values)
                valid = patch[patch != 0.0]
                if len(valid) > 0:
                    values.extend(valid.tolist())
            
            if len(values) > 0:
                channel_means.append(np.mean(values))
                channel_stds.append(np.std(values))
            else:
                channel_means.append(0.0)
                channel_stds.append(1.0)
        
        # Extract stats for our selected variables
        x_mean = np.array([channel_means[i] for i in self.input_indices], dtype=np.float32)
        x_std = np.array([channel_stds[i] for i in self.input_indices], dtype=np.float32)
        y_mean = np.array([channel_means[i] for i in self.output_indices], dtype=np.float32)
        y_std = np.array([channel_stds[i] for i in self.output_indices], dtype=np.float32)
        
        x_std = np.maximum(x_std, 1e-8)
        y_std = np.maximum(y_std, 1e-8)
        
        print(f"  Input  ({self.input_vars}): mean={x_mean}, std={x_std}")
        print(f"  Output ({self.output_vars}): mean={y_mean}, std={y_std}")
        
        return NormStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    
    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            x: (C_in, H, W) input tensor
            y: (C_out, H, W) output tensor
            mask: (1, H, W) validity mask (1.0=valid, 0.0=invalid)
        """
        actual_idx = self.indices[idx]
        
        # Load full sample
        sample = self.data_mmap[actual_idx].copy()  # (C_total, H, W)
        
        # Extract input and output channels
        x = sample[self.input_indices]  # (C_in, H, W)
        y = sample[self.output_indices]  # (C_out, H, W)
        
        # Create mask: valid where no channel is exactly 0 (NaN replacement)
        # Check all channels in the full sample
        mask = (np.abs(sample).sum(axis=0) > 1e-10).astype(np.float32)  # (H, W)
        
        # Normalize if requested
        if self.normalize and self.stats is not None:
            mask_3d_x = mask[None, :, :]  # (1, H, W)
            mask_3d_y = mask[None, :, :]
            
            x_norm = (x - self.stats.x_mean[:, None, None]) / self.stats.x_std[:, None, None]
            y_norm = (y - self.stats.y_mean[:, None, None]) / self.stats.y_std[:, None, None]
            
            x = np.where(mask_3d_x > 0.5, x_norm, 0.0)
            y = np.where(mask_3d_y > 0.5, y_norm, 0.0)
        
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(mask[None, :, :]).float()  # (1, H, W)
        )
    
    def get_stats(self) -> Optional[NormStats]:
        """Get normalization statistics."""
        return self.stats
    
    def get_stats_torch(self, device: torch.device = None) -> Optional[NormStatsTorch]:
        """Get normalization statistics as torch tensors."""
        if self.stats is None:
            return None
        return self.stats.to_torch(device)


def create_stacked_train_val_datasets(
    data_dir: str,
    data_file: Optional[str] = None,
    channel_order: Optional[List[str]] = None,
    input_vars: Optional[List[str]] = None,
    output_vars: Optional[List[str]] = None,
    val_fraction: float = 0.2,
    normalize: bool = True,
    seed: int = 42,
    use_precomputed_split: bool = True,
    verbose: bool = True,
) -> Tuple[StackedNumpyDataset, StackedNumpyDataset]:
    """Create train and validation datasets for stacked numpy format.
    
    This function handles the new data format with pre-computed temporal splits.
    
    Args:
        data_dir: Directory containing stacked data and metadata
        data_file: Name of stacked data file (auto-detect if None)
        channel_order: Channel names in order (auto-detect if None)
        input_vars: Variables to use as input
        output_vars: Variables to use as output
        val_fraction: Fraction for validation (only used if no precomputed split)
        normalize: Whether to normalize data
        seed: Random seed for fallback split
        use_precomputed_split: Use train_indices/test_indices from metadata
        verbose: Print split information
    
    Returns:
        (train_dataset, val_dataset)
    """
    # Load metadata — try master metadata JSON first, then fallback to metadata.json
    master_meta_path = os.path.join(data_dir, 'global-final-64_master_metadata.json')
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(master_meta_path):
        with open(master_meta_path, 'r') as f:
            metadata = json.load(f)
    elif os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Check for pre-computed temporal split indices
    train_indices = None
    val_indices = None
    
    if use_precomputed_split:
        # Try to load from metadata (supports both 'train_indices' and 'test_indices' keys,
        # as well as the master metadata format with those same keys)
        if 'train_indices' in metadata and 'test_indices' in metadata:
            train_indices = np.array(metadata['train_indices'])
            val_indices = np.array(metadata['test_indices'])
            if verbose:
                print(f"[TEMPORAL SPLIT] Using precomputed split from metadata")
                print(f"[TEMPORAL SPLIT] Train: {len(train_indices):,} samples")
                print(f"[TEMPORAL SPLIT] Test:  {len(val_indices):,} samples")
        # Also accept 'val_indices' key for backwards compatibility
        elif 'train_indices' in metadata and 'val_indices' in metadata:
            train_indices = np.array(metadata['train_indices'])
            val_indices = np.array(metadata['val_indices'])
            if verbose:
                print(f"[TEMPORAL SPLIT] Using precomputed split from metadata (val key)")
                print(f"[TEMPORAL SPLIT] Train: {len(train_indices):,} samples")
                print(f"[TEMPORAL SPLIT] Val:   {len(val_indices):,} samples")
        
        # Try to load from separate files
        elif os.path.exists(os.path.join(data_dir, 'train_indices.npy')):
            train_indices = np.load(os.path.join(data_dir, 'train_indices.npy'))
            val_indices = np.load(os.path.join(data_dir, 'test_indices.npy'))
            if verbose:
                print(f"[TEMPORAL SPLIT] Using precomputed split from files")
                print(f"[TEMPORAL SPLIT] Train: {len(train_indices):,} samples")
                print(f"[TEMPORAL SPLIT] Val:   {len(val_indices):,} samples")
    
    # Fallback: create random split (with warning)
    if train_indices is None:
        # First, create a temp dataset to get total samples
        temp_dataset = StackedNumpyDataset(
            data_dir=data_dir,
            data_file=data_file,
            channel_order=channel_order,
            input_vars=input_vars,
            output_vars=output_vars,
            normalize=False,
        )
        total_samples = temp_dataset.total_samples
        
        warnings.warn(
            f"No precomputed train/test split found in {data_dir}. "
            f"Using random split with val_fraction={val_fraction}. "
            f"WARNING: Random splits may cause temporal leakage!",
            UserWarning
        )
        
        np.random.seed(seed)
        all_indices = np.random.permutation(total_samples)
        n_val = int(total_samples * val_fraction)
        train_indices = all_indices[n_val:]
        val_indices = all_indices[:n_val]
        
        if verbose:
            print(f"[RANDOM SPLIT] Train: {len(train_indices):,} samples")
            print(f"[RANDOM SPLIT] Val:   {len(val_indices):,} samples")
    
    # Create datasets
    train_dataset = StackedNumpyDataset(
        data_dir=data_dir,
        data_file=data_file,
        channel_order=channel_order,
        input_vars=input_vars,
        output_vars=output_vars,
        normalize=normalize,
        subset_indices=train_indices,
    )
    
    val_dataset = StackedNumpyDataset(
        data_dir=data_dir,
        data_file=data_file,
        channel_order=channel_order,
        input_vars=input_vars,
        output_vars=output_vars,
        normalize=normalize,
        subset_indices=val_indices,
    )
    
    # Share stats between datasets
    if train_dataset.stats is not None:
        val_dataset.stats = train_dataset.stats
    
    return train_dataset, val_dataset


def detect_data_format(data_dir: str) -> str:
    """Detect the data format in a directory.
    
    Returns:
        'stacked': New stacked numpy format
        'mmap': Legacy mmap format with separate input/output files
    """
    # Check for stacked format files
    stacked_files = [
        'global-final-64_stacked_DEEP.npy',
        'global-final-64_stacked_FULL.npy',
        'global-final-64_stacked_DEEP_GRAD.npy',
        'global-final-64_stacked_FULL_GRAD.npy',
    ]
    for f in stacked_files:
        if os.path.exists(os.path.join(data_dir, f)):
            return 'stacked'
    
    # Check for any stacked file pattern
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    for f in npy_files:
        if 'stacked' in f.lower():
            return 'stacked'
    
    # Check for legacy mmap format
    if os.path.exists(os.path.join(data_dir, 'train_input.npy')):
        return 'mmap'
    
    raise ValueError(f"Could not detect data format in {data_dir}")


def create_datasets_auto(
    data_dir: str,
    input_vars: Optional[List[str]] = None,
    output_vars: Optional[List[str]] = None,
    val_fraction: float = 0.2,
    normalize: bool = True,
    temporal_split: bool = True,
    seed: int = 42,
    verbose: bool = True,
    **kwargs
) -> Tuple[Dataset, Dataset]:
    """Auto-detect format and create appropriate datasets.
    
    This is the recommended entry point for creating datasets.
    It automatically detects the data format and uses the appropriate
    dataset class.
    
    Args:
        data_dir: Directory containing training data
        input_vars: Input variable names (auto-detect if None)
        output_vars: Output variable names (auto-detect if None)
        val_fraction: Validation fraction
        normalize: Whether to normalize
        temporal_split: Use temporal split (for legacy format)
        seed: Random seed
        verbose: Print info
        **kwargs: Additional arguments passed to dataset constructors
    
    Returns:
        (train_dataset, val_dataset)
    """
    fmt = detect_data_format(data_dir)
    
    if verbose:
        print(f"[INFO] Detected data format: {fmt}")
    
    if fmt == 'stacked':
        return create_stacked_train_val_datasets(
            data_dir=data_dir,
            input_vars=input_vars,
            output_vars=output_vars,
            val_fraction=val_fraction,
            normalize=normalize,
            seed=seed,
            verbose=verbose,
            **kwargs
        )
    else:
        return create_train_val_datasets(
            mmap_dir=data_dir,
            val_fraction=val_fraction,
            normalize=normalize,
            seed=seed,
            temporal_split=temporal_split,
            verbose=verbose,
            **kwargs
        )


if __name__ == "__main__":
    # Test loading a dataset
    import sys
    if len(sys.argv) > 1:
        mmap_dir = sys.argv[1]
        print(f"Loading dataset from {mmap_dir}")
        
        dataset = MMapDataset(mmap_dir)
        print(f"Total samples: {len(dataset)}")
        print(f"Input vars: {dataset.input_vars}")
        print(f"Output vars: {dataset.output_vars}")
        print(f"Input shape: {dataset.input_shape}")
        print(f"Output shape: {dataset.output_shape}")
        
        # Test loading a sample
        x, y, mask = dataset[0]
        print(f"Sample x shape: {x.shape}")
        print(f"Sample y shape: {y.shape}")
        print(f"Sample mask shape: {mask.shape}")
    else:
        print("Usage: python mmap_dataset.py <mmap_dir>")
