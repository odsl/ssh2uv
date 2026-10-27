#!/usr/bin/env python3
"""
preprocess.py

Regrid LLC4320 Cape Basin patches from the native curvilinear grid to a
regular 2 km grid, compute SSH gradient channels, extract 64×64 patches,
and write the stacked .npy files consumed by scripts/train.py.

Input
-----
  cape_processed_data/          (written by notebooks/01_download_llc4320.ipynb)
      eta/, us_d/, ...           per-variable folders of per-granule .npy files;
                                 combined into <var>_combined.npy on first run

Output
------
  {prefix}_stacked_DEEP_GRAD.npy   (N, 7, 64, 64)  channels: [a, b, windx, windy, theta, u1, v1]
  {prefix}_stacked_FULL_GRAD.npy   (N, 7, 64, 64)  channels: [a, b, windx, windy, theta, u0, v0]
  {prefix}_{var}.npy               individual variable arrays
  {prefix}_{var}_metadata.json     per-patch metadata
  {prefix}_stacked_DEEP.npy        (N, 6, 64, 64)  channels: [Eta, windx, windy, theta, u1, v1]
  {prefix}_preprocess_metadata.json  config + temporal train/test indices (not used by default)

Usage
-----
  python scripts/preprocess.py

Environment variables (all optional)
-------------------------------------
  INTERP_MAIN_FOLDER          Input folder  [cape_processed_data]
  INTERP_OUTPUT_PREFIX        Output prefix [data/global-final-64]
  INTERP_NAN_THRESHOLD        Max NaN fraction per patch [0.0]
  INTERP_EDGE_TRIM            Timesteps trimmed from each end [50]
  INTERP_N_WORKERS            Worker threads (-1=all CPUs, 0=sequential) [-1]
  INTERP_TEST_MAX_FILES_PER_VAR  Limit files per var for testing [0=all]
  INTERP_TEST_MAX_TIMESTEPS   Limit timesteps for testing [0=all]
"""
import json
import multiprocessing as mp
import os
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from tqdm.auto import tqdm

# ── configuration ─────────────────────────────────────────────────────────────
MAIN_FOLDER    = os.getenv("INTERP_MAIN_FOLDER", "cape_processed_data")
SUBDOMAIN_SIZE = 64
OVERLAP        = 0
STEP           = SUBDOMAIN_SIZE - OVERLAP
GRID_STEP      = 2e3   # 2 km regular grid
OUTPUT_PREFIX  = os.getenv("INTERP_OUTPUT_PREFIX", f"data/global-final-{SUBDOMAIN_SIZE}")
NAN_THRESHOLD  = float(os.getenv("INTERP_NAN_THRESHOLD", "0.0"))
EDGE_TRIM      = int(os.getenv("INTERP_EDGE_TRIM", "50"))
N_WORKERS      = int(os.getenv("INTERP_N_WORKERS", "-1"))
TEST_MAX_FILES_PER_VAR = int(os.getenv("INTERP_TEST_MAX_FILES_PER_VAR", "0"))
TEST_MAX_TIMESTEPS     = int(os.getenv("INTERP_TEST_MAX_TIMESTEPS", "0"))

# Channel layout of each stacked output (SSH is replaced by gradient channels a, b)
COMBO_DEEP_VARS = ["windx", "windy", "theta", "u1", "v1"]
COMBO_FULL_VARS = ["windx", "windy", "theta", "u0", "v0"]
REQUIRED_VARS   = ["Eta", "windx", "windy", "theta", "u0", "v0", "u1", "v1"]

VAR_SOURCE_MAP = {
    "Eta":   "eta",
    "theta": "theta",
    "u0":    "us_s",
    "v0":    "vs_s",
    "u1":    "us_d",
    "v1":    "vs_d",
    "windx": "oceaTAUX",
    "windy": "oceaTAUY",
}

progress_lock = threading.Lock()


# ── file helpers ──────────────────────────────────────────────────────────────

def _combined_filename(var_name: str) -> str:
    suffix = "_combined_test.npy" if TEST_MAX_FILES_PER_VAR > 0 else "_combined.npy"
    return f"{var_name}{suffix}"


def combine_and_save_subfolders() -> None:
    """Merge per-timestep .npy files into one combined array per variable."""
    if not os.path.isdir(MAIN_FOLDER):
        raise FileNotFoundError(f"Input folder not found: {MAIN_FOLDER}")

    subfolders = [f for f in os.listdir(MAIN_FOLDER)
                  if os.path.isdir(os.path.join(MAIN_FOLDER, f))]
    print(f"Found {len(subfolders)} subfolders: {subfolders}")

    for subfolder in subfolders:
        out = os.path.join(MAIN_FOLDER, _combined_filename(subfolder))
        if os.path.exists(out):
            print(f"  skip {subfolder} (already combined)")
            continue
        files = sorted(glob.glob(os.path.join(MAIN_FOLDER, subfolder, "*.npy")))
        if not files:
            print(f"  [WARN] no .npy in {subfolder}")
            continue
        if TEST_MAX_FILES_PER_VAR > 0:
            files = files[:TEST_MAX_FILES_PER_VAR]
        arr = np.stack([np.load(f) for f in tqdm(files, desc=subfolder)], axis=0)
        np.save(out, arr)
        print(f"  saved {out}  shape={arr.shape}")
        del arr


# ── grid & interpolation ──────────────────────────────────────────────────────

def get_triangulation(dxc: np.ndarray, dyc: np.ndarray):
    """Build Delaunay triangulation and target regular grid once."""
    xx = np.cumsum(dxc, axis=1) - dxc[:, :1]
    yy = np.cumsum(dyc, axis=0) - dyc[:1, :]
    points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    xmax = min(xx[0].max(), xx[-1].max())
    xg = np.arange(0, xmax, GRID_STEP)
    yg = np.arange(0, yy.max(), GRID_STEP)
    Xg, Yg = np.meshgrid(xg, yg)
    print(f"Delaunay: {len(points)} source points → target grid {Xg.shape}")
    return Delaunay(points), Xg, Yg


class FastInterpolator:
    """Reuse Delaunay triangulation for repeated interpolations."""
    def __init__(self, delaunay: Delaunay):
        self._d = delaunay

    def __call__(self, values: np.ndarray, Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
        return LinearNDInterpolator(self._d, values)(Xg, Yg)


def compute_valid_patch_mask(
    delaunay: Delaunay, Xg: np.ndarray, Yg: np.ndarray
) -> Tuple[List[Tuple[int, int]], dict]:
    """Determine which 64×64 patch positions have acceptable NaN coverage."""
    ny, nx = Xg.shape
    test = FastInterpolator(delaunay)(np.ones(delaunay.points.shape[0]), Xg, Yg)
    valid = []
    for j in range(0, ny - SUBDOMAIN_SIZE + 1, STEP):
        for i in range(0, nx - SUBDOMAIN_SIZE + 1, STEP):
            patch = test[j:j + SUBDOMAIN_SIZE, i:i + SUBDOMAIN_SIZE]
            if np.isnan(patch).mean() <= NAN_THRESHOLD:
                valid.append((j, i))
    print(f"Valid patches: {len(valid)} (NaN threshold: {NAN_THRESHOLD})")
    info = {
        "grid_shape": (ny, nx),
        "subdomain_size": SUBDOMAIN_SIZE,
        "step": STEP,
        "n_valid_patches": len(valid),
    }
    return valid, info


def compute_ssh_gradients(patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centered SSH gradient channels, both padded back to (H, W).

    a: x-direction  b: y-direction
    """
    a = np.diff(patch, axis=1)
    a = (a[1:] + a[:-1]) / 2.0
    a = np.pad(a, ((0, 1), (0, 1)), mode="edge")

    b = np.diff(patch, axis=0)
    b = (b[:, 1:] + b[:, :-1]) / 2.0
    b = np.pad(b, ((0, 1), (0, 1)), mode="edge")

    return a.astype(np.float32), b.astype(np.float32)


def interpolate_and_segment(
    var_3d: np.ndarray,
    delaunay: Delaunay,
    Xg: np.ndarray,
    Yg: np.ndarray,
    valid_patches: List[Tuple[int, int]],
    pbar_callback=None,
) -> Tuple[np.ndarray, List[dict]]:
    """Regrid every timestep and extract patches; return (N, H, W) array + metadata."""
    fi = FastInterpolator(delaunay)
    segments, metadata = [], []
    for t in range(var_3d.shape[0]):
        grid = fi(var_3d[t].flatten(), Xg, Yg)
        for idx, (j, i) in enumerate(valid_patches):
            patch = grid[j:j + SUBDOMAIN_SIZE, i:i + SUBDOMAIN_SIZE]
            if np.isnan(patch).any():
                patch = np.nan_to_num(patch, nan=0.0)
            segments.append(patch)
            metadata.append({
                "timestep": int(t),
                "patch_row": j // STEP,
                "patch_col": i // STEP,
                "patch_origin_j": int(j),
                "patch_origin_i": int(i),
                "spatial_patch_idx": idx,
            })
        if pbar_callback:
            pbar_callback(1)
    return np.array(segments, dtype=np.float32), metadata


def save_variable(
    segments: np.ndarray, metadata: List[dict], name: str, patch_grid_info: dict
) -> None:
    np.save(f"{OUTPUT_PREFIX}_{name}.npy", segments)
    meta = {
        "variable_name": name,
        "created_at": datetime.now().isoformat(),
        "config": {
            "subdomain_size": SUBDOMAIN_SIZE,
            "step": STEP,
            "grid_step": GRID_STEP,
            "nan_threshold": NAN_THRESHOLD,
            "edge_trim": EDGE_TRIM,
        },
        "patch_grid_info": patch_grid_info,
        "shape": list(segments.shape),
        "n_timesteps": len({m["timestep"] for m in metadata}),
        "n_spatial_patches": patch_grid_info["n_valid_patches"],
        "patches": metadata,
    }
    with open(f"{OUTPUT_PREFIX}_{name}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def create_gradient_stacked_output(
    all_segs: dict, other_vars: List[str], combo_name: str
) -> Optional[np.ndarray]:
    """Build (N, 7, 64, 64) stacked array with [a, b, *other_vars]."""
    missing = [v for v in other_vars + ["Eta"] if v not in all_segs]
    if missing:
        print(f"[WARN] Cannot create {combo_name}: missing {missing}")
        return None
    a_list, b_list = [], []
    for patch in tqdm(all_segs["Eta"], desc="SSH gradients", leave=False):
        a, b = compute_ssh_gradients(patch)
        a_list.append(a)
        b_list.append(b)
    a_arr = np.stack(a_list)[:, None]
    b_arr = np.stack(b_list)[:, None]
    rest  = np.stack([all_segs[v] for v in other_vars], axis=1)
    out   = np.concatenate([a_arr, b_arr, rest], axis=1)
    path  = f"{OUTPUT_PREFIX}_{combo_name}.npy"
    np.save(path, out)
    print(f"Saved {path}  shape={out.shape}")
    return out


# ── train/test split ──────────────────────────────────────────────────────────

def get_train_test_split(
    metadata: List[dict], test_fraction: float = 0.2
) -> Tuple[List[int], List[int], dict]:
    """Temporal block split — last test_fraction of timesteps held out."""
    timesteps = sorted({m["timestep"] for m in metadata})
    n_test = max(1, int(len(timesteps) * test_fraction))
    test_set = set(timesteps[-n_test:])
    train_idx = [i for i, m in enumerate(metadata) if m["timestep"] not in test_set]
    test_idx  = [i for i, m in enumerate(metadata) if m["timestep"] in test_set]
    info = {
        "method": "temporal_block",
        "test_fraction_actual": len(test_idx) / len(metadata),
        "n_train_samples": len(train_idx),
        "n_test_samples":  len(test_idx),
        "n_train_timesteps": len(timesteps) - n_test,
        "n_test_timesteps":  n_test,
    }
    return train_idx, test_idx, info


# ── main ──────────────────────────────────────────────────────────────────────

def process_variable(args, pbar=None):
    """Worker: load → trim → interpolate → save one variable."""
    name, source, delaunay, Xg, Yg, valid_patches, grid_info, \
        min_t, t0, t1 = args

    raw = np.load(os.path.join(MAIN_FOLDER, _combined_filename(source)))
    raw = raw.reshape(-1, *raw.shape[-2:]).astype(np.float32, copy=False)
    raw = raw[:min_t][t0:t1]

    def _cb(n):
        if pbar:
            with progress_lock:
                pbar.update(n)
                pbar.set_postfix_str(name)

    segs, meta = interpolate_and_segment(raw, delaunay, Xg, Yg, valid_patches, _cb)
    save_variable(segs, meta, name, grid_info)
    return name, segs, meta


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PREFIX) or ".", exist_ok=True)
    n_workers = mp.cpu_count() if N_WORKERS == -1 else max(0, N_WORKERS)
    print(f"Workers: {n_workers or 'sequential'}")

    combine_and_save_subfolders()

    print("\nLoading grid spacing...")
    dxc = np.load(os.path.join(MAIN_FOLDER, _combined_filename("dxc")))[0]
    dyc = np.load(os.path.join(MAIN_FOLDER, _combined_filename("dyc")))[0]

    delaunay, Xg, Yg = get_triangulation(dxc, dyc)
    valid_patches, grid_info = compute_valid_patch_mask(delaunay, Xg, Yg)
    if not valid_patches:
        raise RuntimeError("No valid patches found — check NAN_THRESHOLD or grid data.")

    # Find common timestep count across all variables
    shapes = {}
    for name, src in VAR_SOURCE_MAP.items():
        path = os.path.join(MAIN_FOLDER, _combined_filename(src))
        arr = np.load(path, mmap_mode="r")
        shapes[name] = arr.reshape(-1, *arr.shape[-2:]).shape[0]
        del arr
    min_t = min(shapes.values())
    t0 = EDGE_TRIM
    t1 = min_t - EDGE_TRIM
    if TEST_MAX_TIMESTEPS > 0:
        t1 = min(t1, t0 + TEST_MAX_TIMESTEPS)
    n_t = t1 - t0
    print(f"Timesteps per variable: min={min_t}, using [{t0}:{t1}] → {n_t}")

    all_segs: dict = {}
    all_meta: dict = {}

    args_list = [
        (name, VAR_SOURCE_MAP[name], delaunay, Xg, Yg,
         valid_patches, grid_info, min_t, t0, t1)
        for name in REQUIRED_VARS
    ]
    total = n_t * len(REQUIRED_VARS)

    if n_workers > 0:
        with tqdm(total=total, desc="Interpolating", unit="ts") as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(process_variable, a, pbar): a[0] for a in args_list}
                for fut in as_completed(futures):
                    name, segs, meta = fut.result()
                    all_segs[name] = segs
                    all_meta[name] = meta
                    tqdm.write(f"  ✓ {name}  shape={segs.shape}")
    else:
        with tqdm(total=total, desc="Interpolating", unit="ts") as pbar:
            for a in args_list:
                name, segs, meta = process_variable(a, pbar)
                all_segs[name] = segs
                all_meta[name] = meta

    # Validate alignment
    ref_shape = all_segs[REQUIRED_VARS[0]].shape
    for v in REQUIRED_VARS[1:]:
        assert all_segs[v].shape == ref_shape, f"Shape mismatch: {v}"
    print(f"✓ All variables aligned  shape={ref_shape}")

    # Build stacked gradient outputs
    deep = create_gradient_stacked_output(all_segs, COMBO_DEEP_VARS, "stacked_DEEP_GRAD")
    full = create_gradient_stacked_output(all_segs, COMBO_FULL_VARS, "stacked_FULL_GRAD")

    # Train/test split (temporal block, last 20 % of timesteps)
    ref_meta = all_meta[REQUIRED_VARS[0]]
    train_idx, test_idx, split_info = get_train_test_split(ref_meta, test_fraction=0.2)

    deep_ch = ["a", "b"] + COMBO_DEEP_VARS
    full_ch  = ["a", "b"] + COMBO_FULL_VARS

    master = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "main_folder": MAIN_FOLDER,
            "subdomain_size": SUBDOMAIN_SIZE,
            "step": STEP,
            "grid_step": GRID_STEP,
            "nan_threshold": NAN_THRESHOLD,
            "edge_trim": EDGE_TRIM,
            "output_prefix": OUTPUT_PREFIX,
            "ssh_representation": "centered_gradient_padded",
        },
        "processed_timesteps": n_t,
        "trim_range": [t0, t1],
        "patch_grid_info": grid_info,
        "variables": REQUIRED_VARS,
        "stacked_outputs": {
            "stacked_DEEP_GRAD": {
                "variables": deep_ch,
                "shape": list(deep.shape) if deep is not None else None,
            },
            "stacked_FULL_GRAD": {
                "variables": full_ch,
                "shape": list(full.shape) if full is not None else None,
            },
        },
        "train_test_split": split_info,
        "train_indices": train_idx,
        "test_indices":  test_idx,
    }
    # Saved under a name the data loader does not pick up automatically: the paper
    # runs used a random patch split. Rename to {prefix}_master_metadata.json to
    # train with this temporal split instead (see README).
    with open(f"{OUTPUT_PREFIX}_preprocess_metadata.json", "w") as f:
        json.dump(master, f, indent=2)

    # Raw-SSH stack used by the SSH-input models, the geostrophic baseline and figures
    deep_ssh = np.stack([all_segs[v] for v in ["Eta"] + COMBO_DEEP_VARS], axis=1)
    np.save(f"{OUTPUT_PREFIX}_stacked_DEEP.npy", deep_ssh)
    print(f"Saved {OUTPUT_PREFIX}_stacked_DEEP.npy  shape={deep_ssh.shape}")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"  DEEP: {OUTPUT_PREFIX}_stacked_DEEP_GRAD.npy  {deep_ch}")
    print(f"  FULL: {OUTPUT_PREFIX}_stacked_FULL_GRAD.npy  {full_ch}")
    print(f"  Train: {split_info['n_train_samples']}  Test: {split_info['n_test_samples']}")


if __name__ == "__main__":
    main()
