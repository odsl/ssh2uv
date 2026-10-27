#!/usr/bin/env python3
"""Write a temporal train / validation / test split for the stacked patches.

Samples are stored as index = t * 6 + p (hour t, subdomain p). The record is cut
into three consecutive blocks of hours, separated by gaps so that hourly
autocorrelation cannot leak across the boundaries:

    train  [0, t1)      validation  [t1 + gap, t2)      test  [t2 + gap, T)

The training loader reads ``train_indices`` and ``test_indices`` (used as the
validation set for learning-rate scheduling and checkpoint selection) from
``<out_dir>/global-final-64_master_metadata.json``; ``heldout_test_indices`` is
only used for the final evaluation. ``<out_dir>`` also receives links to the data
files, so that the random-split pipeline can be trained on this split as well.

Usage:
    python scripts/make_temporal_split.py --data_dir data --out_dir data/temporal_split
"""

import argparse
import json
import os

import numpy as np

from ssh2uv.kinematics import N_PATCHES

DATA_FILES = [
    "global-final-64_stacked_DEEP_GRAD.npy",
    "global-final-64_stacked_DEEP.npy",
    "global-final-64_stacked_FULL.npy",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="data/temporal_split")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--gap_hours", type=int, default=48)
    return p.parse_args()


def main():
    args = parse_args()
    n = np.load(os.path.join(args.data_dir, DATA_FILES[0]), mmap_mode="r").shape[0]
    n_hours = n // N_PATCHES
    t1 = int(args.train_frac * n_hours)
    t2 = int((args.train_frac + args.val_frac) * n_hours)
    blocks = {
        "train": (0, t1),
        "val": (t1 + args.gap_hours, t2),
        "test": (t2 + args.gap_hours, n_hours),
    }
    idx = {k: [t * N_PATCHES + p for t in range(a, b) for p in range(N_PATCHES)] for k, (a, b) in blocks.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    for f in DATA_FILES:
        src = os.path.relpath(os.path.join(args.data_dir, f), args.out_dir)
        dst = os.path.join(args.out_dir, f)
        if os.path.exists(os.path.join(args.data_dir, f)) and not os.path.lexists(dst):
            os.symlink(src, dst)

    meta = {
        "split": {
            "method": "temporal_blocks",
            "n_hours": n_hours,
            "gap_hours": args.gap_hours,
            "hours": {k: list(v) for k, v in blocks.items()},
            "n_samples": {k: len(v) for k, v in idx.items()},
            "note": "test_indices = validation block (used for LR schedule / checkpoint selection); "
                    "heldout_test_indices = final test block",
        },
        "train_indices": idx["train"],
        "test_indices": idx["val"],
        "heldout_test_indices": idx["test"],
    }
    path = os.path.join(args.out_dir, "global-final-64_master_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f)
    print(json.dumps(meta["split"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
