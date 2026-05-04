"""
extract_windows.py

Extract train/val windows from the SP500WindowDataset using the same
splitting logic as csdi_train.py, and save each window as a CSV with dates.

Usage:
    python extract_windows.py \
        --data_root ./data/replication \
        --out_dir ./data/extracted_windows \
        --seq_len 252 \
        --stride 5 \
        --seed 42 \
        --train_subset_size 1000 \
        --val_split_ratio 0.1
python -m src.utils.extract_windows --data_root ./data/replication_returns_other --out_dir ./data/extracted_windows --seq_len 2048 --stride 400 --seed 50 --train_subset_size 1
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


from src.utils.dataloader import SP500WindowDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_subset_size", type=int, default=None)
    parser.add_argument("--val_split_ratio", type=float, default=None)
    return parser.parse_args()


def save_window(dataset: SP500WindowDataset, idx: int, out_path: str) -> None:
    file_idx, start = dataset.index[idx]
    fp = dataset.files[file_idx]

    df = pd.read_csv(fp)
    window_df = df.iloc[start : start + dataset.seq_len].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    window_df.to_csv(out_path, index=False)


def build_window_filename(dataset: SP500WindowDataset, idx: int) -> str:
    file_idx, start = dataset.index[idx]
    stem = os.path.splitext(os.path.basename(dataset.files[file_idx]))[0]
    return f"{stem}_start{start}.csv"


def main():
    args = parse_args()

    dataset = SP500WindowDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        stride=args.stride,
        cache_data=False,
        drop_incomplete=True,
    )
    dataset_size = len(dataset)
    print(f"Total windows in dataset: {dataset_size}")

    # Same permutation as csdi_train.py
    rng = torch.Generator().manual_seed(args.seed)
    all_indices = torch.randperm(dataset_size, generator=rng).tolist()

    # Carve out validation first
    val_indices = []
    train_pool = all_indices
    if args.val_split_ratio is not None:
        if not (0 < args.val_split_ratio < 1):
            raise ValueError("val_split_ratio must be in (0, 1)")
        val_size = max(1, int(dataset_size * args.val_split_ratio))
        val_indices = all_indices[:val_size]
        train_pool = all_indices[val_size:]
        print(f"Validation set: {val_size} windows")

    # Carve out train subset
    if args.train_subset_size is not None:
        if args.train_subset_size <= 0:
            raise ValueError("train_subset_size must be > 0")
        train_indices = train_pool[: min(args.train_subset_size, len(train_pool))]
    else:
        train_indices = train_pool
    print(f"Train set: {len(train_indices)} windows")

    # Save train windows
    train_dir = os.path.join(args.out_dir, "train")
    for i, idx in enumerate(train_indices):
        fname = build_window_filename(dataset, idx)
        out_path = os.path.join(train_dir, fname)
        save_window(dataset, idx, out_path)
        if (i + 1) % 500 == 0 or (i + 1) == len(train_indices):
            print(f"  train: {i + 1}/{len(train_indices)}")

    # Save val windows
    if val_indices:
        val_dir = os.path.join(args.out_dir, "val")
        for i, idx in enumerate(val_indices):
            fname = build_window_filename(dataset, idx)
            out_path = os.path.join(val_dir, fname)
            save_window(dataset, idx, out_path)
            if (i + 1) % 500 == 0 or (i + 1) == len(val_indices):
                print(f"  val:   {i + 1}/{len(val_indices)}")

    print(f"Done. Windows saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
