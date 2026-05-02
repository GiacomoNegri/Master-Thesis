#!/usr/bin/env python3
"""
top_n_difficult_windows.py

Selects the N most "difficult" windows from all possible sliding windows,
where difficulty = sum of z-scores of: std, excess kurtosis, max absolute return.

Each kept window is saved as an individual CSV with its original dates.
Overlapping windows will share the same date values for overlapping rows — this is
intentional and correct: we record actual calendar dates, not relative positions.

Usage example
-------------
python src/utils/top_n_difficult_windows.py \
    --data_folder  data/replication_returns_other \
    --output_folder data/top_difficult_windows \
    --seq_length 2048 \
    --stride 400 \
    --n_windows 500
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.dataloader import SP500WindowDataset  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the N windows with the highest difficulty score."
    )
    parser.add_argument("--data_folder",   required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--seq_length",    type=int, default=252)
    parser.add_argument("--stride",        type=int, default=1)
    parser.add_argument("--n_windows",     type=int, required=True,
                        help="Number of top-difficulty windows to keep.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-window statistics
# ---------------------------------------------------------------------------

def _excess_kurtosis(x: np.ndarray) -> float:
    s = float(x.std())
    if s == 0.0:
        return 0.0
    return float((((x - x.mean()) / s) ** 4).mean() - 3.0)


def collect_windows(dataset: SP500WindowDataset) -> list[dict]:
    """Iterate the dataset once, storing values and metadata for every window."""
    records = []
    for i in range(len(dataset)):
        sample  = dataset[i]
        data_np = sample["observed_data"].numpy()
        values  = data_np[0] if data_np.shape[0] == 1 else data_np.flatten()
        meta    = sample["meta"]
        start   = int(meta["start"])
        records.append({
            "source_file": meta["file"],
            "window_id":   i,
            "start_idx":   start,
            "end_idx":     start + dataset.seq_len,
            "values":      values,
            # raw stats computed once here, used below
            "stat_std":    float(values.std()),
            "stat_kurt":   _excess_kurtosis(values),
            "stat_maxabs": float(np.max(np.abs(values))),
        })
    return records


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def add_difficulty(records: list[dict]) -> list[dict]:
    """
    For each of the three statistics, compute the z-score across all windows,
    then sum the three z-scores into a single difficulty score.

    A window with high std, heavy tails, and large single-period swings will
    have a high difficulty score.
    """
    for stat in ("stat_std", "stat_kurt", "stat_maxabs"):
        vals = np.array([r[stat] for r in records], dtype=float)
        mu, sigma = vals.mean(), vals.std()
        # If all windows share the same value, z-scores are 0 — safe to divide.
        z = (vals - mu) / sigma if sigma > 0 else np.zeros_like(vals)
        for r, zi in zip(records, z):
            r[f"z_{stat}"] = float(zi)

    for r in records:
        r["difficulty"] = r["z_stat_std"] + r["z_stat_kurt"] + r["z_stat_maxabs"]

    return records


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _build_date_cache(records: list[dict], data_folder: str) -> dict[str, np.ndarray]:
    """Load the date column once per unique source file."""
    cache: dict[str, np.ndarray] = {}
    for fname in {r["source_file"] for r in records}:
        fp = os.path.join(data_folder, fname)
        cache[fname] = pd.read_csv(fp, usecols=["date"])["date"].values
    return cache


def save_windows(records: list[dict], data_folder: str, output_folder: str) -> None:
    """
    Write one CSV per window.  The date column carries the original calendar
    dates from the source file — two overlapping windows will have the same
    dates for their shared rows, which is the correct behaviour.
    """
    os.makedirs(output_folder, exist_ok=True)
    date_cache = _build_date_cache(records, data_folder)

    for rank, r in enumerate(records):  # records already sorted by difficulty desc
        dates  = date_cache[r["source_file"]][r["start_idx"]:r["end_idx"]]
        ticker = os.path.splitext(r["source_file"])[0]
        fname  = f"rank{rank:05d}_{ticker}_w{r['window_id']:07d}_s{r['start_idx']}.csv"
        pd.DataFrame({
            "date":          dates,
            "log_adj_close": r["values"],
        }).to_csv(os.path.join(output_folder, fname), index=False)

    print(f"  Saved {len(records)} window files -> {output_folder}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Top-N Difficult Windows")
    print("=" * 60)
    print(f"  data_folder   : {args.data_folder}")
    print(f"  output_folder : {args.output_folder}")
    print(f"  seq_length    : {args.seq_length}")
    print(f"  stride        : {args.stride}")
    print(f"  n_windows     : {args.n_windows}")
    print()

    print("[1/4] Initialising SP500WindowDataset ...")
    dataset = SP500WindowDataset(
        root_dir=args.data_folder,
        seq_len=args.seq_length,
        stride=args.stride,
        columns=("date", "log_adj_close"),
        time_mode="index_norm",
        cache_data=True,
        drop_incomplete=True,
    )
    print(f"  Files found   : {len(dataset.files)}")
    print(f"  Total windows : {len(dataset)}")

    if args.n_windows > len(dataset):
        raise ValueError(
            f"--n_windows ({args.n_windows}) exceeds total windows ({len(dataset)})."
        )

    print("\n[2/4] Collecting windows and computing raw statistics ...")
    records = collect_windows(dataset)

    print("\n[3/4] Computing z-score difficulty scores ...")
    records = add_difficulty(records)

    # Sort descending; take top N
    records.sort(key=lambda r: r["difficulty"], reverse=True)
    top_n = records[: args.n_windows]

    scores = [r["difficulty"] for r in top_n]
    print(f"  Difficulty range (top {args.n_windows}): "
          f"[{min(scores):.4f}, {max(scores):.4f}]")

    print(f"\n[4/4] Saving top {args.n_windows} windows ...")
    save_windows(top_n, args.data_folder, args.output_folder)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()