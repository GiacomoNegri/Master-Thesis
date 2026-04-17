#!/usr/bin/env python3
"""
filter_windows.py

Creates filtered subsets of time-series windows from a folder of CSV files
containing log-returns.  Uses the project's SP500WindowDataset so that the
extracted windows are identical to those seen during training.

Usage example
-------------
python src/utils/filter_windows.py \
    --data_folder  data/sp500_individual_gbm \
    --output_folder data/filtered_windows \
    --seq_length 252 \
    --stride 1 \
    --percentage_zero 0.10

 2048 --stride 400 --percentage_zero 0.05

"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# The script lives at src/utils/, so we go two levels up to reach the
# project root, which makes "from src.utils.dataloader import ..." work
# regardless of where the script is called from.
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))   # …/src/utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR)) # …/project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.dataloader import SP500WindowDataset  # noqa: E402  (import after sys.path patch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter S&P 500 log-return windows into labelled subsets."
    )
    parser.add_argument(
        "--data_folder", required=True,
        help="Folder that contains per-ticker CSV files with columns: date, log_adj_close",
    )
    parser.add_argument(
        "--output_folder", required=True,
        help="Root folder where subset sub-folders will be created.",
    )
    parser.add_argument(
        "--seq_length", type=int, default=252,
        help="Window length (number of time steps). Default: 252",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Step between consecutive window start indices. Default: 1",
    )
    parser.add_argument(
        "--percentage_zero", type=float, default=0.10,
        help=(
            "Dual-purpose zero threshold (0–1).\n"
            "  low_zero : keep windows with zero_ratio < percentage_zero\n"
            "  high_zero: keep top <percentage_zero> fraction of windows by zero_ratio\n"
            "Default: 0.10"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def collect_windows(dataset: SP500WindowDataset) -> list[dict]:
    """
    Iterate over every window in the dataset and return a list of records.

    SP500WindowDataset.__getitem__ returns:
        observed_data  : torch.Tensor  shape (K, L)  – K features, L time steps
        observed_mask  : torch.Tensor  shape (K, L)  – all-ones mask
        observed_tp    : torch.Tensor  shape (L,)    – normalised time positions
        meta           : dict          keys: 'file' (basename), 'start' (int)

    Since columns=("date", "log_adj_close") the feature dimension is K=1, so
    we squeeze it to obtain a plain 1-D numpy array of length seq_len.
    """
    records = []
    n = len(dataset)
    print(f"  Iterating over {n} windows ...", flush=True)

    for i in range(n):
        sample = dataset[i]

        # observed_data shape: (K, L).  K=1 for single-feature (log_adj_close).
        data_np = sample["observed_data"].numpy()       # (K, L)
        if data_np.shape[0] == 1:
            values = data_np[0]                         # (L,)
        else:
            # Defensive fallback for unexpected multi-feature data
            values = data_np.flatten()

        meta = sample["meta"]
        start_idx = int(meta["start"])
        end_idx   = start_idx + dataset.seq_len

        records.append({
            "source_file": meta["file"],   # basename, e.g. "AAPL.csv"
            "window_id":   i,
            "start_idx":   start_idx,
            "end_idx":     end_idx,
            "values":      values,         # np.ndarray shape (seq_len,)
        })

    return records


def add_stats(records: list[dict]) -> list[dict]:
    """Augment each record with zero_ratio and variance."""
    for r in records:
        vals = r["values"]
        r["zero_ratio"] = float(np.mean(vals == 0.0))
        r["variance"]   = float(np.var(vals))
    return records


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _build_date_cache(records: list[dict], data_folder: str) -> dict[str, np.ndarray]:
    """
    Pre-load the 'date' column for every source file that appears in records.
    Caches as {basename -> np.ndarray of date strings}.
    """
    cache: dict[str, np.ndarray] = {}
    needed_files = {r["source_file"] for r in records}
    for fname in needed_files:
        fp = os.path.join(data_folder, fname)
        cache[fname] = pd.read_csv(fp, usecols=["date"])["date"].values
    return cache


def save_windows_as_csv(
    records: list[dict],
    data_folder: str,
    output_folder: str,
    subset_name: str,
) -> None:
    """
    Save each window as an individual CSV with columns ['date', 'log_adj_close']
    under  output_folder / subset_name /.

    The 'date' column is taken directly from the source CSV so it matches the
    original calendar exactly.
    """
    subset_dir = os.path.join(output_folder, subset_name)
    os.makedirs(subset_dir, exist_ok=True)

    date_cache = _build_date_cache(records, data_folder)

    for r in records:
        source  = r["source_file"]
        dates   = date_cache[source][r["start_idx"]:r["end_idx"]]
        ticker  = os.path.splitext(source)[0]

        # File name encodes ticker, window id, and start position for traceability
        fname = f"{ticker}_w{r['window_id']:07d}_s{r['start_idx']}.csv"

        pd.DataFrame({
            "date":          dates,
            "log_adj_close": r["values"],
        }).to_csv(os.path.join(subset_dir, fname), index=False)

    print(f"  [{subset_name}]  {len(records)} files  ->  {subset_dir}")


def save_general_table(records: list[dict], seq_len: int) -> None:
    """
    Save a single wide CSV to ./data/general/windows_metadata.csv.

    Columns:
        source_file, window_id, start_idx, end_idx, zero_ratio, variance,
        v_0, v_1, ..., v_{seq_len-1}

    This table allows every saved window CSV to be traced back to its
    position in the original dataset.
    """
    general_dir = os.path.join("data", "general")
    os.makedirs(general_dir, exist_ok=True)
    out_path = os.path.join(general_dir, "windows_metadata.csv")

    rows = []
    for r in records:
        row: dict = {
            "source_file": r["source_file"],
            "window_id":   r["window_id"],
            "start_idx":   r["start_idx"],
            "end_idx":     r["end_idx"],
            "zero_ratio":  r["zero_ratio"],
            "variance":    r["variance"],
        }
        for j, v in enumerate(r["values"]):
            row[f"v_{j}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  General metadata table: {len(df)} rows × {len(df.columns)} cols  ->  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Window Filter Script")
    print("=" * 60)
    print(f"  data_folder     : {args.data_folder}")
    print(f"  output_folder   : {args.output_folder}")
    print(f"  seq_length      : {args.seq_length}")
    print(f"  stride          : {args.stride}")
    print(f"  percentage_zero : {args.percentage_zero}")
    print()

    # ------------------------------------------------------------------
    # 1. Build dataset using the project's SP500WindowDataset
    # ------------------------------------------------------------------
    print("[1/5] Initialising SP500WindowDataset ...")
    dataset = SP500WindowDataset(
        root_dir=args.data_folder,
        seq_len=args.seq_length,
        stride=args.stride,
        columns=("date", "log_adj_close"),  # date column used for length only
        time_mode="index_norm",
        cache_data=True,       # keep all CSV data in RAM for fast random access
        drop_incomplete=True,  # discard windows shorter than seq_len
    )
    print(f"  Files found : {len(dataset.files)}")
    print(f"  Total windows: {len(dataset)}")

    # ------------------------------------------------------------------
    # 2. Collect every window into a list of plain dicts
    # ------------------------------------------------------------------
    print("\n[2/5] Collecting windows ...")
    records = collect_windows(dataset)

    # ------------------------------------------------------------------
    # 3. Compute per-window statistics
    # ------------------------------------------------------------------
    print("\n[3/5] Computing statistics (zero_ratio, variance) ...")
    records = add_stats(records)

    variances   = np.array([r["variance"]   for r in records])
    zero_ratios = np.array([r["zero_ratio"] for r in records])

    # ------------------------------------------------------------------
    # 4. Derive filtering thresholds
    # ------------------------------------------------------------------
    var_p40  = float(np.percentile(variances, 40))
    var_p60  = float(np.percentile(variances, 60))
    var_p90  = float(np.percentile(variances, 90))

    # high_zero: top <percentage_zero> fraction → percentile = (1 - pct) * 100
    zero_high_pct_rank   = (1.0 - args.percentage_zero) * 100.0
    zero_high_threshold  = float(np.percentile(zero_ratios, zero_high_pct_rank))

    print()
    print(f"  Variance   – 40th pct  : {var_p40:.8f}")
    print(f"  Variance   – 60th pct  : {var_p60:.8f}")
    print(f"  Variance   – 90th pct  : {var_p90:.8f}")
    print(f"  Zero-ratio – high threshold (top {args.percentage_zero*100:.1f}%): {zero_high_threshold:.8f}")

    # ------------------------------------------------------------------
    # 5. Apply filters
    # ------------------------------------------------------------------
    print("\n[4/5] Filtering subsets ...")

    low_zero     = [r for r in records if r["zero_ratio"] < args.percentage_zero]
    high_zero    = [r for r in records if r["zero_ratio"] >= zero_high_threshold]
    moderate_var = [r for r in records if var_p40 <= r["variance"] <= var_p60]
    high_var     = [r for r in records if r["variance"] >= var_p90]

    total = len(records)
    print(f"  Total windows            : {total}")
    print(f"  low_zero_windows         : {len(low_zero):>7}  ({len(low_zero)/total*100:.1f}%)")
    print(f"  high_zero_windows        : {len(high_zero):>7}  ({len(high_zero)/total*100:.1f}%)")
    print(f"  moderate_variance_windows: {len(moderate_var):>7}  ({len(moderate_var)/total*100:.1f}%)")
    print(f"  high_variance_windows    : {len(high_var):>7}  ({len(high_var)/total*100:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Save individual window CSVs
    # ------------------------------------------------------------------
    print("\n[5/5] Saving output files ...")
    save_windows_as_csv(low_zero,     args.data_folder, args.output_folder, "low_zero_windows")
    save_windows_as_csv(high_zero,    args.data_folder, args.output_folder, "high_zero_windows")
    save_windows_as_csv(moderate_var, args.data_folder, args.output_folder, "moderate_variance_windows")
    save_windows_as_csv(high_var,     args.data_folder, args.output_folder, "high_variance_windows")

    # ------------------------------------------------------------------
    # 7. Save the general traceability table
    # ------------------------------------------------------------------
    save_general_table(records, args.seq_length)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
