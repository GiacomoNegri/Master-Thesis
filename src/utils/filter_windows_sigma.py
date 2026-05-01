#!/usr/bin/env python3
"""
filter_windows_sigma.py

Keeps only those windows whose chosen statistic(s) fall within
±X standard deviations of the mean across all windows.

Usage example
-------------
python src/utils/filter_windows_sigma.py \
    --data_folder  data/replication_returns_other \
    --output_folder data/filtered_windows_sigma \
    --seq_length 2048 \
    --stride 400 \
    --sigma 2.0 \
    --stats variance kurtosis
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

AVAILABLE_STATS = ("variance", "kurtosis", "zero_ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep windows within ±X standard deviations of the mean "
            "for the chosen per-window statistic(s)."
        )
    )
    parser.add_argument("--data_folder",   required=True,
                        help="Folder with per-ticker CSV files (date, log_adj_close).")
    parser.add_argument("--output_folder", required=True,
                        help="Root folder where the filtered subset will be saved.")
    parser.add_argument("--seq_length",    type=int,   default=252)
    parser.add_argument("--stride",        type=int,   default=1)
    parser.add_argument(
        "--sigma", type=float, required=True,
        help="Number of standard deviations. Windows outside ±sigma are removed.",
    )
    parser.add_argument(
        "--stats", nargs="+", default=["variance"],
        choices=list(AVAILABLE_STATS),
        help=(
            "Which statistic(s) to apply the sigma filter to. "
            "A window is kept only if it passes ALL selected filters. "
            f"Choices: {AVAILABLE_STATS}. Default: variance"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers (shared logic with filter_windows.py)
# ---------------------------------------------------------------------------

def _excess_kurtosis(x: np.ndarray) -> float:
    sigma = float(x.std())
    if sigma == 0.0:
        return 0.0
    return float((((x - x.mean()) / sigma) ** 4).mean() - 3.0)


def collect_windows(dataset: SP500WindowDataset) -> list[dict]:
    records = []
    n = len(dataset)
    print(f"  Iterating over {n} windows ...", flush=True)
    for i in range(n):
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
        })
    return records


def add_stats(records: list[dict]) -> list[dict]:
    for r in records:
        v = r["values"]
        r["zero_ratio"] = float(np.mean(v == 0.0))
        r["variance"]   = float(np.var(v))
        r["kurtosis"]   = _excess_kurtosis(v)
    return records


def _build_date_cache(records: list[dict], data_folder: str) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for fname in {r["source_file"] for r in records}:
        fp = os.path.join(data_folder, fname)
        cache[fname] = pd.read_csv(fp, usecols=["date"])["date"].values
    return cache


def save_windows_as_csv(
    records: list[dict],
    data_folder: str,
    output_folder: str,
    subset_name: str,
) -> None:
    subset_dir = os.path.join(output_folder, subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    date_cache = _build_date_cache(records, data_folder)
    for r in records:
        source = r["source_file"]
        dates  = date_cache[source][r["start_idx"]:r["end_idx"]]
        ticker = os.path.splitext(source)[0]
        fname  = f"{ticker}_w{r['window_id']:07d}_s{r['start_idx']}.csv"
        pd.DataFrame({
            "date":          dates,
            "log_adj_close": r["values"],
        }).to_csv(os.path.join(subset_dir, fname), index=False)
    print(f"  [{subset_name}]  {len(records)} files  ->  {subset_dir}")


def save_per_ticker_concatenated(
    records: list[dict],
    data_folder: str,
    output_folder: str,
    subset_name: str,
) -> None:
    """
    For each ticker, take the union of all row indices that appear in any kept
    window, load the original full series, keep only those rows, and save a
    single CSV sorted by date descending.

    The row index in the output file is the position within the kept slice of
    the original series, so it can be used directly as an absolute time
    embedding index — unlike individual per-window CSVs whose indices always
    reset to 0.
    """
    out_dir = os.path.join(output_folder, subset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Group records by ticker
    by_ticker: dict[str, list[dict]] = {}
    for r in records:
        by_ticker.setdefault(r["source_file"], []).append(r)

    for fname, ticker_records in sorted(by_ticker.items()):
        ticker = os.path.splitext(fname)[0]

        # Union of all row indices covered by kept windows (no duplicates)
        kept_indices: set[int] = set()
        for r in ticker_records:
            kept_indices.update(range(r["start_idx"], r["end_idx"]))

        # Load the original series for this ticker
        src_path = os.path.join(data_folder, fname)
        full_df  = pd.read_csv(src_path)   # columns: date, log_adj_close

        # Retain only rows that belong to at least one kept window
        filtered = full_df.iloc[sorted(kept_indices)].copy()

        # Sort descending by date (most recent first)
        # filtered = filtered.sort_values("date", ascending=False).reset_index(drop=True)
        # after
        filtered["date"] = pd.to_datetime(filtered["date"], format="%d/%m/%Y")
        filtered = filtered.sort_values("date", ascending=False).reset_index(drop=True)
        filtered["date"] = filtered["date"].dt.strftime("%d/%m/%Y")  # preserve original format

        out_path = os.path.join(out_dir, f"{ticker}_concatenated.csv")
        filtered.to_csv(out_path, index=False)

    n_tickers = len(by_ticker)
    print(f"  [{subset_name}]  {n_tickers} ticker file(s)  ->  {out_dir}")


# ---------------------------------------------------------------------------
# Sigma filter
# ---------------------------------------------------------------------------

def sigma_filter(
    records: list[dict],
    stats: list[str],
    n_sigma: float,
) -> tuple[list[dict], dict]:
    """
    Return (kept_records, info_dict).
    A window is kept only if, for EVERY stat in `stats`,
    |value - mean| <= n_sigma * std.
    """
    info: dict = {}
    mask = np.ones(len(records), dtype=bool)

    for stat in stats:
        values = np.array([r[stat] for r in records])
        mu     = float(values.mean())
        sigma  = float(values.std())
        lo     = mu - n_sigma * sigma
        hi     = mu + n_sigma * sigma
        stat_mask = (values >= lo) & (values <= hi)
        n_removed = int((~stat_mask).sum())
        info[stat] = {"mean": mu, "std": sigma, "lo": lo, "hi": hi, "removed": n_removed}
        mask &= stat_mask

    kept = [r for r, ok in zip(records, mask) if ok]
    return kept, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Window Sigma-Filter Script")
    print("=" * 60)
    print(f"  data_folder   : {args.data_folder}")
    print(f"  output_folder : {args.output_folder}")
    print(f"  seq_length    : {args.seq_length}")
    print(f"  stride        : {args.stride}")
    print(f"  sigma         : {args.sigma}")
    print(f"  stats         : {args.stats}")
    print()

    print("[1/5] Initialising SP500WindowDataset ...")
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

    print("\n[2/5] Collecting windows ...")
    records = collect_windows(dataset)

    print("\n[3/5] Computing per-window statistics ...")
    records = add_stats(records)

    print("\n[4/5] Applying sigma filter ...")
    kept, info = sigma_filter(records, args.stats, args.sigma)

    total  = len(records)
    n_kept = len(kept)
    print()
    for stat, d in info.items():
        print(f"  {stat}:")
        print(f"    mean = {d['mean']:.8f},  std = {d['std']:.8f}")
        print(f"    keep range: [{d['lo']:.8f},  {d['hi']:.8f}]")
        print(f"    removed by this filter: {d['removed']} "
              f"({d['removed'] / total * 100:.1f}%)")
    print()
    print(f"  Total windows : {total}")
    print(f"  Kept          : {n_kept}  ({n_kept / total * 100:.1f}%)")
    print(f"  Removed       : {total - n_kept}  ({(total - n_kept) / total * 100:.1f}%)")

    stats_tag  = "_".join(args.stats)
    sigma_tag  = str(args.sigma).replace(".", "p")
    subset_name = f"sigma{sigma_tag}_{stats_tag}"

    print(f"\n[5/5] Saving per-ticker concatenated files as '{subset_name}' ...")
    save_per_ticker_concatenated(kept, args.data_folder, args.output_folder, subset_name)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
