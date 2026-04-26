"""
Window-wise normalization of OHLC CSV files.

For each CSV in <ref_folder>:
  1. Slice into windows of length seq_len (step = stride). When stride < seq_len,
     windows overlap and each window is saved separately.
  2. If --compute_returns: compute log returns for close column only (log(close_t/close_t-1)).
     Other columns are normalized using close mean/std (computed before returns).
     The close column is never normalized when returns are computed.
  3. Otherwise, normalize each window independently according to --norm_type:
       'local' : per-column z-score  x = (x - col_mean) / (col_std + eps)
       'close' : use the mean/std of the 'close' column (within each window) to normalise
                 ALL columns (open, high, low, close).
       'global_close' : use the global mean/std of the 'close' column (across entire file)
                        to normalise ALL columns.
  4. Write each window as a separate CSV with naming:
     {original_name}_w{window_id:07d}_s{start_idx}.csv

Usage (CLI):
    python -m src.utils.windowwise_normalize \\
        --ref_folder ./data/fake_fts \\
        --seq_len 2048 \\
        --stride 400 \\
        --cols open high low close \\
        --norm_type local \\
        --compute_returns  # optional
"""

import argparse
import os

import numpy as np
import pandas as pd


def _compute_log_returns(arr: np.ndarray) -> np.ndarray:
    """Compute log returns: log(X_t / X_t-1). Returns shape (n_rows-1, n_features)."""
    return np.log(arr[1:] / arr[:-1])


def normalize_folder(
    ref_folder: str,
    seq_len: int = 64,
    stride: int = 64,
    cols: list[str] | None = None,
    date_col: str = "date",
    eps: float = 1e-8,
    norm_type: str = "local",
    compute_returns: bool = False,
) -> None:
    if cols is None:
        cols = ["open", "high", "low", "close"]

    if norm_type not in ("local", "close", "global_close"):
        raise ValueError(f"norm_type must be 'local', 'close', or 'global_close', got '{norm_type}'.")

    if norm_type in ("close", "global_close") and "close" not in cols:
        raise ValueError(f"norm_type='{norm_type}' requires 'close' to be in cols.")

    if stride > seq_len:
        raise ValueError(f"stride ({stride}) cannot be greater than seq_len ({seq_len}).")

    suffix = "_processed_close" if norm_type == "close" else "_processed"
    out_dir = ref_folder.rstrip("/\\") + suffix
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(ref_folder) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{ref_folder}'.")

    close_idx = cols.index("close")

    # Precompute global close stats if needed
    if norm_type == "global_close":
        global_close_mean_all = []
        global_close_std_all = []
        for fname in csv_files:
            df = pd.read_csv(os.path.join(ref_folder, fname))
            arr = df[cols].to_numpy(dtype=np.float32)
            global_close_mean_all.append(arr[:, close_idx].mean())
            global_close_std_all.append(arr[:, close_idx].std())
        global_close_mean = np.mean(global_close_mean_all)
        global_close_std = np.mean(global_close_std_all)

    n_processed = 0
    for fname in csv_files:
        df = pd.read_csv(os.path.join(ref_folder, fname))

        arr = df[cols].to_numpy(dtype=np.float32)   # (n_rows, n_features)
        dates = df[date_col].values
        n_rows, n_features = arr.shape

        if n_rows < seq_len:
            print(f"Skipping {fname}: only {n_rows} rows, need {seq_len}.")
            continue

        n_windows = (n_rows - seq_len) // stride + 1
        base_name = os.path.splitext(fname)[0]

        for window_id in range(n_windows):
            start_idx = window_id * stride
            end_idx = start_idx + seq_len

            window_arr = arr[start_idx:end_idx]  # (seq_len, n_features)
            window_dates = dates[start_idx:end_idx]
            window_norm = window_arr.copy()

            # Compute log returns for close column if needed
            if compute_returns:
                close_returns = np.log(window_arr[1:, close_idx] / window_arr[:-1, close_idx])
                window_arr_for_norm = window_arr[1:]  # align with returns (exclude first row)
                window_dates_for_norm = window_dates[1:]
            else:
                window_arr_for_norm = window_arr
                window_dates_for_norm = window_dates

            # Normalize based on norm_type

            if norm_type == "local":
                # per-column z-score
                mean = window_arr_for_norm.mean(axis=0, keepdims=True)
                std = window_arr_for_norm.std(axis=0, keepdims=True)
                window_norm = (window_arr_for_norm - mean) / (std + eps)

            elif norm_type == "close":
                # Use close column's mean/std for normalizing other columns
                close_vals = window_arr[:, close_idx]
                close_mean = close_vals.mean()
                close_std = close_vals.std()
                window_norm = (window_arr_for_norm - close_mean) / (close_std + eps)

            elif norm_type == "global_close":
                # Use global close stats for normalization
                window_norm = (window_arr_for_norm - global_close_mean) / (global_close_std + eps)

            # If compute_returns, replace close with log returns and exclude the first row
            if compute_returns:
                window_norm = window_norm[:-1] if len(window_norm) > len(close_returns) else window_norm
                window_norm[:, close_idx] = close_returns
                window_dates = window_dates_for_norm

            # Save window as individual CSV
            df_window = pd.DataFrame(window_norm, columns=cols)
            df_window.insert(0, date_col, window_dates)

            window_fname = f"{base_name}_w{window_id:07d}_s{start_idx}.csv"
            df_window.to_csv(os.path.join(out_dir, window_fname), index=False)
            n_processed += 1

    print(
        f"Done — {n_processed} windows saved from {len(csv_files)} files.\n"
        f"  seq_len={seq_len}, stride={stride}, norm_type={norm_type}, compute_returns={compute_returns}\n"
        f"  columns={cols}\n"
        f"  Output: '{out_dir}'"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Window-wise column-wise normalization of OHLC CSVs."
    )
    parser.add_argument("--ref_folder", required=True, help="Folder containing raw CSV files.")
    parser.add_argument("--seq_len", type=int, default=64, help="Window length (default: 64).")
    parser.add_argument("--stride", type=int, default=64, help="Window stride (default: 64).")
    parser.add_argument(
        "--cols",
        nargs="+",
        default=["open", "high", "low", "close"],
        help="Feature columns to normalize (default: open high low close).",
    )
    parser.add_argument("--date_col", default="date", help="Name of the date column (default: date).")
    parser.add_argument("--eps", type=float, default=1e-8, help="Stability epsilon (default: 1e-8).")
    parser.add_argument(
        "--norm_type",
        choices=["local", "close", "global_close"],
        default="local",
        help="Normalization type: 'local' (per-column z-score), 'close' (columns normalised by window close mean/std), or 'global_close' (columns normalised by file-wide close mean/std). Default: local.",
    )
    parser.add_argument(
        "--compute_returns",
        action="store_true",
        help="If set, compute log returns for the close column: log(close_t / close_t-1). Other columns are normalized using close stats but close is never normalized.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    normalize_folder(
        ref_folder=args.ref_folder,
        seq_len=args.seq_len,
        stride=args.stride,
        cols=args.cols,
        date_col=args.date_col,
        eps=args.eps,
        norm_type=args.norm_type,
        compute_returns=args.compute_returns,
    )

#python -m src.utils.windowwise_normalize --ref_folder ./data/fake_fts --seq_len 64 --stride 64 --cols open high low close --norm_type close