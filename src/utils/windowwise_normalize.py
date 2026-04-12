"""
Window-wise normalization of OHLC CSV files.

For each CSV in <ref_folder>:
  1. Slice into non-overlapping windows of length seq_len (step = stride).
  2. Normalize each window independently according to --norm_type:
       'local' : per-column z-score  x = (x - col_mean) / (col_std + eps)
       'close' : use the mean/std of the 'close' column to normalise
                 ALL columns (open, high, low, close).
  3. Write the result to <ref_folder>_processed/ (local) or
     <ref_folder>_processed_close/ (close) under the same filename.

NOTE: the reshape trick (windows -> flat array) only produces a contiguous layout
when stride == seq_len.  When stride < seq_len windows overlap and cannot be
flattened without losing data; the script raises an error in that case.

Usage (CLI):
    python -m src.utils.windowwise_normalize \\
        --ref_folder ./data/fake_fts \\
        --seq_len 64 \\
        --stride 64 \\
        --cols open high low close \\
        --norm_type local   # or close
"""

import argparse
import os

import numpy as np
import pandas as pd


def normalize_folder(
    ref_folder: str,
    seq_len: int = 64,
    stride: int = 64,
    cols: list[str] | None = None,
    date_col: str = "date",
    eps: float = 1e-8,
    norm_type: str = "local",
) -> None:
    if cols is None:
        cols = ["open", "high", "low", "close"]

    if norm_type not in ("local", "close"):
        raise ValueError(f"norm_type must be 'local' or 'close', got '{norm_type}'.")

    if norm_type == "close" and "close" not in cols:
        raise ValueError("norm_type='close' requires 'close' to be in cols.")

    if stride != seq_len:
        raise ValueError(
            f"stride ({stride}) != seq_len ({seq_len}). "
            "Overlapping windows cannot be flattened back to a CSV. "
            "Set stride == seq_len for non-overlapping windows."
        )

    suffix = "_processed_close" if norm_type == "close" else "_processed"
    out_dir = ref_folder.rstrip("/\\") + suffix
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(ref_folder) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{ref_folder}'.")

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
        usable_rows = (n_windows - 1) * stride + seq_len

        arr_trimmed = arr[:usable_rows]
        dates_trimmed = dates[:usable_rows]

        # (n_windows, seq_len, n_features)
        windows = arr_trimmed.reshape(n_windows, seq_len, n_features)

        # per-window normalization
        if norm_type == "local":
            # per-column z-score
            mean = windows.mean(axis=1, keepdims=True)  # (n_windows, 1, n_features)
            std = windows.std(axis=1, keepdims=True)
        else:  # norm_type == "close"
            # use close column's mean/std for all columns
            close_idx = cols.index("close")
            close_vals = windows[:, :, close_idx]  # (n_windows, seq_len)
            mean = close_vals.mean(axis=1, keepdims=True)[:, :, np.newaxis]  # (n_windows, 1, 1)
            std = close_vals.std(axis=1, keepdims=True)[:, :, np.newaxis]
        windows_norm = (windows - mean) / (std + eps)

        arr_norm = windows_norm.reshape(usable_rows, n_features)

        df_out = pd.DataFrame(arr_norm, columns=cols)
        df_out.insert(0, date_col, dates_trimmed)
        df_out.to_csv(os.path.join(out_dir, fname), index=False)
        n_processed += 1

    print(
        f"Done — {n_processed}/{len(csv_files)} files processed.\n"
        f"  seq_len={seq_len}, stride={stride}, norm_type={norm_type}, columns={cols}\n"
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
        choices=["local", "close"],
        default="local",
        help="Normalization type: 'local' (per-column z-score) or 'close' (all columns normalised by close mean/std). Default: local.",
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
    )

#python -m src.utils.windowwise_normalize --ref_folder ./data/fake_fts --seq_len 64 --stride 64 --cols open high low close --norm_type close