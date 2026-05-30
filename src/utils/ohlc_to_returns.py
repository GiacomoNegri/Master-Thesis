"""
Convert raw OHLC CSVs to log-return representation.

For each CSV in --ref_folder:
  - close  -> log return: log(C_t / C_{t-1})
  - open, high, low -> log ratio with prior close: log(X_t / C_{t-1})
  - first row is dropped (no prior close)
  - output saved to --out_folder with the same filename

Usage:
    python -m src.utils.ohlc_to_returns --ref_folder ./data/SNP500_individual --out_folder ./data/SNP500_individual_processed
    python -m src.utils.ohlc_to_returns --ref_folder ./data/SNP500_individual --out_folder ./data/SNP500_individual_normalized --col_normalization
"""

import argparse
import os

import numpy as np
import pandas as pd

COLS = ["date", "open", "high", "low", "close"]


def process_file(src: str, dst: str) -> None:
    df = pd.read_csv(src, usecols=COLS, parse_dates=["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=COLS[1:])  # drop rows with any NaN OHLC price
    df = df[df["close"] > 0].reset_index(drop=True)  # drop zero/negative close

    close = df["close"].to_numpy(dtype=np.float64)
    log_ret = np.log(close[1:] / close[:-1])

    out = pd.DataFrame()
    out["date"] = df["date"].iloc[1:].dt.strftime("%d/%m/%Y")
    out["close"] = log_ret
    for col in ["open", "high", "low"]:
        out[col] = np.log(df[col].to_numpy(dtype=np.float64)[1:] / close[:-1])

    out.to_csv(dst, index=False)


def main(ref_folder: str, out_folder: str, col_normalization: bool = False) -> None:
    os.makedirs(out_folder, exist_ok=True)
    csv_files = sorted(f for f in os.listdir(ref_folder) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{ref_folder}'.")

    if not col_normalization:
        for fname in csv_files:
            src = os.path.join(ref_folder, fname)
            dst = os.path.join(out_folder, fname)
            try:
                process_file(src, dst)
                print(f"OK  {fname}")
            except Exception as e:
                print(f"SKIP {fname}: {e}")
    else:
        VALUE_COLS = ["close", "open", "high", "low"]

        # Pass 1: build all transformed DataFrames
        frames, names = {}, []
        for fname in csv_files:
            src = os.path.join(ref_folder, fname)
            try:
                df = pd.read_csv(src, usecols=COLS, parse_dates=["date"], dayfirst=True)
                df = df.sort_values("date").reset_index(drop=True)
                df = df.dropna(subset=COLS[1:])
                df = df[df["close"] > 0].reset_index(drop=True)
                close = df["close"].to_numpy(dtype=np.float64)
                out = pd.DataFrame()
                out["date"] = df["date"].iloc[1:].dt.strftime("%d/%m/%Y")
                out["close"] = np.log(close[1:] / close[:-1])
                for col in ["open", "high", "low"]:
                    out[col] = np.log(df[col].to_numpy(dtype=np.float64)[1:] / close[:-1])
                frames[fname] = out
                names.append(fname)
                print(f"OK  {fname}")
            except Exception as e:
                print(f"SKIP {fname}: {e}")

        # Compute global mean and std per column (axis=0) across all files
        all_data = pd.concat([frames[f][VALUE_COLS] for f in names], ignore_index=True)
        mean = all_data.mean(axis=0)
        std = all_data.std(axis=0)

        print("\nGlobal column means:")
        for col in mean.index:
            print(f"  {col}: {mean[col]:.6f}")
        print("\nGlobal column stds:")
        for col in std.index:
            print(f"  {col}: {std[col]:.6f}")
        
        # Save stats for later unnormalization
        stats = pd.DataFrame({"mean": mean, "std": std})
        stats.to_csv(os.path.join(out_folder, "normalization_stats.csv"))
        print(f"Saved normalization stats → '{out_folder}/normalization_stats.csv'")

        # Pass 2: z-score each file and write
        for fname in names:
            out = frames[fname].copy()
            out[VALUE_COLS] = (out[VALUE_COLS] - mean) / std
            out.to_csv(os.path.join(out_folder, fname), index=False)

    print(f"\nDone — {len(csv_files)} files processed → '{out_folder}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_folder", required=True)
    parser.add_argument("--out_folder", required=True)
    parser.add_argument("--col_normalization", action="store_true", default=False)
    args = parser.parse_args()
    main(args.ref_folder, args.out_folder, args.col_normalization)
