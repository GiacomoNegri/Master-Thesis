"""
Convert raw OHLC CSVs to log-return representation.

For each CSV in --ref_folder:
  - close  -> log return: log(C_t / C_{t-1})
  - open, high, low -> log ratio with contemporaneous close: log(X_t / C_t)
  - first row is dropped (no prior close)
  - output saved to --out_folder with the same filename

Usage:
    python -m src.utils.ohlc_to_returns --ref_folder ./data/SNP500_individual --out_folder ./data/SNP500_returns
"""

import argparse
import os

import numpy as np
import pandas as pd

COLS = ["date", "open", "high", "low", "close"]


def process_file(src: str, dst: str) -> None:
    df = pd.read_csv(src, usecols=COLS, parse_dates=["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"].to_numpy(dtype=np.float64)
    log_ret = np.log(close[1:] / close[:-1])

    out = pd.DataFrame()
    out["date"] = df["date"].iloc[1:].values
    out["close"] = log_ret
    for col in ["open", "high", "low"]:
        out[col] = np.log(df[col].to_numpy(dtype=np.float64)[1:] / close[1:])

    out.to_csv(dst, index=False)


def main(ref_folder: str, out_folder: str) -> None:
    os.makedirs(out_folder, exist_ok=True)
    csv_files = sorted(f for f in os.listdir(ref_folder) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{ref_folder}'.")

    for fname in csv_files:
        src = os.path.join(ref_folder, fname)
        dst = os.path.join(out_folder, fname)
        try:
            process_file(src, dst)
            print(f"OK  {fname}")
        except Exception as e:
            print(f"SKIP {fname}: {e}")

    print(f"\nDone — {len(csv_files)} files processed → '{out_folder}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_folder", required=True)
    parser.add_argument("--out_folder", required=True)
    args = parser.parse_args()
    main(args.ref_folder, args.out_folder)
