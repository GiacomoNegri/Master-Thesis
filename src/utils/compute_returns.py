#!/usr/bin/env python3
"""
Reads CSVs from ref_folder, computes log returns for the price column,
and saves results to out_folder.

Usage:
    python src/utils/compute_returns.py data/replication data/replication_returns
    python src/utils/compute_returns.py data/replication data/replication_returns --cols date adj_close
"""

import os
import argparse

import numpy as np
import pandas as pd


def compute_returns(ref_folder: str, out_folder: str, cols: list[str]) -> None:
    date_col, price_col = cols[0], cols[1]
    os.makedirs(out_folder, exist_ok=True)

    files = sorted(f for f in os.listdir(ref_folder) if f.endswith(".csv"))
    for fname in files:
        df = pd.read_csv(os.path.join(ref_folder, fname))
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
            df = df.sort_values(date_col).reset_index(drop=True)
            df[date_col] = df[date_col].dt.strftime("%d/%m/%Y")
        df["log_adj_close"] = np.log(df[price_col] / df[price_col].shift(1))
        df = df.dropna(subset=["log_adj_close"])
        df[[date_col, "log_adj_close"]].to_csv(os.path.join(out_folder, fname), index=False)
        print(f"{fname}: {len(df)} rows -> {out_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_folder")
    parser.add_argument("out_folder")
    parser.add_argument("--cols", nargs=2, default=["date", "log_adj_close"],
                        metavar=("DATE_COL", "PRICE_COL"))
    args = parser.parse_args()
    compute_returns(args.ref_folder, args.out_folder, args.cols)