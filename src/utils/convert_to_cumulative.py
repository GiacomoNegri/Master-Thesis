"""
Convert log-return CSVs to cumulative log-return CSVs.

Usage:
    python src/utils/convert_to_cumulative.py <ref_folder> <out_folder>

Each input CSV must have columns: date (%d/%m/%Y), log_adj_close.
The cumulative sum starts from index 0 (no zero prepended): row count is
preserved and the date column is left untouched.
"""
import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_folder", type=str, help="Folder containing input CSVs")
    parser.add_argument("out_folder", type=str, help="Folder to write converted CSVs")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.ref_folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {args.ref_folder}")

    for fp in files:
        df = pd.read_csv(fp)
        df["log_adj_close"] = df["log_adj_close"].cumsum()
        out_path = os.path.join(args.out_folder, os.path.basename(fp))
        df.to_csv(out_path, index=False)
        print(f"Converted: {os.path.basename(fp)}")


if __name__ == "__main__":
    main()
