#!/usr/bin/env python3
"""
filter_by_ref_tickers.py

Retains files from --to_filter whose ticker appears in --ref_folder,
filters rows to dates <= --cutoff_date, and saves to --out_dir.

Filename formats:
  ref_folder  : raw_TICKER_YYYY-MM-DD_YYYY-MM-DD.csv  (ticker is 2nd field)
  to_filter   : TICKER_YYYY-MM-DD_YYYY-MM-DD.csv      (ticker is 1st field)

Usage:
  python src/utils/filter_by_ref_tickers.py \
      --ref_folder data/replication \
      --to_filter  data/SNP500_individual_normalized \
      --out_dir    data/SNP500_individual_normalized_filtered \
      --cutoff_date 10/04/2026
python src/utils/filter_by_ref_tickers.py --ref_folder data/replication_returns_other --to_filter data/SNP500_individual_normalized --out_dir data/SNP500_individual_normalized_replication --cutoff_date 10/04/2026
"""

import argparse
import os
import re
import pandas as pd

REF_RE    = re.compile(r"(?:raw_)?([A-Z^.]+)_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.csv$")
FILTER_RE = re.compile(r"([A-Z^.]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")


def collect_ref_tickers(ref_folder: str) -> set:
    tickers = set()
    for fname in os.listdir(ref_folder):
        m = REF_RE.match(fname)
        if m:
            tickers.add(m.group(1))
    return tickers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_folder",   required=True, help="Folder whose filenames define the allowed tickers")
    parser.add_argument("--to_filter",    required=True, help="Folder with files to filter")
    parser.add_argument("--out_dir",      required=True, help="Output folder")
    parser.add_argument("--cutoff_date",  required=True, help="Keep rows <= this date (dd/mm/YYYY)")
    args = parser.parse_args()

    cutoff = pd.to_datetime(args.cutoff_date, dayfirst=True)
    os.makedirs(args.out_dir, exist_ok=True)

    ref_tickers = collect_ref_tickers(args.ref_folder)
    print(f"Reference tickers found: {len(ref_tickers)}")

    saved, skipped = 0, 0
    for fname in sorted(os.listdir(args.to_filter)):
        m = FILTER_RE.match(fname)
        if not m:
            continue

        ticker, _start, end = m.groups()
        if ticker not in ref_tickers:
            print(f"SKIP {fname}: ticker '{ticker}' not in ref_folder")
            skipped += 1
            continue

        df = pd.read_csv(os.path.join(args.to_filter, fname))
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df = df[df["date"] <= cutoff]

        if df.empty:
            print(f"SKIP {fname}: no rows after cutoff")
            skipped += 1
            continue

        new_start = df["date"].min().strftime("%Y-%m-%d")
        new_end   = df["date"].max().strftime("%Y-%m-%d")
        out_name  = f"{ticker}_{new_start}_{new_end}.csv"
        df.to_csv(os.path.join(args.out_dir, out_name), index=False, date_format="%d/%m/%Y")
        print(f"OK   {fname}  →  {out_name}  ({len(df)} rows)")
        saved += 1

    print(f"\nDone — saved {saved}, skipped {skipped}")


if __name__ == "__main__":
    main()
