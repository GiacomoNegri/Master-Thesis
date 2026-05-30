#!/usr/bin/env python3
"""
filter_by_date_range.py

Reads all CSVs from REF_FOLDER, keeps only rows within [START_DATE, END_DATE]
(inclusive), and saves the filtered files to OUT_DIR with updated date range
in the filename.

Usage: edit REF_FOLDER, START_DATE, END_DATE, OUT_DIR below, then run.
"""

import os
import re
import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────
REF_FOLDER = "data/SNP500_individual"
START_DATE = "01/01/2002"        # dd/mm/YYYY  (inclusive)
END_DATE   = "10/04/2026"           # dd/mm/YYYY  (inclusive)
OUT_DIR    = "data/SNP500_individual_post2002"
# ──────────────────────────────────────────────────────────────────────────────

start = pd.to_datetime(START_DATE, dayfirst=True)
end   = pd.to_datetime(END_DATE,   dayfirst=True)

if start > end:
    raise ValueError(f"START_DATE ({START_DATE}) must be <= END_DATE ({END_DATE})")

os.makedirs(OUT_DIR, exist_ok=True)

# Regex captures ticker and date bounds from filenames like raw_AAPL_1980-12-12_2026-04-10
FNAME_RE = re.compile(r"(?:raw_)?([A-Z^.]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")

for fname in os.listdir(REF_FOLDER):
    m = FNAME_RE.match(fname)
    if not m:
        continue

    ticker, _orig_start, _orig_end = m.groups()
    df = pd.read_csv(os.path.join(REF_FOLDER, fname))
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    if df.empty:
        print(f"SKIP {fname}: no rows in range")
        continue

    new_start = df["date"].min().strftime("%Y-%m-%d")
    new_end   = df["date"].max().strftime("%Y-%m-%d")
    out_name  = f"{ticker}_{new_start}_{new_end}.csv"
    df.to_csv(os.path.join(OUT_DIR, out_name), index=False, date_format="%d/%m/%Y")
    print(f"OK   {fname}  →  {out_name}  ({len(df)} rows)")
