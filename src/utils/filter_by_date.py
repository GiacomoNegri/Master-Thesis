#!/usr/bin/env python3
"""
filter_by_date.py

Reads all CSVs from REF_FOLDER, drops rows before CUTOFF_DATE,
and saves the filtered files to OUT_DIR with updated date range in the filename.

Usage: edit REF_FOLDER, CUTOFF_DATE, OUT_DIR below, then run.
"""

import os
import re
import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────
REF_FOLDER  = "data/replication_returns_other"
CUTOFF_DATE = "01/01/2002"          # dd/mm/YYYY
OUT_DIR     = "data/replication_returns_other_post2001"
# ──────────────────────────────────────────────────────────────────────────────

cutoff = pd.to_datetime(CUTOFF_DATE, dayfirst=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Regex captures ticker and end-date from filenames like raw_AAPL_1980-12-12_2026-04-10
FNAME_RE = re.compile(r"(?:raw_)?([A-Z^.]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")

for fname in os.listdir(REF_FOLDER):
    m = FNAME_RE.match(fname)
    if not m:
        continue

    ticker, _start, end = m.groups()
    df = pd.read_csv(os.path.join(REF_FOLDER, fname))
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df[df["date"] >= cutoff]

    if df.empty:
        print(f"SKIP {fname}: no rows after cutoff")
        continue

    new_start = df["date"].min().strftime("%Y-%m-%d")
    out_name  = f"{ticker}_{new_start}_{end}.csv"
    df.to_csv(os.path.join(OUT_DIR, out_name), index=False)
    print(f"OK   {fname}  →  {out_name}  ({len(df)} rows)")
