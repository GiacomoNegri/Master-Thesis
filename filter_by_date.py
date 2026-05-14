import argparse
import pandas as pd
from pathlib import Path

EXCLUDE_TICKERS = {"HUBB", "BEN", "NVR", "WMB"}
MIN_YEARS = 38


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_folder", required=True)
    parser.add_argument("--cutoff_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff_date)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped, saved = 0, 0
    for csv_path in sorted(Path(args.ref_folder).glob("*.csv")):
        ticker = csv_path.stem.upper()

        if ticker in EXCLUDE_TICKERS:
            print(f"  SKIP {ticker}: excluded ticker")
            skipped += 1
            continue

        df = pd.read_csv(csv_path)
        if "date" not in df.columns:
            print(f"  SKIP {ticker}: no 'date' column (columns: {list(df.columns)})")
            skipped += 1
            continue
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df = df[df["date"] <= cutoff]

        if df.empty:
            print(f"  SKIP {ticker}: no rows after filtering")
            skipped += 1
            continue

        earliest_required = cutoff - pd.DateOffset(years=MIN_YEARS)
        if df["date"].min() > earliest_required:
            print(f"  SKIP {ticker}: earliest record {df['date'].min().date()} is after {earliest_required.date()}")
            skipped += 1
            continue

        df.to_csv(out_dir / csv_path.name, index=False, date_format="%d/%m/%Y")
        saved += 1
        print(f"  SAVE {ticker}: {len(df)} rows")

    print(f"\nDone — saved {saved}, skipped {skipped}")


if __name__ == "__main__":
    main()

#python filter_by_date.py --ref_folder data/SNP500_individual_normalized --cutoff_date 2026-04-10 --out_dir data/SNP500_individual_normalized_filtered