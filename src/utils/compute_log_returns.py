import os
import argparse

import numpy as np
import pandas as pd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_log_returns(folder_name: str, out_folder: str, global_normalization: bool = False) -> None:
    os.makedirs(out_folder, exist_ok=True)

    files = sorted(f for f in os.listdir(folder_name) if f.endswith(".csv"))

    # First pass: compute log returns for every file and keep them in memory
    file_data: dict[str, pd.DataFrame] = {}
    for filename in files:
        filepath = os.path.join(folder_name, filename)
        df = pd.read_csv(filepath, parse_dates=["date"])
        df["log_returns"] = df["log_adj_close"].diff()
        df = df[["date", "log_returns"]].dropna()
        file_data[filename] = df

    # Compute and persist the global std when requested
    global_std = 1.0
    if global_normalization:
        all_values = np.concatenate([df["log_returns"].values for df in file_data.values()])
        global_std = float(np.std(all_values, ddof=0))
        print(f"Global std of log returns across all stocks: {global_std:.8f}")

        std_path = os.path.join(out_folder, "global_std.txt")
        with open(std_path, "w") as fh:
            fh.write(f"{global_std}\n")
        print(f"Saved global std -> {std_path}  (needed for de-normalisation at generation time)")

    # Second pass: optionally normalise and write CSVs
    for filename, df in file_data.items():
        if global_normalization:
            df = df.copy()
            df["log_returns"] = df["log_returns"] / global_std

        out_path = os.path.join(out_folder, filename)
        df.to_csv(out_path, index=False)
        print(f"Saved {filename} ({len(df)} rows) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute log returns from log prices.")
    parser.add_argument("folder_name", help="Input folder containing CSVs with log_adj_close")
    parser.add_argument("out_folder", help="Output folder for CSVs with log_returns")
    parser.add_argument(
        "--global_normalization",
        type=str2bool,
        default=False,
        help=(
            "If true, divide every log return by the pooled std computed across ALL stocks "
            "before saving. The scalar is written to <out_folder>/global_std.txt so it can "
            "be applied in reverse at generation time. Default: false."
        ),
    )
    args = parser.parse_args()

    compute_log_returns(args.folder_name, args.out_folder, args.global_normalization)

# Usage examples:
# python compute_log_returns.py data/replication data/replication_returns
# python compute_log_returns.py data/replication data/replication_returns --global_normalization true