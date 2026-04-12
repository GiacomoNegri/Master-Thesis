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


def build_out_folder(folder_name: str, compute_returns: bool, normalization: str) -> str:
    parent = os.path.dirname(os.path.abspath(folder_name))
    returns_part = "returns_" if compute_returns else "prices_"
    if normalization == "global":
        norm_part = "global"
    elif normalization == "window":
        norm_part = "window"
    else:
        norm_part = "other"
    return os.path.join(parent, f"replication_{returns_part}{norm_part}")


def process_files(
    folder_name: str,
    compute_returns: bool = False,
    normalization: str = "none",
    seq_len: int = 24,
) -> None:
    out_folder = build_out_folder(folder_name, compute_returns, normalization)
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output folder: {out_folder}")

    files = sorted(f for f in os.listdir(folder_name) if f.endswith(".csv"))

    # First pass: load and optionally compute returns
    file_data: dict[str, pd.DataFrame] = {}
    for filename in files:
        filepath = os.path.join(folder_name, filename)
        df = pd.read_csv(filepath, parse_dates=["date"])
        if compute_returns:
            df["log_adj_close"] = df["log_adj_close"].diff()
            df = df.dropna(subset=["log_adj_close"])
        file_data[filename] = df

    # Normalization
    if normalization == "global":
        all_values = np.concatenate([df["log_adj_close"].values for df in file_data.values()])
        global_mean = float(np.mean(all_values))
        global_std = float(np.std(all_values, ddof=0))
        print(f"Global mean: {global_mean:.8f}  std: {global_std:.8f}")

        stats_path = os.path.join(out_folder, "global_stats.txt")
        with open(stats_path, "w") as fh:
            fh.write(f"mean={global_mean}\nstd={global_std}\n")
        print(f"Saved global stats -> {stats_path}")

        for filename, df in file_data.items():
            df = df.copy()
            df["log_adj_close"] = (df["log_adj_close"] - global_mean) / global_std
            file_data[filename] = df

    elif normalization == "window":
        for filename, df in file_data.items():
            df = df.copy()
            values = df["log_adj_close"].values.copy()
            n_windows = len(values) // seq_len
            window_means = []
            window_stds = []

            for i in range(n_windows):
                s, e = i * seq_len, (i + 1) * seq_len
                w_mean = float(np.mean(values[s:e]))
                w_std = float(np.std(values[s:e], ddof=0))
                if w_std == 0:
                    w_std = 1.0
                values[s:e] = (values[s:e] - w_mean) / w_std
                window_means.append(w_mean)
                window_stds.append(w_std)

            # Tail rows that don't fill a full window are left unnormalized
            df["log_adj_close"] = values
            file_data[filename] = df

            # Save window means and stds for de-normalization
            stats_path = os.path.join(out_folder, filename.replace(".csv", "_window_stats.txt"))
            with open(stats_path, "w") as fh:
                fh.write("mean,std\n")
                for m, s in zip(window_means, window_stds):
                    fh.write(f"{m},{s}\n")

    # Write output CSVs
    for filename, df in file_data.items():
        out_path = os.path.join(out_folder, filename)
        df.to_csv(out_path, index=False)
        print(f"Saved {filename} ({len(df)} rows) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log_adj_close: optionally compute returns and normalize.")
    parser.add_argument("folder_name", help="Input folder containing CSVs with log_adj_close")
    parser.add_argument(
        "--compute_returns",
        type=str2bool,
        default=False,
        help="If true, convert log_adj_close to log returns via differencing. Default: false.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["global", "window", "none"],
        help=(
            "'global': z-score using pooled mean/std across all stocks (saves global_stats.txt). "
            "'window': z-score each non-overlapping window of --seq_len rows by its own mean/std "
            "(saves per-file *_window_stats.txt). "
            "'none': no normalization. Default: none."
        ),
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=24,
        help="Window length for window-wise normalization. Only used when --normalization window. Default: 24.",
    )
    args = parser.parse_args()

    process_files(
        folder_name=args.folder_name,
        compute_returns=args.compute_returns,
        normalization=args.normalization,
        seq_len=args.seq_len,
    )

# Usage examples:
# python src.utils.processing_log_adj_close.py data/replication
# python src.utils.processing_log_adj_close.py data/replication --compute_returns true --normalization global
# python src.utils.processing_log_adj_close.py data/replication --compute_returns true --normalization window --seq_len 24
# python src.utils.processing_log_adj_close.py data/replication --compute_returns false --normalization none
