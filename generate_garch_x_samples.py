"""
generate_garch_x_samples.py

Generate conditional Close samples using a fitted GARCH-X (log-GARCH with Student-t
innovations) model, given OHLC windows reconstructed from a CSDI checkpoint config.

Window loading replicates the exact split logic from generate_samples.py so that the
same windows are used for both the diffusion model and the GARCH-X baseline.

Q_var construction per window: [open^2, (high - low)^2], standardised with q_mu/q_sd
saved at GARCH-X fit time.  Innovations z_t are drawn i.i.d. from the fitted
standardised Student-t(nu) distribution (Var = 1).

Run:
    python generate_garch_x_samples.py \
        --checkpoint_folder ohlc_conditional \
        --checkpoint_name   MY.pt \
        --theta_path        garch_x_theta.npy \
        --q_stats_path      garch_x_q_stats.npy \
        --num_csv           16 \
        --num_samples       10 \
        --out_dir           ./data/generated/garch_x \
        --seed              42

Inputs
------
--theta_path   : .npy file containing result.x from fit_log_garch_x (1-D float array).
--q_stats_path : .npy file of shape (2, k_var) where row 0 = q_mu, row 1 = q_sd
                 (the same arrays returned by standardize_train_apply at fit time).

Output
------
<out_dir>/train_generated_close.csv
<out_dir>/val_generated_close.csv   (only if val windows exist)

CSV schema (identical to generate_samples.py generated_close files):
    window_idx | sample_idx | file | window_start | step_000 ... step_{L-1}
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

# Reuse GARCH-X internals; file must be importable from the same directory.
from garch_x_studentt_innovations_modified import unpack_params, simulate_bootstrap_series


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate GARCH-X conditional Close samples from OHLC windows."
    )
    # Checkpoint (for config / split reconstruction only — not for model weights)
    p.add_argument("--checkpoint_folder", type=str, required=True,
                   help="Sub-folder inside checkpoints/ that holds the .pt file.")
    p.add_argument("--checkpoint_name",   type=str, required=True,
                   help="Filename of the checkpoint (.pt). Used only to read config.")

    # GARCH-X fitted objects
    p.add_argument("--theta_path",    type=str, required=True,
                   help=".npy file with fitted theta vector (result.x).")
    p.add_argument("--q_stats_path",  type=str, required=True,
                   help=".npy file of shape (2, k_var): row 0 = q_mu, row 1 = q_sd.")

    # Generation volume
    p.add_argument("--num_csv",     type=int, required=True,
                   help="Number of windows to use from each split.")
    p.add_argument("--num_samples", type=int, default=10,
                   help="Number of Close paths to generate per window.")

    # Output
    p.add_argument("--out_dir", type=str, default="./data/generated/garch_x",
                   help="Directory where output CSVs are written.")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--date_format", type=str, default="%d/%m/%Y",
                   help="strptime format of the date column in the CSVs.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Split reconstruction (mirrors generate_samples.py / reconstruct_split)
# ---------------------------------------------------------------------------

def reconstruct_split(config, repo_root, date_format="%d/%m/%Y"):
    """
    Rebuild the exact train/val window indices from the checkpoint config.
    Identical logic to reconstruct_split in generate_samples.py.
    """
    seq_len      = int(config["train"]["seq_len"])
    stride       = int(config["train"]["stride"])
    seed         = int(config["train"]["seed"])
    data_root    = config["train"]["data_root"]
    columns      = tuple(config["data"].get("columns", ("date", "log_adj_close")))
    subset_size  = config["train"].get("train_subset_size", None)
    subset_ratio = config["train"].get("train_subset_ratio", None)
    val_ratio    = config["train"].get("val_split_ratio", None)

    date_col = columns[0]

    abs_root = os.path.normpath(os.path.join(repo_root, data_root))
    files    = sorted(glob.glob(os.path.join(abs_root, "*.csv")))
    assert len(files) > 0, f"No CSVs found in {abs_root}"
    print(f"Data root : {abs_root}  ({len(files)} files)")

    all_dates_set = set()
    file_lengths  = []
    for fp in files:
        df = pd.read_csv(fp, usecols=[date_col])
        all_dates_set.update(df[date_col].tolist())
        file_lengths.append(len(df))

    all_dates_sorted = sorted(
        all_dates_set,
        key=lambda s: pd.to_datetime(s, format=date_format),
    )
    date_to_idx = {d: i for i, d in enumerate(all_dates_sorted)}

    flat_index = []
    for fi, T in enumerate(file_lengths):
        max_start = T - seq_len
        if max_start < 0:
            continue
        for s in range(0, max_start + 1, stride):
            flat_index.append((fi, s))

    dataset_size = len(flat_index)
    print(f"Total windows: {dataset_size}")

    rng_torch   = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(dataset_size, generator=rng_torch).tolist()

    if val_ratio is not None and 0 < val_ratio < 1:
        val_size    = max(1, int(dataset_size * val_ratio))
        val_indices = all_indices[:val_size]
        train_pool  = all_indices[val_size:]
    else:
        val_indices = []
        train_pool  = all_indices

    if subset_size is not None:
        n_train = min(int(subset_size), len(train_pool))
    elif subset_ratio is not None:
        n_train = max(1, int(dataset_size * subset_ratio))
        n_train = min(n_train, len(train_pool))
    else:
        n_train = len(train_pool)

    train_indices = train_pool[:n_train]

    print(f"Train windows: {n_train}  |  Val windows: {len(val_indices)}")
    return files, flat_index, train_indices, val_indices, date_col, columns, seq_len


# ---------------------------------------------------------------------------
# Q_var construction from a single OHLC window
# ---------------------------------------------------------------------------

def build_Q_window(win_df, open_col, high_col, low_col, q_mu, q_sd):
    """
    Construct and standardise Q_var for one window.

    Parameters
    ----------
    win_df : DataFrame with columns open_col, high_col, low_col, shape (L, >=3)
    q_mu, q_sd : 1-D arrays of length k_var, from fit time

    Returns
    -------
    Q : np.ndarray of shape (L, k_var)

    Q_var columns follow the same order as construct_Q in
    garch_x_studentt_innovations_modified.py:
        col 0: gap2   = open^2
        col 1: range2 = (high - low)^2
    """
    o = win_df[open_col].to_numpy(dtype=float)
    h = win_df[high_col].to_numpy(dtype=float)
    l = win_df[low_col].to_numpy(dtype=float)

    gap2   = o ** 2
    range2 = (h - l) ** 2

    Q_raw = np.column_stack([gap2, range2])           # (L, 2)
    Q     = (Q_raw - q_mu) / q_sd                     # standardise with train stats
    return Q


# ---------------------------------------------------------------------------
# Single-window GARCH-X simulation
# ---------------------------------------------------------------------------

def simulate_one_window(theta, y_window, Q_window, num_samples, rng, k_mean=0, k_var=2):
    """
    Draw `num_samples` Close paths for one window using the GARCH-X recursion.

    Parameters
    ----------
    theta     : fitted parameter vector (result.x)
    y_window  : 1-D array of length L — raw close values for this window
    Q_window  : 2-D array (L, k_var) — standardised Q_var for this window
    num_samples : int
    rng       : np.random.Generator

    Returns
    -------
    samples : np.ndarray of shape (num_samples, L)

    Simulation details
    ------------------
    - z_t ~ standardised Student-t(nu), i.e. z_t = T_t * sqrt((nu-2)/nu)
      where T_t ~ t_nu.  This matches student_t_logpdf_standardized in the
      GARCH-X file (same reparametrisation).
    - simulate_bootstrap_series from the GARCH-X file accepts pre-drawn z_boot,
      so we just pass the drawn shocks directly.
    - Initial conditions per simulate_bootstrap_series: y_sim[0] = y_window[0],
      sig2[0] = var(y_window), matching the likelihood initialisation in
      neg_loglik_log_garch_x.
    """
    _, _, _, _, _, _, _, nu = unpack_params(theta, k_mean, k_var)

    # Scale factor so that z has unit variance (same as student_t_logpdf_standardized)
    s = np.sqrt((nu - 2.0) / nu)

    L = len(y_window)
    samples = np.zeros((num_samples, L), dtype=float)

    for i in range(num_samples):
        # Draw raw t_nu shocks then scale to unit variance
        T_draws  = rng.standard_t(df=nu, size=L)
        z_boot   = T_draws * s                        # standardised Student-t shocks

        y_sim = simulate_bootstrap_series(
            theta  = theta,
            y      = y_window,        # used only for length + initial value
            X_mean = None,
            Q_var  = Q_window,
            z_boot = z_boot,
        )
        samples[i] = y_sim

    return samples


# ---------------------------------------------------------------------------
# Generate + save for one split
# ---------------------------------------------------------------------------

def run_split(split_name, window_indices, flat_index, files,
              date_col, open_col, high_col, low_col, close_col,
              seq_len, theta, q_mu, q_sd, k_var,
              num_samples, rng, out_dir):

    step_cols  = [f"step_{t:03d}" for t in range(seq_len)]
    n_windows  = len(window_indices)
    rows       = []

    for w_idx, dataset_idx in enumerate(window_indices):
        fi, start = flat_index[dataset_idx]
        df_full   = pd.read_csv(files[fi])
        win_df    = df_full.iloc[start : start + seq_len].reset_index(drop=True)

        y_window = win_df[close_col].to_numpy(dtype=float)          # (L,)
        Q_window = build_Q_window(win_df, open_col, high_col, low_col, q_mu, q_sd)  # (L, k_var)

        samples = simulate_one_window(
            theta       = theta,
            y_window    = y_window,
            Q_window    = Q_window,
            num_samples = num_samples,
            rng         = rng,
            k_var       = k_var,
        )   # (num_samples, L)

        fname = os.path.basename(files[fi])
        print(f"  [{split_name}] window {w_idx} <- {fname}  (start={start})")
        for s_idx in range(num_samples):
            row = {
                "window_idx":   w_idx,
                "sample_idx":   s_idx,
                "file":         fname,
                "window_start": start,
            }
            row.update(zip(step_cols, samples[s_idx]))
            rows.append(row)


    df_out   = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"{split_name}_generated_close.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {split_name} -> {out_path}  "
          f"({n_windows} windows x {num_samples} samples = {len(df_out)} rows)")
    return df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # -- Load GARCH-X fitted objects -----------------------------------------
    theta   = np.load(args.theta_path)
    q_stats = np.load(args.q_stats_path)   # shape (2, k_var)
    q_mu    = q_stats[0]
    q_sd    = q_stats[1]
    k_var   = q_mu.shape[0]

    print(f"theta shape : {theta.shape}")
    print(f"k_var       : {k_var}  (inferred from q_stats)")

    # Unpack for a quick sanity print (k_mean=0 as in the reference script)
    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean=0, k_var=k_var)
    print(f"GARCH-X params  alpha={alpha:.4f}  beta={beta:.4f}  "
          f"alpha+beta={alpha+beta:.4f}  nu={nu:.4f}  delta={delta}")

    # -- Load checkpoint config (split reconstruction only) ------------------
    ckpt_path = os.path.join("checkpoints", args.checkpoint_folder, args.checkpoint_name)
    print(f"Loading checkpoint config from: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]

    # -- Reconstruct split ---------------------------------------------------
    repo_root = os.path.abspath(os.path.dirname(__file__))
    files, flat_index, train_indices, val_indices, date_col, columns, seq_len = \
        reconstruct_split(config, repo_root, date_format=args.date_format)

    # Identify column names; columns = (date_col, open, high, low, close) by convention
    feat_cols = list(columns[1:])
    assert "open"  in feat_cols, f"'open' not found in feat_cols: {feat_cols}"
    assert "high"  in feat_cols, f"'high' not found in feat_cols: {feat_cols}"
    assert "low"   in feat_cols, f"'low' not found in feat_cols: {feat_cols}"
    assert "close" in feat_cols, f"'close' not found in feat_cols: {feat_cols}"

    open_col  = "open"
    high_col  = "high"
    low_col   = "low"
    close_col = "close"

    # -- Select windows ------------------------------------------------------
    selected_train = train_indices[:min(args.num_csv, len(train_indices))]
    selected_val   = val_indices[:min(args.num_csv, len(val_indices))]

    print(f"\nGenerating: train={len(selected_train)} windows, "
          f"val={len(selected_val)} windows, {args.num_samples} samples each")

    # -- Generate ------------------------------------------------------------
    shared = dict(
        flat_index = flat_index,
        files      = files,
        date_col   = date_col,
        open_col   = open_col,
        high_col   = high_col,
        low_col    = low_col,
        close_col  = close_col,
        seq_len    = seq_len,
        theta      = theta,
        q_mu       = q_mu,
        q_sd       = q_sd,
        k_var      = k_var,
        num_samples = args.num_samples,
        rng        = rng,
        out_dir    = args.out_dir,
    )

    print("\n-- TRAIN split --")
    run_split("train", selected_train, **shared)

    if selected_val:
        print("\n-- VAL split --")
        run_split("val", selected_val, **shared)
    else:
        print("\nNo val windows — skipping val split.")

    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
