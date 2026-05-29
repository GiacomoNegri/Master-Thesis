"""
garch_simulate_windows.py

Load pre-fitted GARCH-X parameters (from fit_garch_from_csv.py) and
simulate num_samples independent close paths per sliding window over each
ticker CSV. Output format matches the EDM-generated CSV so results can
be compared directly.

For each window:
  - y[0] is fixed to the actual first close value (shared across all samples)
  - y[1..T-1] are drawn auto-regressively from the GARCH-X model using only
    open/high/low as conditioning (Q_var = [open^2, (high-low)^2])
  - GARCH recursion is initialised with the full-series sample variance

Output columns: window_idx, sample_idx, file, window_start, step_000, ..., step_{T-1}

Usage:
    python garch_simulate_windows.py \
        --input_folder ./data/fake_fts_processed \
        --garch_dir    ./checkpoints/garch \
        --out_dir      ./data/generated/garch \
        --seq_len 512 \
        --stride  100 \
        --num_samples 250 \
        [--seed 42]
python garch_simulate_windows.py --input_folder ./data/SNP500_individual_processed --garch_dir ./checkpoints/garch --out_dir ./data/generated/garch --seq_len 512 --stride 100 --num_samples 10 --seed 42
python garch_simulate_windows.py --input_folder ./data/SNP500_individual_processed_replication --garch_dir ./checkpoints/garch_replication --out_dir ./data/generated/garch_replication --seq_len 512 --stride 100 --num_samples 10 --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameter unpacking (identical to fit_garch_from_csv.py)
# ---------------------------------------------------------------------------

def unpack_params(theta, k_mean, k_var):
    idx = 0
    mu    = theta[idx]; idx += 1
    phi   = theta[idx]; idx += 1
    gamma = theta[idx:idx + k_mean]; idx += k_mean
    omega = theta[idx]; idx += 1
    raw_alpha = theta[idx]; idx += 1
    raw_beta  = theta[idx]; idx += 1
    exp_a = np.exp(raw_alpha)
    exp_b = np.exp(raw_beta)
    denom = 1.0 + exp_a + exp_b
    alpha = exp_a / denom
    beta  = exp_b / denom
    delta = theta[idx:idx + k_var]; idx += k_var
    nu    = 2.01 + np.exp(theta[idx])
    return mu, phi, gamma, omega, alpha, beta, delta, nu


# ---------------------------------------------------------------------------
# Per-window simulation
# ---------------------------------------------------------------------------

INNOVATION_CLIP = 5.0  # clip innovations at ±5σ_t (bounded-influence simulation)


def simulate_window(y_window, Q_var_window, theta, num_samples, rng, init_log_sig2):
    """
    Generate num_samples close paths for one window.

    y[0] (actual first close) seeds the AR recursion for all samples.
    y[1..T-1] are drawn: y[t] = mu + phi*y[t-1] + sigma[t]*z[t],
    where z[t] ~ standardised Student-t(nu) and sigma[t] follows the
    log-GARCH-X recursion driven by Q_var_window (open/high/low data).

    Innovation clipping: eps[t] is capped at ±INNOVATION_CLIP*sigma[t] before
    being fed back into the variance recursion. This prevents a single extreme
    draw from permanently destabilising log_sig2 over subsequent steps (bounded-
    influence GARCH, Muler & Yohai 2008). The clip is applied to both the path
    value and the lagged residual; INNOVATION_CLIP=5.0 is more permissive than
    any historically observed single-day equity return (~11σ on Black Monday).

    Returns array of shape (num_samples, T).
    """
    mu, phi, _, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean=0, k_var=2)
    T          = len(y_window)
    c          = 1e-8
    std_scale  = np.sqrt((nu - 2.0) / nu)  # rescale t_nu draw to unit variance

    paths = np.empty((num_samples, T), dtype=float)
    paths[:, 0] = y_window[0]

    for s in range(num_samples):
        eps_prev      = y_window[0] - mu  # residual at t=0 (no prior lag)
        log_sig2_prev = init_log_sig2

        for t in range(1, T):
            mean_t       = mu + phi * paths[s, t - 1]
            log_sig2_t   = (omega
                            + alpha * np.log(eps_prev ** 2 + c)
                            + beta  * log_sig2_prev
                            + Q_var_window[t] @ delta)
            log_sig2_t   = np.clip(log_sig2_t, -30.0, 30.0)
            sigma_t      = np.sqrt(np.exp(log_sig2_t))
            eps_t        = sigma_t * rng.standard_t(nu) * std_scale
            eps_t        = np.clip(eps_t, -INNOVATION_CLIP * sigma_t,
                                          +INNOVATION_CLIP * sigma_t)
            paths[s, t]  = mean_t + eps_t
            eps_prev      = eps_t        # clipped residual feeds the variance recursion
            log_sig2_prev = log_sig2_t

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate GARCH-X close paths per window; output mirrors EDM CSV format."
    )
    p.add_argument("--input_folder", required=True,
                   help="Folder with CSVs (columns: date, close, open, high, low).")
    p.add_argument("--garch_dir", required=True,
                   help="Folder with <stem>_theta.npy files (output of fit_garch_from_csv.py).")
    p.add_argument("--out_dir", required=True,
                   help="Output folder for generated_close.csv.")
    p.add_argument("--seq_len", type=int, required=True,
                   help="Window length (number of time steps).")
    p.add_argument("--stride", type=int, required=True,
                   help="Stride between consecutive windows.")
    p.add_argument("--num_samples", type=int, default=250,
                   help="Number of independent paths to generate per window.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng       = np.random.default_rng(args.seed)
    step_cols = [f"step_{t:03d}" for t in range(args.seq_len)]
    out_path  = os.path.join(args.out_dir, "generated_close.csv")

    csv_files = sorted(f for f in os.listdir(args.input_folder) if f.endswith(".csv"))
    print(f"Found {len(csv_files)} CSV files in {args.input_folder}")

    header_written = False
    total_windows  = 0

    for fname in csv_files:
        stem       = os.path.splitext(fname)[0]
        theta_path = os.path.join(args.garch_dir, f"{stem}_theta.npy")

        if not os.path.exists(theta_path):
            print(f"  SKIP {fname} — theta not found at {theta_path}")
            continue

        theta = np.load(theta_path)

        df = pd.read_csv(os.path.join(args.input_folder, fname))
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df = df.sort_values("date").reset_index(drop=True)

        close     = df["close"].values
        log_range = df["high"].values - df["low"].values  # log(H_t/L_t), always > 0
        Q_var     = np.column_stack([
            df["open"].values ** 2,                # squared overnight log-return
            log_range ** 2 / (4.0 * np.log(2)),   # Parkinson variance estimator
        ])

        # full-series sample variance used to initialise the GARCH recursion
        init_log_sig2 = np.log(max(np.var(close), 1e-12))

        n       = len(df)
        starts  = list(range(0, n - args.seq_len + 1, args.stride))
        print(f"  {fname}: {len(starts)} window(s)")

        rows = []
        for w_idx, ws in enumerate(starts):
            y_w    = close[ws: ws + args.seq_len]
            Q_w    = Q_var[ws: ws + args.seq_len]
            paths  = simulate_window(y_w, Q_w, theta, args.num_samples, rng, init_log_sig2)

            for s in range(args.num_samples):
                row = {"window_idx": w_idx, "sample_idx": s,
                       "file": fname, "window_start": ws}
                row.update(zip(step_cols, paths[s]))
                rows.append(row)

        chunk = pd.DataFrame(rows)
        chunk.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        total_windows += len(starts)

    print(f"\nDone. {total_windows} window(s) × {args.num_samples} samples.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
