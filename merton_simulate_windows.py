"""
merton_simulate_windows.py

Load pre-fitted Merton-X parameters (from fit_merton_from_csv.py) and
simulate num_samples independent close paths per sliding window over each
ticker CSV. Output format matches the EDM-generated CSV so results can
be compared directly.

For each window, all T steps are drawn independently:
    r_t = mu + nu_t * Z_t + sum_{i=1}^{P_t} Y_{t,i}
where:
    nu_t    = sqrt(exp(a0 + a.T @ q_t)),   q_t = [open_t^2, (high_t-low_t)^2]
    Z_t     ~ N(0, 1)
    P_t     ~ Poisson(lambda)
    Y_{t,i} ~ N(mu_J, sigma_J^2)

No initialisation from actual close values is needed because the model has
no AR component: each r_t depends only on contemporaneous q_t and the noise.

Output columns: window_idx, sample_idx, file, window_start, step_000, ..., step_{T-1}

Usage:
    python merton_simulate_windows.py \
        --input_folder ./data/SNP500_individual_processed \
        --merton_dir   ./checkpoints/merton \
        --out_dir      ./data/generated/merton \
        --seq_len 512 \
        --stride  100 \
        --num_samples 250 \
        [--seed 42]

python merton_simulate_windows.py --input_folder ./data/SNP500_individual_processed_copy --merton_dir ./checkpoints/merton --out_dir ./data/generated/merton --seq_len 512 --stride 100 --num_samples 50 --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameter unpacking (identical to fit_merton_from_csv.py)
# ---------------------------------------------------------------------------

def unpack_params(theta, k_var):
    idx = 0
    mu      = theta[idx];              idx += 1
    a0      = theta[idx];              idx += 1
    a       = theta[idx: idx + k_var]; idx += k_var
    lam     = np.exp(theta[idx]);      idx += 1
    mu_J    = theta[idx];              idx += 1
    sigma_J = np.exp(theta[idx]);      idx += 1
    return mu, a0, a, lam, mu_J, sigma_J


# ---------------------------------------------------------------------------
# Per-window simulation
# ---------------------------------------------------------------------------

def simulate_window(Q_var_window, theta, num_samples, rng):
    """
    Generate num_samples close paths for one window.

    All T steps are drawn independently: each r_t depends only on q_t
    (open/high/low conditioning) and the model parameters. No actual
    close value is used.

    The sum of P_t i.i.d. N(mu_J, sigma_J^2) draws equals
    N(P_t * mu_J, P_t * sigma_J^2), which is used directly when P_t > 0
    and set to 0 when P_t == 0.

    Returns array of shape (num_samples, T).
    """
    mu, a0, a, lam, mu_J, sigma_J = unpack_params(theta, k_var=2)
    T = Q_var_window.shape[0]

    log_nu2 = np.clip(a0 + Q_var_window @ a, -30, 30)  # (T,)
    nu      = np.sqrt(np.exp(log_nu2))                  # (T,)

    Z        = rng.standard_normal((num_samples, T))           # diffusion noise
    n_jumps  = rng.poisson(lam, size=(num_samples, T))         # jump counts

    # Sum of n_jumps[s,t] i.i.d. N(mu_J, sigma_J^2) ~ N(n*mu_J, n*sigma_J^2).
    # When n==0 the sum is 0; masked to avoid drawing from a degenerate N(0,0).
    has_jump  = n_jumps > 0
    jump_std  = np.where(has_jump, np.sqrt(n_jumps.astype(float)) * sigma_J, 1.0)
    jump_sum  = np.where(has_jump, rng.normal(n_jumps * mu_J, jump_std), 0.0)

    paths = mu + nu[None, :] * Z + jump_sum  # (num_samples, T)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate Merton-X close paths per window; output mirrors EDM CSV format."
    )
    p.add_argument("--input_folder", required=True,
                   help="Folder with CSVs (columns: date, close, open, high, low).")
    p.add_argument("--merton_dir", required=True,
                   help="Folder with <stem>_theta.npy files (output of fit_merton_from_csv.py).")
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
        theta_path = os.path.join(args.merton_dir, f"{stem}_theta.npy")

        if not os.path.exists(theta_path):
            print(f"  SKIP {fname} — theta not found at {theta_path}")
            continue

        theta = np.load(theta_path)

        df = pd.read_csv(os.path.join(args.input_folder, fname))
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df = df.sort_values("date").reset_index(drop=True)

        Q_var = np.column_stack([df["open"].values ** 2,
                                 (df["high"].values - df["low"].values) ** 2])

        n      = len(df)
        starts = list(range(0, n - args.seq_len + 1, args.stride))
        print(f"  {fname}: {len(starts)} window(s)")

        rows = []
        for w_idx, ws in enumerate(starts):
            Q_w   = Q_var[ws: ws + args.seq_len]
            paths = simulate_window(Q_w, theta, args.num_samples, rng)

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
