"""
fit_merton_from_gt.py

Fit one Merton-X jump-diffusion model per ticker from a ground-truth OHLC CSV
produced by generate_samples.py.

The gt CSV has columns:
    window_idx, feature, file, window_start, step_000, ..., step_{L-1}

For each unique ticker (file), all available windows are merged into a single
time series (same reconstruction logic as fit_garch_from_gt.py), and then the
Merton-X model is fitted on [close] with Q = [open^2, (high-low)^2].

Merton-X model recap
---------------------
    r_t = mu + nu_t * Z_t + sum_{i=1}^{P_t} Y_{t,i}

    log(nu_t^2) = a0 + a.T @ q_t           (time-varying diffusion variance)
    P_t  ~ Poisson(lambda)                  (number of jumps)
    Y_{t,i} ~ N(mu_J, sigma_J^2)           (jump sizes)
    q_t = [open^2, (high - low)^2]

NOTE: Functions are inlined from merton_x_jump_diffusion.py rather than
imported because that file executes fitting code at module level.

Usage:
    python fit_merton_from_gt.py \\
        --generated_path ./data/generated/.../train_gt_ohlc.csv \\
        --seq_len 512 \\
        [--out_dir ./checkpoints/merton] \\
        [--print_every 50]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, poisson

# ---------------------------------------------------------------------------
# Merton-X core  (inlined — source file has module-level side effects)
# ---------------------------------------------------------------------------

KMAX = 30   # Poisson sum truncation; covers lambda up to ~15


def unpack_params(theta, k_var):
    """Extract interpretable parameters from the flat optimizer vector.

    Layout: [mu, a0, a1, ..., a_{k_var}, log_lambda, mu_J, log_sigma_J]
    """
    idx = 0
    mu  = theta[idx];               idx += 1
    a0  = theta[idx];               idx += 1
    a   = theta[idx: idx + k_var];  idx += k_var
    lam     = np.exp(theta[idx]);   idx += 1   # lambda > 0
    mu_J    = theta[idx];           idx += 1
    sigma_J = np.exp(theta[idx]);   idx += 1   # sigma_J > 0
    return mu, a0, a, lam, mu_J, sigma_J


def pack_params(mu, a0, a, lam, mu_J, sigma_J):
    return np.r_[mu, a0, a, np.log(lam), mu_J, np.log(sigma_J)]


def compute_log_nu2(a0, a, Q):
    """log(nu_t^2) = a0 + a.T @ q_t, clipped for numerical stability."""
    return np.clip(a0 + Q @ a, -30, 30)


def neg_loglik_merton_x_vec(theta, y, Q, kmax=KMAX):
    """Vectorized negative log-likelihood of the Merton-X model.

    Evaluates the Poisson-mixture density for all t simultaneously:
        p(r_t | q_t) = sum_{k=0}^{kmax} Pois(k; lambda)
                       * N(r_t; mu + k*mu_J, nu_t^2 + k*sigma_J^2)
    via logsumexp for numerical stability.
    """
    k_var = Q.shape[1]
    mu, a0, a, lam, mu_J, sigma_J = unpack_params(theta, k_var)

    log_nu2 = compute_log_nu2(a0, a, Q)   # (T,)
    nu2     = np.exp(log_nu2)             # (T,)

    k_vals     = np.arange(kmax + 1)                 # (K+1,)
    log_pois_w = poisson.logpmf(k_vals, lam)         # (K+1,)

    # Broadcast (T,1) with (1,K+1) -> (T, K+1)
    mix_means = mu + k_vals[None, :] * mu_J
    mix_vars  = nu2[:, None] + k_vals[None, :] * (sigma_J ** 2)

    log_gauss = (
        -0.5 * np.log(2 * np.pi * mix_vars)
        - 0.5 * ((y[:, None] - mix_means) ** 2) / mix_vars
    )  # (T, K+1)

    log_p = logsumexp(log_pois_w[None, :] + log_gauss, axis=1)  # (T,)

    if not np.all(np.isfinite(log_p)):
        return 1e12

    return -np.sum(log_p)


def build_initial_theta(y, k_var, lam0, mu_J0, sigma_J0, a_val):
    """Construct one starting point for the optimizer."""
    mu0  = float(np.mean(y))
    a0_0 = float(np.log(np.var(y) + 1e-8))
    a_0  = np.full(k_var, a_val)
    return pack_params(mu=mu0, a0=a0_0, a=a_0, lam=lam0, mu_J=mu_J0, sigma_J=sigma_J0)


# Multi-start initialization grid (same as merton_x_jump_diffusion.py)
_INIT_GRID = [
    {"lam0": 0.05,  "mu_J0": 0.0, "sigma_J0": 0.005, "a_val":  0.0},
    {"lam0": 0.10,  "mu_J0": 0.0, "sigma_J0": 0.010, "a_val":  0.0},
    {"lam0": 0.20,  "mu_J0": 0.0, "sigma_J0": 0.020, "a_val":  0.0},
    {"lam0": 0.50,  "mu_J0": 0.0, "sigma_J0": 0.050, "a_val":  0.0},
    {"lam0": 1.00,  "mu_J0": 0.0, "sigma_J0": 0.100, "a_val":  0.0},
    {"lam0": 0.10,  "mu_J0": 0.0, "sigma_J0": 0.010, "a_val":  0.01},
    {"lam0": 0.10,  "mu_J0": 0.0, "sigma_J0": 0.010, "a_val": -0.01},
]


def fit_merton_x(y, Q, max_iter=5000, print_every=0, ticker=""):
    """Fit Merton-X via multi-start L-BFGS-B; return best OptimizeResult."""
    y     = np.asarray(y, dtype=float)
    Q     = np.asarray(Q, dtype=float)
    k_var = Q.shape[1]

    best_result = None

    for run_id, init in enumerate(_INIT_GRID, start=1):
        theta0 = build_initial_theta(y=y, k_var=k_var, **init)

        if print_every > 0:
            state = {"iter": 0}
            def callback(th, _state=state, _rid=run_id, _y=y, _Q=Q):
                _state["iter"] += 1
                if _state["iter"] % print_every == 0:
                    nll = neg_loglik_merton_x_vec(th, _y, _Q)
                    _, _, _, lam_, mu_J_, sigma_J_ = unpack_params(th, _Q.shape[1])
                    print(f"  [{ticker}] run={_rid} iter={_state['iter']:4d} "
                          f"nll={nll:.4f} lam={lam_:.4f} mu_J={mu_J_:.4f} sigma_J={sigma_J_:.4f}")
        else:
            callback = None

        result = minimize(
            neg_loglik_merton_x_vec,
            theta0,
            args=(y, Q),
            method="L-BFGS-B",
            options={"maxiter": max_iter},
            callback=callback,
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result


# ---------------------------------------------------------------------------
# Series reconstruction from gt windows  (from fit_garch_from_gt.py)
# ---------------------------------------------------------------------------

def reconstruct_series_for_ticker(ticker_df, seq_len):
    """Merge overlapping windows for one ticker into a single time-ordered series.

    Returns a DataFrame indexed by global_pos with columns [close, open, high, low].
    When windows overlap, the earliest window's value is kept.
    """
    step_cols = [f"step_{t:03d}" for t in range(seq_len)]
    records   = {}  # global_pos -> {feature: value}

    for feature in ("close", "open", "high", "low"):
        feat_rows = ticker_df[ticker_df["feature"] == feature]
        for _, row in feat_rows.iterrows():
            ws = int(row["window_start"])
            for t, col in enumerate(step_cols):
                gpos = ws + t
                if gpos not in records:
                    records[gpos] = {}
                if feature not in records[gpos]:   # first occurrence wins
                    records[gpos][feature] = float(row[col])

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "global_pos"
    df = df.sort_index().dropna(subset=["close", "open", "high", "low"])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit per-ticker Merton-X from gt OHLC CSV."
    )
    p.add_argument("--generated_path", type=str, required=True,
                   help="Path to train_gt_ohlc.csv (or equivalent).")
    p.add_argument("--seq_len", type=int, required=True,
                   help="Window length (number of step_* columns).")
    p.add_argument("--out_dir", type=str, default="./checkpoints/merton",
                   help="Output directory for .npy files.")
    p.add_argument("--max_iter", type=int, default=5000,
                   help="Max L-BFGS-B iterations per initialization.")
    p.add_argument("--print_every", type=int, default=0,
                   help="Print progress every N optimizer iterations (0 = silent).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading: {args.generated_path}")
    gt = pd.read_csv(args.generated_path)

    step_cols = [f"step_{t:03d}" for t in range(args.seq_len)]
    missing   = [c for c in step_cols if c not in gt.columns]
    if missing:
        raise ValueError(f"Missing step columns (first few): {missing[:5]}")

    tickers = sorted(gt["file"].unique())
    print(f"Tickers found: {len(tickers)}")

    manifest_rows = []

    for i, ticker in enumerate(tickers, start=1):
        stem      = os.path.splitext(ticker)[0]
        ticker_df = gt[gt["file"] == ticker]
        n_windows = ticker_df["window_idx"].nunique()

        print(f"\n[{i}/{len(tickers)}] {stem}  ({n_windows} window(s))")

        # --- reconstruct series ---
        try:
            series = reconstruct_series_for_ticker(ticker_df, args.seq_len)
        except Exception as exc:
            print(f"  SKIP — reconstruction failed: {exc}")
            manifest_rows.append({"ticker": stem, "status": "reconstruction_failed",
                                  "n_obs": 0, "nll": np.nan,
                                  "lambda": np.nan, "mu_J": np.nan, "sigma_J": np.nan})
            continue

        n_obs = len(series)
        print(f"  Reconstructed series: {n_obs} obs")

        if n_obs < 30:
            print(f"  SKIP — too few observations ({n_obs})")
            manifest_rows.append({"ticker": stem, "status": "too_few_obs",
                                  "n_obs": n_obs, "nll": np.nan,
                                  "lambda": np.nan, "mu_J": np.nan, "sigma_J": np.nan})
            continue

        # --- build y and Q ---
        y     = series["close"].values
        o     = series["open"].values
        h     = series["high"].values
        l_    = series["low"].values
        Q     = np.column_stack([o ** 2, (h - l_) ** 2])   # q_t = [open^2, range^2]

        # --- fit ---
        try:
            result = fit_merton_x(
                y=y, Q=Q,
                max_iter=args.max_iter,
                print_every=args.print_every,
                ticker=stem,
            )
        except Exception as exc:
            print(f"  SKIP — optimization failed: {exc}")
            manifest_rows.append({"ticker": stem, "status": "optimization_failed",
                                  "n_obs": n_obs, "nll": np.nan,
                                  "lambda": np.nan, "mu_J": np.nan, "sigma_J": np.nan})
            continue

        mu_, a0_, a_, lam_, mu_J_, sigma_J_ = unpack_params(result.x, k_var=Q.shape[1])
        print(f"  nll={result.fun:.4f}  lambda={lam_:.4f}  mu_J={mu_J_:.4f}  "
              f"sigma_J={sigma_J_:.4f}  success={result.success}")

        # --- save theta ---
        theta_path = os.path.join(args.out_dir, f"{stem}_theta.npy")
        np.save(theta_path, result.x)

        manifest_rows.append({
            "ticker":   stem,
            "status":   "ok" if result.success else "converged_with_warning",
            "n_obs":    n_obs,
            "n_windows": n_windows,
            "nll":      result.fun,
            "mu":       mu_,
            "a0":       a0_,
            "lambda":   lam_,
            "mu_J":     mu_J_,
            "sigma_J":  sigma_J_,
        })

    # --- save manifest ---
    manifest      = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"\nDone. {len(tickers)} tickers processed.")
    print(f"Results saved to: {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    ok   = (manifest["status"] == "ok").sum()
    warn = (manifest["status"] == "converged_with_warning").sum()
    skip = len(manifest) - ok - warn
    print(f"  ok={ok}  warnings={warn}  skipped={skip}")


if __name__ == "__main__":
    main()