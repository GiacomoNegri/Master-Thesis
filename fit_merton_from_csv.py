"""
fit_merton_from_csv.py

Fit one Merton-X jump-diffusion model per CSV file in an input folder.
Each CSV must have columns: date, close, open, high, low (date format %d/%m/%Y).
The full time series (all rows, sorted by date) is used for fitting.

Model (from fit_merton_from_gt.py):
    r_t = mu + nu_t * Z_t + sum_{i=1}^{P_t} Y_{t,i}
    log(nu_t^2) = a0 + a.T @ q_t
    P_t  ~ Poisson(lambda)
    Y_{t,i} ~ N(mu_J, sigma_J^2)
    q_t = [open_t^2, (high_t - low_t)^2]

Two initialisations are tried; the one with the lower NLL is kept.

Outputs per ticker (in out_dir):
    <stem>_theta.npy       — raw parameter vector (for simulation)
    manifest.csv           — summary across all tickers

Usage:
    python fit_merton_from_csv.py \
        --input_folder ./data/fake_fts_processed \
        --out_dir      ./checkpoints/merton \
        [--max_iter 5000]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# Merton-X core  (inlined from fit_merton_from_gt.py — identical logic)
# ---------------------------------------------------------------------------

KMAX = 30  # Poisson sum truncation; covers lambda up to ~15


def unpack_params(theta, k_var):
    idx = 0
    mu      = theta[idx];              idx += 1
    a0      = theta[idx];              idx += 1
    a       = theta[idx: idx + k_var]; idx += k_var
    lam     = np.exp(theta[idx]);      idx += 1  # lambda > 0
    mu_J    = theta[idx];              idx += 1
    sigma_J = np.exp(theta[idx]);      idx += 1  # sigma_J > 0
    return mu, a0, a, lam, mu_J, sigma_J


def pack_params(mu, a0, a, lam, mu_J, sigma_J):
    return np.r_[mu, a0, a, np.log(lam), mu_J, np.log(sigma_J)]


def neg_loglik_merton_x_vec(theta, y, Q, kmax=KMAX):
    k_var = Q.shape[1]
    mu, a0, a, lam, mu_J, sigma_J = unpack_params(theta, k_var)

    log_nu2 = np.clip(a0 + Q @ a, -30, 30)  # (T,)
    nu2     = np.exp(log_nu2)               # (T,)

    k_vals     = np.arange(kmax + 1)
    log_pois_w = poisson.logpmf(k_vals, lam)

    mix_means = mu + k_vals[None, :] * mu_J                          # (T, K+1)
    mix_vars  = nu2[:, None] + k_vals[None, :] * (sigma_J ** 2)     # (T, K+1)

    log_gauss = (
        -0.5 * np.log(2 * np.pi * mix_vars)
        - 0.5 * ((y[:, None] - mix_means) ** 2) / mix_vars
    )

    log_p = logsumexp(log_pois_w[None, :] + log_gauss, axis=1)

    if not np.all(np.isfinite(log_p)):
        return 1e12

    return -np.sum(log_p)


def build_initial_theta(y, k_var, lam0, mu_J0, sigma_J0, a_val):
    return pack_params(
        mu=float(np.mean(y)),
        a0=float(np.log(np.var(y) + 1e-8)),
        a=np.full(k_var, a_val),
        lam=lam0,
        mu_J=mu_J0,
        sigma_J=sigma_J0,
    )


# ---------------------------------------------------------------------------
# Two-init fit  (two most structurally distinct points from the original 7)
# ---------------------------------------------------------------------------

INIT_GRID = [
    {"lam0": 0.10, "mu_J0": 0.0, "sigma_J0": 0.010, "a_val": 0.0},  # low jump rate
    {"lam0": 0.50, "mu_J0": 0.0, "sigma_J0": 0.050, "a_val": 0.0},  # higher jump rate
]


def fit_merton_x(y, Q, max_iter=5000):
    y     = np.asarray(y, dtype=float)
    Q     = np.asarray(Q, dtype=float)
    k_var = Q.shape[1]
    best  = None
    for init in INIT_GRID:
        theta0 = build_initial_theta(y=y, k_var=k_var, **init)
        result = minimize(
            neg_loglik_merton_x_vec,
            theta0,
            args=(y, Q),
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        if best is None or result.fun < best.fun:
            best = result
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit per-ticker Merton-X from raw OHLC CSVs.")
    p.add_argument("--input_folder", required=True,
                   help="Folder with CSVs (columns: date, close, open, high, low).")
    p.add_argument("--out_dir", default="./checkpoints/merton",
                   help="Output directory for theta and manifest files.")
    p.add_argument("--max_iter", type=int, default=5000,
                   help="Max L-BFGS-B iterations per initialisation.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(args.input_folder) if f.endswith(".csv"))
    print(f"Found {len(csv_files)} CSV files in {args.input_folder}")

    manifest_rows = []

    for i, fname in enumerate(csv_files, start=1):
        stem = os.path.splitext(fname)[0]
        path = os.path.join(args.input_folder, fname)

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df = df.sort_values("date").reset_index(drop=True)
        n_obs = len(df)

        print(f"\n[{i}/{len(csv_files)}] {stem}  ({n_obs} obs)")

        if n_obs < 30:
            print(f"  SKIP — too few observations")
            manifest_rows.append({"ticker": stem, "status": "too_few_obs", "n_obs": n_obs,
                                   "nll": np.nan, "lambda": np.nan,
                                   "mu_J": np.nan, "sigma_J": np.nan})
            continue

        y = df["close"].values
        Q = np.column_stack([df["open"].values ** 2,
                             (df["high"].values - df["low"].values) ** 2])

        try:
            result = fit_merton_x(y, Q, max_iter=args.max_iter)
        except Exception as exc:
            print(f"  SKIP — optimisation failed: {exc}")
            manifest_rows.append({"ticker": stem, "status": "optimization_failed",
                                   "n_obs": n_obs, "nll": np.nan,
                                   "lambda": np.nan, "mu_J": np.nan, "sigma_J": np.nan})
            continue

        mu_, a0_, a_, lam_, mu_J_, sigma_J_ = unpack_params(result.x, k_var=2)
        print(f"  nll={result.fun:.4f}  lambda={lam_:.4f}  mu_J={mu_J_:.4f}  "
              f"sigma_J={sigma_J_:.4f}  success={result.success}")

        np.save(os.path.join(args.out_dir, f"{stem}_theta.npy"), result.x)

        manifest_rows.append({
            "ticker":   stem,
            "status":   "ok" if result.success else "converged_with_warning",
            "n_obs":    n_obs,
            "nll":      result.fun,
            "mu":       mu_,
            "a0":       a0_,
            "a_0":      a_[0],
            "a_1":      a_[1],
            "lambda":   lam_,
            "mu_J":     mu_J_,
            "sigma_J":  sigma_J_,
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)

    ok   = (manifest["status"] == "ok").sum()
    warn = (manifest["status"] == "converged_with_warning").sum()
    skip = len(manifest) - ok - warn
    print(f"\nDone. {len(csv_files)} tickers processed.")
    print(f"  ok={ok}  warnings={warn}  skipped={skip}")
    print(f"  Results: {args.out_dir}")


if __name__ == "__main__":
    main()
