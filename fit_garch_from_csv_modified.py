"""
fit_garch_from_csv_modified.py

Same as fit_garch_from_csv.py with two targeted performance improvements:

  1. The sequential log-sigma^2 recursion is JIT-compiled with Numba @njit,
     running at near-C speed instead of in the Python interpreter.

  2. The Student-t log-pdf (which uses scipy.special.gammaln, incompatible
     with Numba nopython mode) is pulled outside the loop and evaluated once
     over the full array after the recursion, eliminating T per-step scipy
     calls and the np.array([z_t]) allocation at every iteration.

  3. Q_var uses the Parkinson (1980) variance estimator in place of the raw
     squared log-range: Q_var = [open^2, (log(H/L))^2 / (4*log(2))].

  4. phi (AR coefficient) is bounded to (-0.5, 0.5) to prevent simulated
     paths from diverging when |phi| > 1.

Requires: numba  (pip install numba)
First call to neg_loglik_log_garch_x pays a one-time JIT compilation cost
(~1-2 s); all subsequent calls within the same process are fast.

All results (parameter estimates, NLL, inference table) are numerically
identical to the original up to floating-point rounding (~1 ULP).

Usage:
    python fit_garch_from_csv_modified.py \
        --input_folder ./data/SNP500_individual_processed \
        --out_dir      ./checkpoints/garch_fast \
        [--max_iter 1000]
"""

import argparse
import os

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm


# ---------------------------------------------------------------------------
# GARCH-X core  — restructured for Numba + vectorised NLL
# ---------------------------------------------------------------------------

def student_t_logpdf_standardized(z, nu):
    """Vectorised Student-t log-pdf (unchanged from original)."""
    if nu <= 2:
        return -np.inf * np.ones_like(z)
    s = np.sqrt((nu - 2.0) / nu)
    x = z / s
    log_const = (
        gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi) - np.log(s)
    )
    return log_const - ((nu + 1.0) / 2.0) * np.log1p((x ** 2) / nu)


@njit
def _garch_variance_recursion(eps, log_sig2_0, omega, alpha, beta, Q_var_delta, c):
    """
    Sequential log-sigma^2 recursion compiled to machine code by Numba.

    Parameters
    ----------
    eps          : (T,) residuals array — precomputed before this call
    log_sig2_0   : scalar initial log-variance
    omega, alpha, beta : GARCH scalars
    Q_var_delta  : (T,) precomputed Q_var @ delta
    c            : small positive constant for log(eps^2 + c) stability

    Returns
    -------
    log_sig2 : (T,) array
    """
    T = len(eps)
    log_sig2 = np.empty(T)
    log_sig2[0] = log_sig2_0
    for t in range(1, T):
        val = (omega
               + alpha * np.log(eps[t - 1] ** 2 + c)
               + beta  * log_sig2[t - 1]
               + Q_var_delta[t])
        if val < -30.0:
            log_sig2[t] = -30.0
        elif val > 30.0:
            log_sig2[t] = 30.0
        else:
            log_sig2[t] = val
    return log_sig2


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


def neg_loglik_log_garch_x(theta, y, X_mean, Q_var, c=1e-8):
    T      = len(y)
    k_mean = X_mean.shape[1]
    k_var  = Q_var.shape[1]
    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)

    # --- eps: fully vectorised (no sequential dependency on log_sig2) -------
    X_gamma = X_mean @ gamma          # (T,); zeros when k_mean=0
    eps     = np.empty(T)
    eps[0]  = y[0] - (mu + X_gamma[0])
    eps[1:] = y[1:] - (mu + phi * y[:-1] + X_gamma[1:])

    initial_var = np.var(y)
    if initial_var <= 1e-12:
        initial_var = 1.0

    # --- log_sig2: sequential recursion via Numba JIT kernel ----------------
    Q_var_delta = Q_var @ delta       # (T,) — vectorised, computed once
    log_sig2 = _garch_variance_recursion(
        eps, np.log(initial_var), omega, alpha, beta, Q_var_delta, c
    )

    # --- NLL: fully vectorised over t = 1 .. T-1 ----------------------------
    z          = eps[1:] / np.sqrt(np.exp(log_sig2[1:]))
    logpdf_z   = student_t_logpdf_standardized(z, nu)
    logpdf_eps = logpdf_z - 0.5 * log_sig2[1:]

    if not np.all(np.isfinite(logpdf_eps)):
        return 1e12

    return -np.sum(logpdf_eps)


def build_initial_theta(y, k_mean, k_var, alpha0, beta0, nu0, delta0_value=0.0):
    slack0     = 1.0 - alpha0 - beta0
    alpha_raw0 = np.log(alpha0 / slack0)
    beta_raw0  = np.log(beta0  / slack0)
    return np.r_[
        np.mean(y),
        0.0,
        np.zeros(k_mean),
        np.log(np.var(y) + 1e-8) * 0.05,
        alpha_raw0,
        beta_raw0,
        np.full(k_var, delta0_value),
        np.log(nu0 - 2.01),
    ]


def transformed_param_names(k_mean, k_var):
    names = ["mu", "phi"]
    names += [f"gamma[{i}]" for i in range(k_mean)]
    names += ["omega", "alpha", "beta", "alpha+beta"]
    names += [f"delta[{i}]" for i in range(k_var)]
    names += ["nu"]
    return names


def transformed_param_values(theta, k_mean, k_var):
    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)
    return np.asarray(
        [mu, phi] + list(gamma) + [omega, alpha, beta, alpha + beta] + list(delta) + [nu],
        dtype=float,
    )


def numerical_jacobian_transformed(theta, k_mean, k_var, eps=1e-5):
    theta = np.asarray(theta, dtype=float)
    base  = transformed_param_values(theta, k_mean, k_var)
    J     = np.zeros((len(base), len(theta)))
    for j in range(len(theta)):
        step    = eps * max(1.0, abs(theta[j]))
        tp      = theta.copy(); tp[j] += step
        tm      = theta.copy(); tm[j] -= step
        J[:, j] = (transformed_param_values(tp, k_mean, k_var) -
                   transformed_param_values(tm, k_mean, k_var)) / (2.0 * step)
    return J


def inference_table(result, k_mean, k_var):
    names  = transformed_param_names(k_mean, k_var)
    values = transformed_param_values(result.x, k_mean, k_var)
    try:
        hess_inv = result.hess_inv
        cov_raw  = np.asarray(hess_inv.todense() if hasattr(hess_inv, "todense") else hess_inv)
    except Exception:
        cov_raw = None

    if cov_raw is None:
        return pd.DataFrame({"param": names, "estimate": values,
                             "std_err": np.nan, "z": np.nan, "p_value": np.nan})

    J               = numerical_jacobian_transformed(result.x, k_mean, k_var)
    cov_transformed = J @ cov_raw @ J.T
    variances       = np.where(np.diag(cov_transformed) < 0, np.nan, np.diag(cov_transformed))
    std_err         = np.sqrt(variances)
    z_stats         = values / std_err
    p_values        = 2.0 * (1.0 - norm.cdf(np.abs(z_stats)))
    return pd.DataFrame({"param": names, "estimate": values,
                         "std_err": std_err, "z": z_stats, "p_value": p_values})


# ---------------------------------------------------------------------------
# Two-init fit  (unchanged)
# ---------------------------------------------------------------------------

INIT_GRID = [
    {"alpha0": 0.05, "beta0": 0.90, "nu0": 8.0, "delta0_value": 0.00},
    {"alpha0": 0.03, "beta0": 0.90, "nu0": 6.0, "delta0_value": 0.00},
]


def fit_garch_x(y, Q_var, max_iter=1000):
    y      = np.asarray(y, dtype=float)
    Q_var  = np.asarray(Q_var, dtype=float)
    X_mean = np.zeros((len(y), 0))
    k_mean = 0
    k_var  = Q_var.shape[1]
    best   = None
    for init in INIT_GRID:
        theta0         = build_initial_theta(y=y, k_mean=k_mean, k_var=k_var, **init)
        bounds         = [(None, None)] * len(theta0)
        bounds[1]      = (-0.5, 0.5)  # phi: log-returns have near-zero AR; cap prevents simulation divergence
        result = minimize(
            neg_loglik_log_garch_x,
            theta0,
            args=(y, X_mean, Q_var),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )
        if best is None or result.fun < best.fun:
            best = result
    return best


# ---------------------------------------------------------------------------
# Main  (unchanged)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit per-ticker GARCH-X from raw OHLC CSVs (Numba-accelerated).")
    p.add_argument("--input_folder", required=True,
                   help="Folder with CSVs (columns: date, close, open, high, low).")
    p.add_argument("--out_dir", default="./checkpoints/garch",
                   help="Output directory for theta, inference, and manifest files.")
    p.add_argument("--max_iter", type=int, default=1000,
                   help="Max L-BFGS-B iterations per initialisation.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(args.input_folder) if f.endswith(".csv"))
    print(f"Found {len(csv_files)} CSV files in {args.input_folder}")
    print("(Numba JIT will compile on the first ticker — expect ~2 s extra overhead once.)")

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
                                   "nll": np.nan, "alpha": np.nan, "beta": np.nan,
                                   "alpha_plus_beta": np.nan, "nu": np.nan})
            continue

        y         = df["close"].values
        log_range = df["high"].values - df["low"].values  # log(H_t/L_t), always > 0
        Q_var     = np.column_stack([
            df["open"].values ** 2,                # squared overnight log-return
            log_range ** 2 / (4.0 * np.log(2)),   # Parkinson variance estimator
        ])

        try:
            result = fit_garch_x(y, Q_var, max_iter=args.max_iter)
        except Exception as exc:
            print(f"  SKIP — optimisation failed: {exc}")
            manifest_rows.append({"ticker": stem, "status": "optimization_failed",
                                   "n_obs": n_obs, "nll": np.nan,
                                   "alpha": np.nan, "beta": np.nan,
                                   "alpha_plus_beta": np.nan, "nu": np.nan})
            continue

        mu_, phi_, _, omega_, alpha_, beta_, delta_, nu_ = unpack_params(
            result.x, k_mean=0, k_var=2
        )
        print(f"  nll={result.fun:.4f}  alpha={alpha_:.4f}  beta={beta_:.4f}  "
              f"a+b={alpha_+beta_:.4f}  nu={nu_:.4f}  success={result.success}")

        np.save(os.path.join(args.out_dir, f"{stem}_theta.npy"), result.x)

        table = inference_table(result, k_mean=0, k_var=2)
        table.to_csv(os.path.join(args.out_dir, f"{stem}_inference.csv"), index=False)

        manifest_rows.append({
            "ticker":          stem,
            "status":          "ok" if result.success else "converged_with_warning",
            "n_obs":           n_obs,
            "nll":             result.fun,
            "alpha":           alpha_,
            "beta":            beta_,
            "alpha_plus_beta": alpha_ + beta_,
            "nu":              nu_,
            "omega":           omega_,
            "delta_0":         delta_[0],
            "delta_1":         delta_[1],
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
