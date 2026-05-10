"""
fit_garch_from_gt.py

Fit one lightweight GARCH-X (log-GARCH, Student-t innovations) model per ticker
from a ground-truth OHLC CSV produced by generate_samples.py.

The gt CSV has columns:
    window_idx, feature, file, window_start, step_000, ..., step_{L-1}

For each unique ticker (file), all available windows are merged into a single
time series by mapping each step to its global position (window_start + step_idx),
deduplicating overlapping positions (keep first/earliest), and sorting.
One GARCH-X is then fitted on [close] with Q_var = [open^2, (high-low)^2].

NOTE: Functions are inlined from garch_x_studentt_innovations.py rather than
imported because that file executes fitting code at module level.

Usage:
    python fit_garch_from_gt.py \\
        --generated_path ./data/generated/.../train_gt_ohlc.csv \\
        --seq_len 512 \\
        [--out_dir ./checkpoints/garch] \\
        [--max_iter 1000] \\
        [--print_every 50]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm


# ---------------------------------------------------------------------------
# GARCH-X core  (inlined — source file has module-level side effects)
# ---------------------------------------------------------------------------

def student_t_logpdf_standardized(z, nu):
    if nu <= 2:
        return -np.inf * np.ones_like(z)
    s = np.sqrt((nu - 2.0) / nu)
    x = z / s
    log_const = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(s)
    )
    return log_const - ((nu + 1.0) / 2.0) * np.log1p((x ** 2) / nu)


def unpack_params(theta, k_mean, k_var):
    idx = 0
    mu  = theta[idx]; idx += 1
    phi = theta[idx]; idx += 1
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
    nu = 2.01 + np.exp(theta[idx])
    return mu, phi, gamma, omega, alpha, beta, delta, nu


def neg_loglik_log_garch_x(theta, y, X_mean, Q_var, c=1e-8):
    T = len(y)
    k_mean = X_mean.shape[1]
    k_var  = Q_var.shape[1]

    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)

    eps     = np.zeros(T)
    log_sig2 = np.zeros(T)
    sig2    = np.zeros(T)

    initial_var = np.var(y)
    if initial_var <= 1e-12:
        initial_var = 1.0

    sig2[0]    = initial_var
    log_sig2[0] = np.log(initial_var)
    eps[0]     = y[0] - (mu + X_mean[0] @ gamma)

    nll = 0.0
    for t in range(1, T):
        mean_t  = mu + phi * y[t - 1] + X_mean[t] @ gamma
        eps[t]  = y[t] - mean_t

        log_sig2[t] = (
            omega
            + alpha * np.log(eps[t - 1] ** 2 + c)
            + beta  * log_sig2[t - 1]
            + Q_var[t] @ delta
        )
        log_sig2[t] = np.clip(log_sig2[t], -30.0, 30.0)
        sig2[t] = np.exp(log_sig2[t])

        z_t = eps[t] / np.sqrt(sig2[t])
        logpdf_z   = student_t_logpdf_standardized(np.array([z_t]), nu)[0]
        logpdf_eps = logpdf_z - 0.5 * log_sig2[t]

        if not np.isfinite(logpdf_eps):
            return 1e12

        nll -= logpdf_eps

    return nll


def build_initial_theta(y, k_mean, k_var, alpha0, beta0, nu0, delta0_value=0.0):
    slack0     = 1.0 - alpha0 - beta0
    alpha_raw0 = np.log(alpha0 / slack0)
    beta_raw0  = np.log(beta0  / slack0)
    theta0 = np.r_[
        np.mean(y),                              # mu
        0.0,                                     # phi
        np.zeros(k_mean),                        # gamma
        np.log(np.var(y) + 1e-8) * 0.05,        # omega
        alpha_raw0,
        beta_raw0,
        np.full(k_var, delta0_value),            # delta
        np.log(nu0 - 2.01),                      # nu_raw
    ]
    return theta0


def transformed_param_names(k_mean, k_var):
    names = ["mu", "phi"]
    names += [f"gamma[{i}]" for i in range(k_mean)]
    names += ["omega", "alpha", "beta", "alpha+beta"]
    names += [f"delta[{i}]" for i in range(k_var)]
    names += ["nu"]
    return names


def transformed_param_values(theta, k_mean, k_var):
    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)
    values = [mu, phi] + list(gamma) + [omega, alpha, beta, alpha + beta] + list(delta) + [nu]
    return np.asarray(values, dtype=float)


def numerical_jacobian_transformed(theta, k_mean, k_var, eps=1e-5):
    theta      = np.asarray(theta, dtype=float)
    base       = transformed_param_values(theta, k_mean, k_var)
    J          = np.zeros((len(base), len(theta)))
    for j in range(len(theta)):
        step          = eps * max(1.0, abs(theta[j]))
        tp            = theta.copy(); tp[j] += step
        tm            = theta.copy(); tm[j] -= step
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
    variances       = np.diag(cov_transformed)
    variances       = np.where(variances < 0, np.nan, variances)
    std_err         = np.sqrt(variances)
    z_stats         = values / std_err
    p_values        = 2.0 * (1.0 - norm.cdf(np.abs(z_stats)))

    return pd.DataFrame({"param": names, "estimate": values,
                         "std_err": std_err, "z": z_stats, "p_value": p_values})


# ---------------------------------------------------------------------------
# Lightweight multi-start fit  (3 inits, no bootstrap)
# ---------------------------------------------------------------------------

INIT_GRID = [
    {"alpha0": 0.05, "beta0": 0.90, "nu0": 8.0,  "delta0_value":  0.00},
    {"alpha0": 0.03, "beta0": 0.90, "nu0": 6.0,  "delta0_value":  0.00},
    {"alpha0": 0.08, "beta0": 0.85, "nu0": 10.0, "delta0_value":  0.00},
]


def fit_garch_x_lightweight(y, Q_var, max_iter=1000, print_every=0, ticker=""):
    y     = np.asarray(y, dtype=float)
    Q_var = np.asarray(Q_var, dtype=float)
    X_mean = np.zeros((len(y), 0))

    k_mean = 0
    k_var  = Q_var.shape[1]

    best_result = None

    for run_id, init in enumerate(INIT_GRID, start=1):
        theta0 = build_initial_theta(
            y=y, k_mean=k_mean, k_var=k_var,
            alpha0=init["alpha0"], beta0=init["beta0"],
            nu0=init["nu0"], delta0_value=init["delta0_value"],
        )

        if print_every > 0:
            state = {"iter": 0}
            def callback(th, _state=state, _rid=run_id):
                _state["iter"] += 1
                if _state["iter"] % print_every == 0:
                    nll = neg_loglik_log_garch_x(th, y, X_mean, Q_var)
                    mu_, phi_, gamma_, omega_, alpha_, beta_, delta_, nu_ = unpack_params(th, k_mean, k_var)
                    print(f"  [{ticker}] run={_rid} iter={_state['iter']:4d} "
                          f"nll={nll:.4f} a={alpha_:.4f} b={beta_:.4f} nu={nu_:.4f}")
        else:
            callback = None

        result = minimize(
            neg_loglik_log_garch_x,
            theta0,
            args=(y, X_mean, Q_var),
            method="L-BFGS-B",
            options={"maxiter": max_iter},
            callback=callback,
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result


# ---------------------------------------------------------------------------
# Series reconstruction from gt windows
# ---------------------------------------------------------------------------

def reconstruct_series_for_ticker(ticker_df, seq_len):
    """
    Merge overlapping windows for one ticker into a single time-ordered series.
    Returns a DataFrame indexed by global_pos with columns [close, open, high, low].
    """
    step_cols = [f"step_{t:03d}" for t in range(seq_len)]
    records   = {}  # global_pos -> {close, open, high, low}

    for feature in ("close", "open", "high", "low"):
        feat_rows = ticker_df[ticker_df["feature"] == feature]
        for _, row in feat_rows.iterrows():
            ws = int(row["window_start"])
            for t, col in enumerate(step_cols):
                gpos = ws + t
                if gpos not in records:
                    records[gpos] = {}
                # keep earliest window's value (first occurrence wins)
                if feature not in records[gpos]:
                    records[gpos][feature] = float(row[col])

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "global_pos"
    df = df.sort_index()

    # drop any position missing a feature
    df = df.dropna(subset=["close", "open", "high", "low"])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit per-ticker GARCH-X from gt OHLC CSV.")
    p.add_argument("--generated_path", type=str, required=True,
                   help="Path to train_gt_ohlc.csv (or equivalent).")
    p.add_argument("--seq_len", type=int, required=True,
                   help="Window length (number of step_* columns).")
    p.add_argument("--out_dir", type=str, default="./checkpoints/garch",
                   help="Output directory for .npy and .csv files.")
    p.add_argument("--max_iter", type=int, default=1000,
                   help="Max L-BFGS-B iterations per initialization.")
    p.add_argument("--print_every", type=int, default=0,
                   help="Print optimization progress every N iters (0 = silent).")
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
        stem       = os.path.splitext(ticker)[0]
        ticker_df  = gt[gt["file"] == ticker]
        n_windows  = ticker_df["window_idx"].nunique()

        print(f"\n[{i}/{len(tickers)}] {stem}  ({n_windows} window(s))")

        try:
            series = reconstruct_series_for_ticker(ticker_df, args.seq_len)
        except Exception as exc:
            print(f"  SKIP — reconstruction failed: {exc}")
            manifest_rows.append({"ticker": stem, "status": "reconstruction_failed",
                                  "n_obs": 0, "nll": np.nan,
                                  "alpha": np.nan, "beta": np.nan,
                                  "alpha_plus_beta": np.nan, "nu": np.nan})
            continue

        n_obs = len(series)
        print(f"  Reconstructed series: {n_obs} obs")

        if n_obs < 30:
            print(f"  SKIP — too few observations ({n_obs})")
            manifest_rows.append({"ticker": stem, "status": "too_few_obs",
                                  "n_obs": n_obs, "nll": np.nan,
                                  "alpha": np.nan, "beta": np.nan,
                                  "alpha_plus_beta": np.nan, "nu": np.nan})
            continue

        y      = series["close"].values
        o      = series["open"].values
        h      = series["high"].values
        l_     = series["low"].values
        Q_var  = np.column_stack([o ** 2, (h - l_) ** 2])

        try:
            result = fit_garch_x_lightweight(
                y=y, Q_var=Q_var,
                max_iter=args.max_iter,
                print_every=args.print_every,
                ticker=stem,
            )
        except Exception as exc:
            print(f"  SKIP — optimization failed: {exc}")
            manifest_rows.append({"ticker": stem, "status": "optimization_failed",
                                  "n_obs": n_obs, "nll": np.nan,
                                  "alpha": np.nan, "beta": np.nan,
                                  "alpha_plus_beta": np.nan, "nu": np.nan})
            continue

        mu_, phi_, gamma_, omega_, alpha_, beta_, delta_, nu_ = unpack_params(
            result.x, k_mean=0, k_var=2
        )
        print(f"  nll={result.fun:.4f}  alpha={alpha_:.4f}  beta={beta_:.4f}  "
              f"a+b={alpha_+beta_:.4f}  nu={nu_:.4f}  success={result.success}")

        # save theta
        theta_path = os.path.join(args.out_dir, f"{stem}_theta.npy")
        np.save(theta_path, result.x)

        # save inference table
        k_var_fit = Q_var.shape[1]
        table     = inference_table(result, k_mean=0, k_var=k_var_fit)
        table_path = os.path.join(args.out_dir, f"{stem}_inference.csv")
        table.to_csv(table_path, index=False)

        manifest_rows.append({
            "ticker":          stem,
            "status":          "ok" if result.success else "converged_with_warning",
            "n_obs":           n_obs,
            "n_windows":       n_windows,
            "nll":             result.fun,
            "alpha":           alpha_,
            "beta":            beta_,
            "alpha_plus_beta": alpha_ + beta_,
            "nu":              nu_,
            "omega":           omega_,
            "delta_0":         delta_[0],
            "delta_1":         delta_[1],
        })

    # save manifest
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nDone. {len(tickers)} tickers processed.")
    print(f"Results saved to: {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    ok = (manifest["status"] == "ok").sum()
    warn = (manifest["status"] == "converged_with_warning").sum()
    skip = len(manifest) - ok - warn
    print(f"  ok={ok}  warnings={warn}  skipped={skip}")


if __name__ == "__main__":
    main()
