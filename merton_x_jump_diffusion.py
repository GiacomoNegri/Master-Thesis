"""
Merton-X Jump-Diffusion Baseline
=================================
Model
-----
    r_t = mu + nu_t * Z_t + sum_{i=1}^{P_t} Y_{t,i}

    Z_t  ~ N(0, 1)                      diffusion noise
    P_t  ~ Poisson(lambda)              number of jumps on day t
    Y_{t,i} ~ N(mu_J, sigma_J^2)        each jump size

    log(nu_t^2) = a0 + a.T @ q_t        log-variance driven by intraday features

    q_t = [open^2, (high - low)^2]      already globally standardized in the CSV

Conditional density (Poisson mixture of Gaussians)
---------------------------------------------------
Summing over the number of jumps k = 0, 1, ..., Kmax:

    p(r_t | q_t) = sum_{k=0}^{Kmax} Pois(k; lambda)
                   * N(r_t ; mu + k*mu_J, nu_t^2 + k*sigma_J^2)

The k=0 term is a pure Gaussian with mean mu and variance nu_t^2 (no jump).
Each subsequent term adds one more jump contribution to both mean and variance.
logsumexp is used for numerical stability.

Parameter vector
----------------
    theta = [mu, a0, a1, ..., a_{k_var}, log_lambda, mu_J, log_sigma_J]

    lambda   = exp(log_lambda)     > 0
    sigma_J  = exp(log_sigma_J)    > 0

Jump parameters (lambda, mu_J, sigma_J) are constant across time.
Only the diffusion variance nu_t^2 is conditional on q_t.

Inputs
------
    CSV with columns: date, close, open, high, low
    close is used as y_t (already standardized globally)
    q_t = [open^2, (high - low)^2] (already standardized; used directly)
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, poisson

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

path    = "./data/SNP500_individual_normalized/BAX_1981-10-27_2026-05-01.csv"
START   = None   # first row to keep (0-based); None = beginning of file
SEQ_LEN = None   # number of rows to use after START; None = all remaining

PRINT_EVERY = 25  # print progress every N optimizer iterations
KMAX        = 30  # truncation of the Poisson sum (covers lambda up to ~15), because poisson is infinite


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

def unpack_params(theta, k_var):
    """
    Extract interpretable parameters from the flat optimizer vector.

    Layout:  [mu, a0, a1, ..., a_{k_var}, log_lambda, mu_J, log_sigma_J]

    Returns
    -------
    mu        : float   — unconditional mean of r_t
    a0        : float   — log-variance intercept
    a         : array   — log-variance coefficients for q_t, shape (k_var,)
    lam       : float   — Poisson jump intensity (strictly positive)
    mu_J      : float   — mean jump size
    sigma_J   : float   — jump-size std (strictly positive)
    """
    idx = 0

    mu  = theta[idx];  idx += 1
    a0  = theta[idx];  idx += 1
    a   = theta[idx : idx + k_var];  idx += k_var

    # unconstrained -> constrained transforms
    lam     = np.exp(theta[idx]);  idx += 1   # lambda > 0
    mu_J    = theta[idx];          idx += 1   # unrestricted
    sigma_J = np.exp(theta[idx]);  idx += 1   # sigma_J > 0

    return mu, a0, a, lam, mu_J, sigma_J


def pack_params(mu, a0, a, lam, mu_J, sigma_J):
    """Inverse of unpack_params: build the raw optimizer vector."""
    return np.r_[mu, a0, a, np.log(lam), mu_J, np.log(sigma_J)]


# ---------------------------------------------------------------------------
# Conditional diffusion variance
# ---------------------------------------------------------------------------

def compute_log_nu2(a0, a, Q):
    """
    Compute log(nu_t^2) = a0 + a.T @ q_t for every t.

    Parameters
    ----------
    a0 : float
    a  : array (k_var,)
    Q  : array (T, k_var)  — intraday features, globally standardized

    Returns
    -------
    log_nu2 : array (T,)
    """
    log_nu2 = a0 + Q @ a                  # shape (T,)
    log_nu2 = np.clip(log_nu2, -30, 30)   # numerical guard against overflow
    return log_nu2


# ---------------------------------------------------------------------------
# Negative log-likelihood
# ---------------------------------------------------------------------------

def neg_loglik_merton_x(theta, y, Q, kmax=KMAX):
    """
    Negative log-likelihood of the Merton-X model.

    For each observation r_t we evaluate the Poisson-mixture density:

        p(r_t | q_t) = sum_{k=0}^{kmax}  Pois(k; lambda)
                       * N(r_t; mu + k*mu_J, nu_t^2 + k*sigma_J^2)

    In log-space:
        log p(r_t | q_t) = logsumexp over k of:
            log Pois(k; lambda) + log N(r_t; mu + k*mu_J, nu_t^2 + k*sigma_J^2)

    Parameters
    ----------
    theta : 1-D array  — raw optimizer parameters
    y     : array (T,) — returns
    Q     : array (T, k_var) — intraday features
    kmax  : int — Poisson sum truncation

    Returns
    -------
    nll : float — negative log-likelihood (to be minimized)
    """
    k_var = Q.shape[1]
    mu, a0, a, lam, mu_J, sigma_J = unpack_params(theta, k_var)

    T = len(y)

    # --- diffusion variance at each time step ---
    log_nu2 = compute_log_nu2(a0, a, Q)   # (T,)
    nu2     = np.exp(log_nu2)             # (T,)

    # --- precompute Poisson log-weights once (same for all t) ---
    # log P(P_t = k) = k*log(lambda) - lambda - log(k!)
    k_vals     = np.arange(kmax + 1)                   # (K+1,)
    log_pois_w = poisson.logpmf(k_vals, lam)            # (K+1,)

    # --- evaluate mixture density for each t ---
    nll = 0.0
    for t in range(T):
        # mixture means and variances for each k
        mix_means = mu + k_vals * mu_J                 # (K+1,)
        mix_vars  = nu2[t] + k_vals * (sigma_J ** 2)  # (K+1,)

        # log N(r_t; mix_mean_k, mix_var_k) for each k
        log_gauss = -0.5 * np.log(2 * np.pi * mix_vars) \
                    - 0.5 * ((y[t] - mix_means) ** 2) / mix_vars   # (K+1,)

        # log[ sum_k Pois(k)*Gauss_k ] via logsumexp
        log_p_t = logsumexp(log_pois_w + log_gauss)

        if not np.isfinite(log_p_t):
            return 1e12   # guard: return large value if density is degenerate

        nll -= log_p_t

    return nll


# ---------------------------------------------------------------------------
# Vectorized NLL (faster for large T)
# ---------------------------------------------------------------------------

def neg_loglik_merton_x_vec(theta, y, Q, kmax=KMAX):
    """
    Vectorized version of neg_loglik_merton_x.
    Avoids the Python loop over t by broadcasting over (T, K+1).

    Memory cost: O(T * kmax) — feasible for typical daily return series.
    """
    k_var = Q.shape[1]
    mu, a0, a, lam, mu_J, sigma_J = unpack_params(theta, k_var)

    T = len(y)

    log_nu2 = compute_log_nu2(a0, a, Q)   # (T,)
    nu2     = np.exp(log_nu2)             # (T,)

    k_vals     = np.arange(kmax + 1)      # (K+1,)
    log_pois_w = poisson.logpmf(k_vals, lam)  # (K+1,)

    # Broadcast: (T, 1) and (1, K+1) -> (T, K+1)
    mix_means = mu + k_vals[None, :] * mu_J                         # (T, K+1)
    mix_vars  = nu2[:, None] + k_vals[None, :] * (sigma_J ** 2)    # (T, K+1)

    # log Normal density for each (t, k)
    log_gauss = (
        -0.5 * np.log(2 * np.pi * mix_vars)
        - 0.5 * ((y[:, None] - mix_means) ** 2) / mix_vars
    )  # (T, K+1)

    # log mixture density per t: logsumexp over k dimension
    log_p = logsumexp(log_pois_w[None, :] + log_gauss, axis=1)  # (T,)

    if not np.all(np.isfinite(log_p)):
        return 1e12

    return -np.sum(log_p)


# ---------------------------------------------------------------------------
# Initial parameter construction
# ---------------------------------------------------------------------------

def build_initial_theta(y, k_var, lam0=0.1, mu_J0=0.0, sigma_J0=0.01, a_val=0.0):
    """
    Sensible starting point for the optimizer.

    Strategy:
    - mu     <- sample mean of y
    - a0     <- log(sample variance of y), so nu_t^2 ~ sample variance when a=0
    - a      <- zeros (no q_t contribution initially)
    - lam0   <- small positive number (few jumps per day for daily returns)
    - mu_J0  <- 0 (symmetric jumps)
    - sigma_J0 <- small relative to return std (jumps are a correction, not
                  the main driver at initialization)
    """
    mu0  = float(np.mean(y))
    # initial log-variance: entire variance attributed to diffusion
    a0_0 = float(np.log(np.var(y) + 1e-8))
    a_0  = np.full(k_var, a_val)

    theta0 = pack_params(
        mu      = mu0,
        a0      = a0_0,
        a       = a_0,
        lam     = lam0,
        mu_J    = mu_J0,
        sigma_J = sigma_J0,
    )
    return theta0


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def transformed_param_names(k_var):
    names = ["mu", "a0"]
    for i in range(k_var):
        names.append(f"a[{i}]")
    names += ["lambda", "mu_J", "sigma_J"]
    return names


def transformed_param_values(theta, k_var):
    mu, a0, a, lam, mu_J, sigma_J = unpack_params(theta, k_var)
    values = [mu, a0] + list(a) + [lam, mu_J, sigma_J]
    return np.asarray(values, dtype=float)


def numerical_jacobian(theta, k_var, eps=1e-5):
    """
    Numerical delta-method Jacobian: d(transformed_params) / d(theta_raw).
    Used to propagate the inverse-Hessian covariance to interpretable params.
    """
    theta = np.asarray(theta, dtype=float)
    base  = transformed_param_values(theta, k_var)
    J     = np.zeros((len(base), len(theta)))

    for j in range(len(theta)):
        step = eps * max(1.0, abs(theta[j]))
        tp   = theta.copy();  tp[j] += step
        tm   = theta.copy();  tm[j] -= step
        J[:, j] = (transformed_param_values(tp, k_var)
                   - transformed_param_values(tm, k_var)) / (2 * step)
    return J


def covariance_from_result(result):
    """Extract approximate covariance from L-BFGS-B inverse Hessian."""
    try:
        h = result.hess_inv
        return np.asarray(h.todense() if hasattr(h, "todense") else h)
    except Exception as exc:
        print("Could not extract inverse Hessian:", exc)
        return None


def inference_table(result, k_var):
    """Approximate SE, z-stat, p-value via delta method."""
    names  = transformed_param_names(k_var)
    values = transformed_param_values(result.x, k_var)

    cov_raw = covariance_from_result(result)
    if cov_raw is None:
        return pd.DataFrame({
            "param": names, "estimate": values,
            "std_err": np.nan, "z": np.nan, "p_value": np.nan,
        })

    J               = numerical_jacobian(result.x, k_var)
    cov_transformed = J @ cov_raw @ J.T
    variances       = np.where(np.diag(cov_transformed) < 0, np.nan,
                               np.diag(cov_transformed))
    std_err  = np.sqrt(variances)
    z_stats  = values / std_err
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_stats)))

    return pd.DataFrame({
        "param": names, "estimate": values,
        "std_err": std_err, "z": z_stats, "p_value": p_values,
    })


def aic_bic(result, n_obs):
    k   = len(result.x)
    nll = result.fun
    return 2.0 * k + 2.0 * nll, np.log(n_obs) * k + 2.0 * nll


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_merton_x(y, Q):
    """
    Fit the Merton-X model to training data (y, Q).

    Multiple initializations vary lambda (the jump intensity) because the
    likelihood surface can have local minima around the jump component.
    Other parameters are initialized identically across runs.

    Returns the scipy OptimizeResult with the lowest NLL, plus an
    .all_results list containing every run's result.
    """
    y     = np.asarray(y, dtype=float)
    Q     = np.asarray(Q, dtype=float)
    k_var = Q.shape[1]

    # Grid over lambda initializations; also vary sigma_J slightly
    # mu_J0: is fixed because mean jumps are hard to identify and typically they are close to zero anyway
    # a_val: also fixed to zero for simplicity, but varied to test sensitivity to the initial diffusion variance, to avoid possibility of local minima initialization
    init_grid = [
        {"lam0": 0.05,  "mu_J0": 0.0, "sigma_J0": 0.005, "a_val":  0.0},
        {"lam0": 0.10,  "mu_J0": 0.0, "sigma_J0": 0.010, "a_val":  0.0},
        {"lam0": 0.20,  "mu_J0": 0.0, "sigma_J0": 0.020, "a_val":  0.0},
        {"lam0": 0.50,  "mu_J0": 0.0, "sigma_J0": 0.050, "a_val":  0.0},
        {"lam0": 1.00,  "mu_J0": 0.0, "sigma_J0": 0.100, "a_val":  0.0},
        {"lam0": 0.10,  "mu_J0": 0.0, "sigma_J0": 0.010, "a_val":  0.01},
        {"lam0": 0.10,  "mu_J0": 0.0, "sigma_J0": 0.010, "a_val": -0.01},
    ]

    print("\nStarting Merton-X optimization")
    print("T:", len(y))
    print("k_var:", k_var)
    print("KMAX (Poisson truncation):", KMAX)

    best_result  = None
    all_results  = []

    for run_id, init in enumerate(init_grid, start=1):
        theta0 = build_initial_theta(
            y       = y,
            k_var   = k_var,
            lam0    = init["lam0"],
            mu_J0   = init["mu_J0"],
            sigma_J0= init["sigma_J0"],
            a_val   = init["a_val"],
        )

        nll0 = neg_loglik_merton_x_vec(theta0, y, Q)

        print(f"\nRun {run_id}  |  lam0={init['lam0']}  sigma_J0={init['sigma_J0']}"
              f"  a_val={init['a_val']}  |  initial NLL={nll0:.4f}")

        state = {"iter": 0, "best_nll": np.inf}

        def callback(theta_cur, _state=state, _run=run_id, _y=y, _Q=Q):
            _state["iter"] += 1
            if _state["iter"] % PRINT_EVERY == 0:
                cur_nll = neg_loglik_merton_x_vec(theta_cur, _y, _Q)
                _state["best_nll"] = min(_state["best_nll"], cur_nll)
                _, _, _, lam, mu_J, sigma_J = unpack_params(theta_cur, _Q.shape[1])
                print(
                    f"  run={_run:2d} | iter={_state['iter']:5d} | "
                    f"nll={cur_nll:,.4f} | best={_state['best_nll']:,.4f} | "
                    f"lam={lam:.4f} | mu_J={mu_J:.4f} | sigma_J={sigma_J:.4f}"
                )

        result = minimize(
            neg_loglik_merton_x_vec,
            theta0,
            args=(y, Q),
            method="L-BFGS-B",
            options={"maxiter": 5000},
            callback=callback,
        )

        all_results.append(result)
        print(f"  success={result.success}  |  message: {result.message}")
        print(f"  final NLL={result.fun:.4f}")

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    print("\nBest run selected  |  NLL:", best_result.fun)
    best_result.all_results = all_results
    return best_result


# ---------------------------------------------------------------------------
# Generation / simulation
# ---------------------------------------------------------------------------

def simulate_merton_x(theta, Q, n_paths, seed=0):
    """
    Generate synthetic return paths conditional on Q (typically Q_test).

    Algorithm (per path, per time step t):
      1. Compute nu_t^2 = exp(a0 + a.T @ q_t)
      2. Draw P_t ~ Poisson(lambda)
      3. Draw Z_t ~ N(0, 1)
      4. Draw Y_{t,i} ~ N(mu_J, sigma_J^2) for i = 1, ..., P_t
      5. r_t = mu + nu_t * Z_t + sum_i Y_{t,i}

    Parameters
    ----------
    theta   : 1-D array  — fitted raw parameter vector
    Q       : array (T, k_var) — conditioning features (e.g. Q_test)
    n_paths : int  — number of independent sample paths
    seed    : int  — RNG seed for reproducibility

    Returns
    -------
    samples : array (n_paths, T) — simulated return paths
    """
    rng   = np.random.default_rng(seed)
    k_var = Q.shape[1]
    T     = Q.shape[0]

    mu, a0, a, lam, mu_J, sigma_J = unpack_params(theta, k_var)

    # Diffusion std at each time step (same for all paths)
    log_nu2 = compute_log_nu2(a0, a, Q)   # (T,)
    nu      = np.exp(0.5 * log_nu2)       # (T,)  — std, not variance

    samples = np.zeros((n_paths, T))

    for p in range(n_paths):
        # Diffusion component: nu_t * Z_t
        Z    = rng.standard_normal(T)          # (T,)
        diff = nu * Z                          # (T,)

        # Jump component: sum_{i=1}^{P_t} Y_{t,i}
        # Draw number of jumps per day
        P    = rng.poisson(lam, size=T)        # (T,) integer counts
        # For each t, draw P_t jump sizes and sum them
        jump = np.zeros(T)
        for t in range(T):
            if P[t] > 0:
                Y_ti   = rng.normal(mu_J, sigma_J, size=P[t])
                jump[t] = Y_ti.sum()

        samples[p] = mu + diff + jump

    return samples   # (n_paths, T)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def construct_Q(path, start=None, seq_len=None):
    """
    Load CSV, build y_t and q_t, split 80/20.

    y_t = close  (already globally standardized in the CSV)
    q_t = [open^2, (high - low)^2]  (built from standardized open/high/low)

    Note: q_t is NOT re-standardized here. The CSV values are already on a
    comparable scale because the raw OHLC features were standardized globally
    before being saved.
    """
    print("Loading:", path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    if start is None and seq_len is None:
        print(f"No row filter — retaining all {len(df)} rows")
    else:
        s = start or 0
        e = (s + seq_len) if seq_len is not None else len(df)
        df = df.iloc[s:e]
        print(f"Filtered to rows [{s}:{e}]  ({len(df)} rows remaining)")

    required = ["close", "open", "high", "low"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    r = df["close"].astype(float)

    # q_t features: gap squared and intraday range squared
    Q_raw = pd.DataFrame({
        "gap2":   o ** 2,
        "range2": (h - l) ** 2,
    }, index=df.index)

    data = pd.concat([r.rename("return"), Q_raw], axis=1).dropna()

    returns = data["return"]
    Q_df    = data[["gap2", "range2"]]

    split = int(len(data) * 0.8)

    returns_train = returns.iloc[:split]
    returns_test  = returns.iloc[split:]
    Q_train       = Q_df.iloc[:split].values
    Q_test        = Q_df.iloc[split:].values

    print(f"After dropna: {len(data)} rows  |  train: {len(returns_train)}  test: {len(returns_test)}")
    print("Q_train shape:", Q_train.shape)
    print("Q_test  shape:", Q_test.shape)

    return returns_train, returns_test, Q_train, Q_test


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

returns_train, returns_test, Q_train, Q_test = construct_Q(
    path, start=START, seq_len=SEQ_LEN
)

y_train = returns_train.values
y_test  = returns_test.values

print("\nReturn diagnostics (train)")
print("  mean:", np.mean(y_train))
print("  std: ", np.std(y_train, ddof=1))
print("  min: ", np.min(y_train))
print("  max: ", np.max(y_train))

# --- Fit ---
result = fit_merton_x(y=y_train, Q=Q_train)

k_var = Q_train.shape[1]

print("\nOptimization result")
print("  success:", result.success)
print("  message:", result.message)
print("  NLL:    ", result.fun)

# --- Summary of all runs ---
print("\nAll initialization outcomes:")
for i, r in enumerate(result.all_results, 1):
    print(f"  run {i:2d}  |  NLL={r.fun:.4f}  |  success={r.success}")

# --- Fitted parameters ---
mu, a0, a, lam, mu_J, sigma_J = unpack_params(result.x, k_var)

print("\nFitted parameters")
print("  mu:      ", mu)
print("  a0:      ", a0)
print("  a:       ", a)
print("  lambda:  ", lam)
print("  mu_J:    ", mu_J)
print("  sigma_J: ", sigma_J)

# --- Information criteria ---
aic, bic = aic_bic(result, n_obs=len(y_train))
print("\nInformation criteria")
print("  AIC:", aic)
print("  BIC:", bic)

# --- Inference table ---
table = inference_table(result, k_var)
print("\nApproximate inference table")
print(table.to_string(index=False))

# --- Save parameters ---
stem = os.path.splitext(os.path.basename(path))[0]
os.makedirs("./merton_parameters", exist_ok=True)
np.save(f"./merton_parameters/{stem}_theta.npy", result.x)
print(f"\nSaved theta to ./merton_parameters/{stem}_theta.npy")

# --- Generate synthetic paths on test set ---
N_PATHS = 100
samples = simulate_merton_x(
    theta   = result.x,
    Q       = Q_test,
    n_paths = N_PATHS,
    seed    = 42,
)

print(f"\nGenerated {N_PATHS} synthetic paths over {Q_test.shape[0]} test days")
print("  samples shape:", samples.shape)
print("  sample mean:  ", samples.mean())
print("  sample std:   ", samples.std())

np.save(f"./merton_parameters/{stem}_samples.npy", samples)
print(f"Saved samples to ./merton_parameters/{stem}_samples.npy")
