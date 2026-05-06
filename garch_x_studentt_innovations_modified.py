import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

path = "./data/SNP500_individual_normalized/BAX_1981-10-27_2026-05-01.csv"
START   = None   # first row to keep (0-based); None = beginning of file
SEQ_LEN = None   # number of rows to keep after START; None = all remaining rows

PRINT_EVERY = 25


def standardize_train_apply(x_train, x_test=None):
    mu = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    x_train_z = (x_train - mu) / sd

    if x_test is None:
        return x_train_z, mu, sd

    x_test_z = (x_test - mu) / sd
    return x_train_z, x_test_z, mu, sd


def student_t_logpdf_standardized(z, nu):
    """
    Log-density of standardized Student-t with Var(z)=1.
    If T ~ t_nu, then z = T * sqrt((nu-2)/nu).
    """
    if nu <= 2:
        return -np.inf * np.ones_like(z)

    # scale factor for standardized t
    # density of z = density_T(z / s) / s, where s = sqrt((nu-2)/nu)
    s = np.sqrt((nu - 2.0) / nu) #any Student-t RV with nu has variacne nu/(nu-2)
    x = z / s #but in garch we would like z_t to have unit variance, so we standardize by s: var(eps_t | F_t-1) = sig2_t, so T=z/s

    #Student-t log normalizing constant for standardized t
    log_const = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(s)
    )

    return log_const - ((nu + 1.0) / 2.0) * np.log1p((x ** 2) / nu) #final log density


def unpack_params(theta, k_mean, k_var): #our model works with a flat vector, but we have many parameters inside.
    """
    theta layout:
    [mu, phi, gamma..., omega, a_raw, b_raw, delta..., nu_raw]
    """
    # Mean params
    idx = 0

    mu = theta[idx]
    idx += 1

    phi = theta[idx]
    idx += 1

    gamma = theta[idx:idx + k_mean]
    idx += k_mean

    # Variance intercept
    omega = theta[idx]
    idx += 1

    # constrain alpha and beta to be positive and alpha + beta < 1
    raw_alpha = theta[idx]
    idx += 1

    raw_beta = theta[idx]
    idx += 1

    # Softmax like constraints, since we must have alpha > 0, beta > 0, alpha + beta < 1
    exp_a = np.exp(raw_alpha)
    exp_b = np.exp(raw_beta)
    denom = 1.0 + exp_a + exp_b

    alpha = exp_a / denom
    beta = exp_b / denom

    # Variance exogenous coefficients: delta^TQ_t
    delta = theta[idx:idx + k_var]
    idx += k_var

    # nu > 2, Student-t degrees of freedom
    nu = 2.01 + np.exp(theta[idx])

    return mu, phi, gamma, omega, alpha, beta, delta, nu


def neg_loglik_log_garch_x(theta, y, X_mean, Q_var, c=1e-8): # Core of the model with negative log-likelihood function
    """
    y: shape (T,), returns, preferably in percent units.
    X_mean: shape (T, k_mean), regressors for mean.
    Q_var: shape (T, k_var), regressors for log variance.

    Optimizer minimize:
    hat_theta = argmin_theta neg_loglik_log_garch_x(theta, y, X_mean, Q_var)
    """
    T = len(y)
    k_mean = X_mean.shape[1] if X_mean is not None else 0
    k_var = Q_var.shape[1] if Q_var is not None else 0

    if X_mean is None:
        X_mean = np.zeros((T, 0))

    if Q_var is None:
        Q_var = np.zeros((T, 0))

    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)

    #Allocate arrays for residuals and variances: eps_t, log_sig2_t, sig2_t, for all timesteps
    eps = np.zeros(T)
    log_sig2 = np.zeros(T)
    sig2 = np.zeros(T)

    # initialize variance using sample variance
    initial_var = np.var(y)
    if initial_var <= 1e-12:
        initial_var = 1.0

    sig2[0] = initial_var
    log_sig2[0] = np.log(initial_var)

    # initialize residual
    mean_0 = mu + X_mean[0] @ gamma
    eps[0] = y[0] - mean_0

    nll = 0.0

    for t in range(1, T):
        mean_t = mu + phi * y[t - 1] + X_mean[t] @ gamma
        eps[t] = y[t] - mean_t

        log_sig2[t] = (
            omega
            + alpha * np.log(eps[t - 1] ** 2 + c)
            + beta * log_sig2[t - 1]
            + Q_var[t] @ delta
        )

        # numerical guards
        log_sig2[t] = np.clip(log_sig2[t], -30.0, 30.0)
        sig2[t] = np.exp(log_sig2[t])

        z_t = eps[t] / np.sqrt(sig2[t])

        logpdf_z = student_t_logpdf_standardized(np.array([z_t]), nu)[0]

        # density of eps = density of z / sigma
        logpdf_eps = logpdf_z - 0.5 * log_sig2[t]

        if not np.isfinite(logpdf_eps):
            return 1e12

        nll -= logpdf_eps

    return nll


def build_initial_theta(y, k_mean, k_var, alpha0=0.05, beta0=0.90, nu0=8.0, delta0_value=0.0):
    # initial params guess
    mu0 = np.mean(y)
    phi0 = 0.0
    gamma0 = np.zeros(k_mean)

    omega0 = np.log(np.var(y) + 1e-8) * 0.05 #starts with no mean regressor

    # Better initialization: intended alpha=0.05, beta=0.90
    if alpha0 <= 0 or beta0 <= 0 or alpha0 + beta0 >= 1:
        raise ValueError("Initial alpha0 and beta0 must satisfy alpha0 > 0, beta0 > 0, alpha0 + beta0 < 1.")

    slack0 = 1.0 - alpha0 - beta0
    alpha_raw0 = np.log(alpha0 / slack0)
    beta_raw0 = np.log(beta0 / slack0)

    delta0 = np.full(k_var, delta0_value)

    nu_raw0 = np.log(nu0 - 2.01)

    theta0 = np.r_[
        mu0,
        phi0,
        gamma0,
        omega0,
        alpha_raw0,
        beta_raw0,
        delta0,
        nu_raw0,
    ]

    return theta0


def transformed_param_names(k_mean, k_var):
    names = ["mu", "phi"]

    for i in range(k_mean):
        names.append(f"gamma[{i}]")

    names += ["omega", "alpha", "beta", "alpha+beta"]

    for i in range(k_var):
        names.append(f"delta[{i}]")

    names.append("nu")

    return names


def transformed_param_values(theta, k_mean, k_var):
    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)

    values = [mu, phi]
    values.extend(list(gamma))
    values.extend([omega, alpha, beta, alpha + beta])
    values.extend(list(delta))
    values.append(nu)

    return np.asarray(values, dtype=float)


def numerical_jacobian_transformed(theta, k_mean, k_var, eps=1e-5):
    """
    Numerical delta-method Jacobian:
    maps raw optimizer parameters theta to transformed/interpretable parameters.
    """
    theta = np.asarray(theta, dtype=float)
    base_values = transformed_param_values(theta, k_mean, k_var)

    J = np.zeros((len(base_values), len(theta)))

    for j in range(len(theta)):
        step = eps * max(1.0, abs(theta[j]))

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[j] += step
        theta_minus[j] -= step

        values_plus = transformed_param_values(theta_plus, k_mean, k_var)
        values_minus = transformed_param_values(theta_minus, k_mean, k_var)

        J[:, j] = (values_plus - values_minus) / (2.0 * step)

    return J


def covariance_from_lbfgsb_result(result):
    """
    Extract approximate covariance matrix from L-BFGS-B inverse Hessian.
    This is approximate. For thesis-level inference, report it as approximate.
    """
    try:
        hess_inv = result.hess_inv

        if hasattr(hess_inv, "todense"):
            cov_raw = np.asarray(hess_inv.todense())
        else:
            cov_raw = np.asarray(hess_inv)

        return cov_raw

    except Exception as exc:
        print("Could not extract inverse Hessian:", exc)
        return None


def inference_table(result, k_mean, k_var):
    """
    Approximate standard errors, z-stats, and p-values for transformed parameters.
    Uses inverse Hessian approximation + numerical delta method.
    """
    names = transformed_param_names(k_mean, k_var)
    values = transformed_param_values(result.x, k_mean, k_var)

    cov_raw = covariance_from_lbfgsb_result(result)

    if cov_raw is None:
        table = pd.DataFrame(
            {
                "param": names,
                "estimate": values,
                "std_err": np.nan,
                "z": np.nan,
                "p_value": np.nan,
            }
        )
        return table

    J = numerical_jacobian_transformed(result.x, k_mean, k_var)
    cov_transformed = J @ cov_raw @ J.T

    variances = np.diag(cov_transformed)
    variances = np.where(variances < 0, np.nan, variances)

    std_err = np.sqrt(variances)
    z_stats = values / std_err
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_stats)))

    table = pd.DataFrame(
        {
            "param": names,
            "estimate": values,
            "std_err": std_err,
            "z": z_stats,
            "p_value": p_values,
        }
    )

    return table


def aic_bic(result, n_obs):
    k = len(result.x)
    nll = result.fun

    aic = 2.0 * k + 2.0 * nll
    bic = np.log(n_obs) * k + 2.0 * nll

    return aic, bic


def fit_log_garch_x(y, X_mean=None, Q_var=None):
    y = np.asarray(y, dtype=float)

    if X_mean is None:
        X_mean = np.zeros((len(y), 0))
    else:
        X_mean = np.asarray(X_mean, dtype=float)

    if Q_var is None:
        Q_var = np.zeros((len(y), 0))
    else:
        Q_var = np.asarray(Q_var, dtype=float)

    k_mean = X_mean.shape[1]
    k_var = Q_var.shape[1]

    # Different initialization
    init_grid = [
        {"alpha0": 0.03, "beta0": 0.90, "nu0": 6.0,  "delta0_value": 0.00},
        {"alpha0": 0.05, "beta0": 0.90, "nu0": 8.0,  "delta0_value": 0.00},
        {"alpha0": 0.08, "beta0": 0.85, "nu0": 10.0, "delta0_value": 0.00},
        {"alpha0": 0.10, "beta0": 0.80, "nu0": 12.0, "delta0_value": 0.00},
        {"alpha0": 0.05, "beta0": 0.90, "nu0": 8.0,  "delta0_value": 0.05},
        {"alpha0": 0.05, "beta0": 0.90, "nu0": 8.0,  "delta0_value": -0.05},
    ]

    best_result = None
    all_results = []

    print("\nStarting optimization")
    print("T:", len(y))
    print("k_mean:", k_mean)
    print("k_var:", k_var)

    for run_id, init in enumerate(init_grid, start=1):
        theta0 = build_initial_theta(
            y=y,
            k_mean=k_mean,
            k_var=k_var,
            alpha0=init["alpha0"],
            beta0=init["beta0"],
            nu0=init["nu0"],
            delta0_value=init["delta0_value"],
        )

        initial_nll = neg_loglik_log_garch_x(theta0, y, X_mean, Q_var)

        print("\nInitialization run:", run_id)
        print("Initial alpha0:", init["alpha0"])
        print("Initial beta0:", init["beta0"])
        print("Initial nu0:", init["nu0"])
        print("Initial delta0_value:", init["delta0_value"])
        print("Initial negative log-likelihood:", initial_nll)

        state = {"iter": 0, "best_nll": np.inf}

        def callback(theta_current):
            state["iter"] += 1

            if state["iter"] % PRINT_EVERY == 0:
                current_nll = neg_loglik_log_garch_x(theta_current, y, X_mean, Q_var)
                state["best_nll"] = min(state["best_nll"], current_nll)

                mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(
                    theta_current,
                    k_mean,
                    k_var
                )

                print(
                    f"run={run_id:2d} | "
                    f"iter={state['iter']:5d} | "
                    f"nll={current_nll:,.4f} | "
                    f"best={state['best_nll']:,.4f} | "
                    f"alpha={alpha:.4f} | "
                    f"beta={beta:.4f} | "
                    f"a+b={alpha + beta:.4f} | "
                    f"nu={nu:.4f} | "
                    f"delta={delta}"
                )

        result = minimize(
            neg_loglik_log_garch_x,
            theta0,
            args=(y, X_mean, Q_var),
            method="L-BFGS-B",
            options={"maxiter": 5000}, #it stops when convergence rules are met
            callback=callback
        )

        all_results.append(result)

        print("Run success:", result.success)
        print("Run message:", result.message)
        print("Run negative log-likelihood:", result.fun)

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    print("\nBest run selected")
    print("Best negative log-likelihood:", best_result.fun)

    best_result.all_results = all_results

    return best_result

def fit_log_garch_x_from_theta0(y, X_mean=None, Q_var=None, theta0=None, maxiter=1000):
    y = np.asarray(y, dtype=float)

    if X_mean is None:
        X_mean = np.zeros((len(y), 0))
    else:
        X_mean = np.asarray(X_mean, dtype=float)

    if Q_var is None:
        Q_var = np.zeros((len(y), 0))
    else:
        Q_var = np.asarray(Q_var, dtype=float)

    if theta0 is None:
        theta0 = build_initial_theta(
            y=y,
            k_mean=X_mean.shape[1],
            k_var=Q_var.shape[1],
        )

    result = minimize(
        neg_loglik_log_garch_x,
        theta0,
        args=(y, X_mean, Q_var),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    return result

# CONSTUCTING Q_t
def construct_Q(path, start=None, seq_len=None):

    print("Loading:", path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    if start is None and seq_len is None:
        print(f"No row filter applied — retaining all {len(df)} rows")
    else:
        s = start or 0
        e = (s + seq_len) if seq_len is not None else len(df)
        df = df.iloc[s:e]
        print(f"Filtered to rows [{s}:{e}]  ({len(df)} rows remaining)")

    print("Loaded rows:", len(df))
    print("Columns:", list(df.columns))

    # for GARCH-X the columns are the following, in this exact order:
    # date, close, open, high, low
    required_cols = ["close", "open", "high", "low"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}. Available columns: {list(df.columns)}"
        )

    # Example: these columns must exist or be computed before
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    r = df["close"].astype(float)

    # Build Q_raw from the unnormalized transformed variables
    gap2 = o ** 2
    range2 = (h - l) ** 2

    Q_raw = pd.DataFrame(
        {
            "gap2": gap2,
            "range2": range2,
        },
        index=df.index,
    )

    # Align everything and drop missing rows together
    data = pd.concat(
        [r.rename("return"), Q_raw],
        axis=1
    ).dropna()

    returns = data["return"]
    Q_raw = data[["gap2", "range2"]]

    split = int(len(data) * 0.8)

    returns_train = returns.iloc[:split]
    returns_test = returns.iloc[split:]

    Q_raw_train = Q_raw.iloc[:split]
    Q_raw_test = Q_raw.iloc[split:]

    Q_train, Q_test, q_mu, q_sd = standardize_train_apply(
        Q_raw_train.values,
        Q_raw_test.values
    )

    print("After dropna rows:", len(data))
    print("Train rows:", len(returns_train))
    print("Test rows:", len(returns_test))
    print("Q_train shape:", Q_train.shape)
    print("Q_test shape:", Q_test.shape)
    print("Q means used for standardization:", q_mu)
    print("Q stds used for standardization:", q_sd)

    return returns_train, returns_test, Q_train, Q_test, q_mu, q_sd

def compute_fitted_paths(theta, y, X_mean, Q_var, c=1e-8):
    """
    Compute fitted mean, residuals, log variance, variance, sigma, and standardized residuals.
    """
    y = np.asarray(y, dtype=float)

    if X_mean is None:
        X_mean = np.zeros((len(y), 0))
    else:
        X_mean = np.asarray(X_mean, dtype=float)

    if Q_var is None:
        Q_var = np.zeros((len(y), 0))
    else:
        Q_var = np.asarray(Q_var, dtype=float)

    T = len(y)
    k_mean = X_mean.shape[1]
    k_var = Q_var.shape[1]

    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)

    mean = np.zeros(T)
    eps = np.zeros(T)
    log_sig2 = np.zeros(T)
    sig2 = np.zeros(T)

    initial_var = np.var(y)
    if initial_var <= 1e-12:
        initial_var = 1.0

    sig2[0] = initial_var
    log_sig2[0] = np.log(initial_var)

    mean[0] = mu + X_mean[0] @ gamma
    eps[0] = y[0] - mean[0]

    for t in range(1, T):
        mean[t] = mu + phi * y[t - 1] + X_mean[t] @ gamma
        eps[t] = y[t] - mean[t]

        log_sig2[t] = (
            omega
            + alpha * np.log(eps[t - 1] ** 2 + c)
            + beta * log_sig2[t - 1]
            + Q_var[t] @ delta
        )

        log_sig2[t] = np.clip(log_sig2[t], -30.0, 30.0)
        sig2[t] = np.exp(log_sig2[t])

    sigma = np.sqrt(sig2)
    z = eps / sigma

    return {
        "mean": mean,
        "eps": eps,
        "log_sig2": log_sig2,
        "sig2": sig2,
        "sigma": sigma,
        "z": z,
    }


def simulate_bootstrap_series(theta, y, X_mean, Q_var, z_boot, c=1e-8):
    """
    Simulate a bootstrap return series using fixed X_mean/Q_var and resampled standardized shocks.
    """
    y = np.asarray(y, dtype=float)

    if X_mean is None:
        X_mean = np.zeros((len(y), 0))
    else:
        X_mean = np.asarray(X_mean, dtype=float)

    if Q_var is None:
        Q_var = np.zeros((len(y), 0))
    else:
        Q_var = np.asarray(Q_var, dtype=float)

    T = len(y)
    k_mean = X_mean.shape[1]
    k_var = Q_var.shape[1]

    mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(theta, k_mean, k_var)

    y_sim = np.zeros(T)
    eps_sim = np.zeros(T)
    log_sig2_sim = np.zeros(T)
    sig2_sim = np.zeros(T)

    initial_var = np.var(y)
    if initial_var <= 1e-12:
        initial_var = 1.0

    y_sim[0] = y[0]
    sig2_sim[0] = initial_var
    log_sig2_sim[0] = np.log(initial_var)

    mean_0 = mu + X_mean[0] @ gamma
    eps_sim[0] = y_sim[0] - mean_0

    for t in range(1, T):
        log_sig2_sim[t] = (
            omega
            + alpha * np.log(eps_sim[t - 1] ** 2 + c)
            + beta * log_sig2_sim[t - 1]
            + Q_var[t] @ delta
        )

        log_sig2_sim[t] = np.clip(log_sig2_sim[t], -30.0, 30.0)
        sig2_sim[t] = np.exp(log_sig2_sim[t])

        mean_t = mu + phi * y_sim[t - 1] + X_mean[t] @ gamma
        eps_sim[t] = np.sqrt(sig2_sim[t]) * z_boot[t]
        y_sim[t] = mean_t + eps_sim[t]

    return y_sim


def bootstrap_log_garch_x(result, y, X_mean, Q_var, n_boot=100, seed=123):
    """
    Residual bootstrap for reliability measures.
    Returns bootstrap estimates, bootstrap SEs, and percentile confidence intervals.
    """
    rng = np.random.default_rng(seed)

    if X_mean is None:
        X_mean_arr = np.zeros((len(y), 0))
    else:
        X_mean_arr = np.asarray(X_mean, dtype=float)

    if Q_var is None:
        Q_var_arr = np.zeros((len(y), 0))
    else:
        Q_var_arr = np.asarray(Q_var, dtype=float)

    k_mean = X_mean_arr.shape[1]
    k_var = Q_var_arr.shape[1]

    fitted = compute_fitted_paths(result.x, y, X_mean_arr, Q_var_arr)

    # remove first observation because it is initialized
    z_hat = fitted["z"][1:]
    z_hat = z_hat[np.isfinite(z_hat)]

    estimates = []

    for b in range(n_boot):
        z_boot = np.empty(len(y))
        z_boot[0] = 0.0
        z_boot[1:] = rng.choice(z_hat, size=len(y) - 1, replace=True)

        y_boot = simulate_bootstrap_series(
            theta=result.x,
            y=y,
            X_mean=X_mean_arr,
            Q_var=Q_var_arr,
            z_boot=z_boot,
        )

        boot_fit = fit_log_garch_x_from_theta0(
            y=y_boot,
            X_mean=X_mean_arr,
            Q_var=Q_var_arr,
            theta0=result.x,
            maxiter=1000,
        )

        if boot_fit.success:
            estimates.append(
                transformed_param_values(boot_fit.x, k_mean, k_var)
            )

        if (b + 1) % 10 == 0:
            print(f"Bootstrap completed: {b + 1}/{n_boot}")

    estimates = np.asarray(estimates)

    names = transformed_param_names(k_mean, k_var)
    point_est = transformed_param_values(result.x, k_mean, k_var)

    se = np.std(estimates, axis=0, ddof=1)
    ci_low = np.percentile(estimates, 2.5, axis=0)
    ci_high = np.percentile(estimates, 97.5, axis=0)

    table = pd.DataFrame(
        {
            "param": names,
            "estimate": point_est,
            "boot_se": se,
            "ci_2.5": ci_low,
            "ci_97.5": ci_high,
        }
    )

    return table, estimates

# RUNNING THE FULL PIPELINE GIVEN THE PATH AS INPUT
returns_train, returns_test, Q_train, Q_test, q_mu, q_sd = construct_Q(path, start=START, seq_len=SEQ_LEN)

# Scale by 100 for numerical stability in GARCH
y_train = returns_train.values #* 100.0

print("\nReturn diagnostics")
print("y_train mean:", np.mean(y_train))
print("y_train std:", np.std(y_train, ddof=1))
print("y_train min:", np.min(y_train))
print("y_train max:", np.max(y_train))

result = fit_log_garch_x(
    y=y_train,
    X_mean=None,
    Q_var=Q_train
)

print("Optimization success:", result.success)
print("Optimizer message:", result.message)
print("Negative log-likelihood:", result.fun)

stem = os.path.splitext(os.path.basename(path))[0]
os.makedirs("./garch_parameters", exist_ok=True)
np.save(f"./garch_parameters/{stem}_theta.npy", result.x)
np.save(f"./garch_parameters/{stem}_q_stats.npy", np.stack([q_mu, q_sd]))
print(f"Saved parameters to ./garch_parameters/{stem}_*.npy")

mu, phi, gamma, omega, alpha, beta, delta, nu = unpack_params(
    result.x,
    k_mean=0,
    k_var=Q_train.shape[1]
)

print("\nEstimated parameters")
print("mu:", mu)
print("phi:", phi)
print("omega:", omega)
print("alpha:", alpha)
print("beta:", beta)
print("alpha + beta:", alpha + beta)
print("delta:", delta)
print("nu:", nu)

# AIC/BIC
aic, bic = aic_bic(result, n_obs=len(y_train))

print("\nInformation criteria")
print("AIC:", aic)
print("BIC:", bic)

# Standard errors and p-values
table = inference_table(
    result,
    k_mean=0,
    k_var=Q_train.shape[1]
)

print("\nApproximate inference table")
print(table.to_string(index=False))


boot_table, boot_estimates = bootstrap_log_garch_x(
    result=result,
    y=y_train,
    X_mean=None,
    Q_var=Q_train,
    n_boot=100,
    seed=123,
)

print("\nBootstrap reliability table")
print(boot_table.to_string(index=False))

result_no_x = fit_log_garch_x(
    y=y_train,
    X_mean=None,
    Q_var=None
)

aic_x, bic_x = aic_bic(result, len(y_train))
aic_no_x, bic_no_x = aic_bic(result_no_x, len(y_train))

print("\nModel comparison")
print("GARCH-X NLL:", result.fun)
print("No-X NLL:", result_no_x.fun)
print("GARCH-X AIC:", aic_x)
print("No-X AIC:", aic_no_x)
print("GARCH-X BIC:", bic_x)
print("No-X BIC:", bic_no_x)

from scipy.stats import chi2

lr_stat = 2.0 * (result_no_x.fun - result.fun)
lr_p = 1.0 - chi2.cdf(lr_stat, df=Q_train.shape[1])

print("LR statistic:", lr_stat)
print("LR p-value:", lr_p)


fitted = compute_fitted_paths(
    result.x,
    y_train,
    X_mean=None,
    Q_var=Q_train,
)

z = fitted["z"][1:]

print("\nStandardized residual diagnostics")
print("z mean:", np.mean(z))
print("z std:", np.std(z, ddof=1))
print("z skew:", pd.Series(z).skew())
print("z kurtosis:", pd.Series(z).kurtosis())