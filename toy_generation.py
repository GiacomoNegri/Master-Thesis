import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 10000
rng = np.random.default_rng(SEED)

SEQ_LEN = 64
MULTIPLIER = 10
N_PATHS = (N // SEQ_LEN) * MULTIPLIER  # 1560

dates = pd.bdate_range(start="2000-01-03", periods=N, freq="B")
gbm_dates = pd.bdate_range(start="2000-01-03", periods=SEQ_LEN, freq="B")  # reused for every GBM path


def save(data, folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": dates, "log_adj_close": data})
    df.to_csv(path / "data.csv", index=False)
    print(f"Saved {len(df)} rows to {path / 'data.csv'}")


# 1. i.i.d. Gaussian — mean=0, std matching typical daily log-returns
gaussian = rng.normal(loc=0.0, scale=1, size=N)
save(gaussian, "./data/toy_gaussian")

# 2. AR(1): x_t = phi * x_{t-1} + eps, phi=0.9, eps ~ N(0, sigma)
phi = 0.9
sigma = 0.01
eps = rng.normal(loc=0.0, scale=sigma, size=N)
ar1 = np.zeros(N)
for t in range(1, N):
    ar1[t] = phi * ar1[t - 1] + eps[t]
save(ar1, "./data/toy_ar1")

# 3. i.i.d. Student-t — df=4 (fat tails), scaled to match std=0.01
df = 4
std = 1
# Var(t_df) = df/(df-2), scale so empirical std ~ 0.01
scale = std / np.sqrt(df / (df - 2))
studentt = rng.standard_t(df=df, size=N) * scale
save(studentt, "./data/toy_studentt")

# 4. Normalized GBM — log-price path under zero-drift GBM (pure diffusion)
# log(S_t) = sigma * W_t = cumsum of i.i.d. N(0, 1) increments
# Non-stationary: variance grows linearly with t (unlike the i.i.d. gaussian above)
# this is like a browninan motion with drift=0 and volatility=sigma, sampled at discrete time steps
gbm_dir = Path("./data/toy_gbm")
gbm_dir.mkdir(parents=True, exist_ok=True)
gbm_norm_dir = Path("./data/toy_gbm_norm")
gbm_norm_dir.mkdir(parents=True, exist_ok=True)

for i in range(N_PATHS):
    inc   = rng.normal(0.0, 1.0, size=SEQ_LEN)
    path  = np.cumsum(inc)                        # starts at inc[0], not 0
    path  = np.insert(path, 0, 0.0)[:-1]          # shift so X_0 = 0
    # equivalently: path = np.concatenate([[0], np.cumsum(inc[:-1])])
    pd.DataFrame({"date": gbm_dates, "log_adj_close": path}).to_csv(gbm_dir / f"gbm_{i:04d}.csv", index=False)

    # Path-wise normalization: zero mean, unit std per path
    path_mean = path.mean()
    path_std  = path.std()
    path_norm = (path - path_mean) / path_std if path_std > 0 else path - path_mean
    pd.DataFrame({"date": gbm_dates, "log_adj_close": path_norm}).to_csv(gbm_norm_dir / f"gbm_{i:04d}.csv", index=False)

print(f"Saved {N_PATHS} GBM paths to {gbm_dir}")
print(f"Saved {N_PATHS} normalised GBM paths to {gbm_norm_dir}")
