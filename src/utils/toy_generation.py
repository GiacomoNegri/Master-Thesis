import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 10000
rng = np.random.default_rng(SEED)

dates = pd.bdate_range(start="2000-01-03", periods=N, freq="B")


def save(data, folder):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": dates, "log_adj_close": data})
    df.to_csv(path / "data.csv", index=False)
    print(f"Saved {len(df)} rows to {path / 'data.csv'}")


# 1. i.i.d. Gaussian — mean=0, std matching typical daily log-returns
gaussian = rng.normal(loc=0.0, scale=0.01, size=N)
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
# Var(t_df) = df/(df-2), scale so empirical std ~ 0.01
scale = 0.01 / np.sqrt(df / (df - 2))
studentt = rng.standard_t(df=df, size=N) * scale
save(studentt, "./data/toy_studentt")