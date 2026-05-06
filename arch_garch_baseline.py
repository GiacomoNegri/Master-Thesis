import pandas as pd
from arch.univariate import ARX, GARCH, StudentsT

path = "./data/replication_returns_other/raw_BAX_1981-10-27_2026-04-10.csv"

df = pd.read_csv(path)

# Adjust column names if yours are different
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

returns = df["log_adj_close"].astype(float).dropna()

# Train/test split
split = int(len(returns) * 0.8)
returns_train = returns.iloc[:split]
returns_test = returns.iloc[split:]

# Scale by 100 for numerical stability in GARCH
y_train = 100.0 * returns_train

model = ARX(y_train, lags=1)
model.volatility = GARCH(p=1, o=0, q=1)
model.distribution = StudentsT()

res = model.fit(disp="off")
print(res.summary())