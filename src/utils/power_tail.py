"""
Power-tail (Pareto alpha) estimation for reference vs generated log-returns.
Uses the same fit_powerlaw_fast() approach as Training_vs_Generated_Analysis.ipynb.

Two directory layouts are supported and auto-detected:

  Unconditional / replication layout
    GEN_DIRECTORY contains plain CSVs with a 'log_adj_close' column or step_* columns.
    REF_DIRECTORY contains per-stock CSVs with a 'log_adj_close' column.
    Loaders: load_ref_full / load_gen

  Conditional (predict_close) layout
    GEN_DIRECTORY contains train_generated_close.csv / val_generated_close.csv
    (step_* columns, one generated Close path per row) and train_gt_ohlc.csv /
    val_gt_ohlc.csv (same step_* columns plus a 'feature' column; close rows = GT).
    REF_DIRECTORY is not used in this mode.
    Loaders: load_ref_conditional / load_gen_conditional
"""

# ── USER INPUTS ───────────────────────────────────────────────────────────────
REF_DIRECTORY = "data/SNP500_individual_normalized_replication"
GEN_DIRECTORY = "data/generated/final/edm/csv20000_samples5_steps200_seed50_Pm-1.4_Ps1.8_20260604_235823"
SEED          = 50
MAX_SAMPLE    = 300_000   # subsample cap for fit_powerlaw_fast
# ─────────────────────────────────────────────────────────────────────────────

import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

_STEP_RE = re.compile(r"^step_\d+$")


def fit_powerlaw_fast(returns, max_sample=MAX_SAMPLE, seed=SEED):
    x = np.abs(np.asarray(returns, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]

    if len(x) > max_sample:
        x = np.random.default_rng(seed).choice(x, size=max_sample, replace=False)

    x = np.sort(x)
    n = len(x)
    log_x = np.log(x)

    rev_cumlog = np.cumsum(log_x[::-1])[::-1]
    n_tail_arr = np.arange(n, 0, -1, dtype=np.float64)
    sum_log_r  = rev_cumlog - n_tail_arr * log_x
    valid      = sum_log_r > 1e-12
    safe_denom = np.where(valid, sum_log_r, 1.0)
    alpha_arr  = np.where(valid, 1.0 + n_tail_arr / safe_denom, np.inf)

    ks_arr = np.full(n, np.inf)
    for i in range(n - 1):
        if not valid[i]:
            continue
        m      = n - i
        a      = alpha_arr[i]
        cdf_th = 1.0 - (x[i] / x[i:]) ** (a - 1.0)
        cdf_em = np.arange(1, m + 1, dtype=np.float64) / m
        ks_arr[i] = np.max(np.abs(cdf_em - cdf_th))

    best = int(np.argmin(ks_arr))
    return {
        "alpha"  : float(alpha_arr[best]),
        "xmin"   : float(x[best]),
        "ks"     : float(ks_arr[best]),
        "n_total": len(x),
        "n_tail" : n - best,
    }


# ── Unconditional / replication loaders ───────────────────────────────────────

def load_ref_full(directory):
    paths = []
    for fp in sorted(Path(directory).glob("*.csv")):
        series = pd.read_csv(fp)["log_adj_close"].values.astype(np.float64)
        paths.append(series)
    return paths


def load_gen(directory):
    paths = []
    for fp in sorted(Path(directory).glob("*.csv")):
        df = pd.read_csv(fp)
        if "log_adj_close" in df.columns:
            paths.append(df["log_adj_close"].values.astype(np.float64))
        else:
            step_cols = sorted(
                [c for c in df.columns if c.startswith("step_")],
                key=lambda c: int(c.split("_")[1]),
            )
            for _, row in df.iterrows():
                paths.append(row[step_cols].values.astype(np.float64))
    return paths


# ── Conditional (predict_close) loaders ───────────────────────────────────────

def _step_cols(df):
    return sorted(
        [c for c in df.columns if _STEP_RE.match(c)],
        key=lambda c: int(c.split("_")[1]),
    )


def load_gen_conditional(directory):
    """Load generated Close paths from train/val CSVs produced by predict_close generation."""
    base = Path(directory)
    paths = []
    for fn in ["train_generated_close.csv", "val_generated_close.csv"]:
        fp = base / fn
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        cols = _step_cols(df)
        for _, row in df.iterrows():
            paths.append(row[cols].values.astype(np.float64))
    return paths


def load_ref_conditional(directory):
    """Load ground-truth Close rows from train/val GT OHLC CSVs (feature == 'close')."""
    base = Path(directory)
    paths = []
    for fn in ["train_gt_ohlc.csv", "val_gt_ohlc.csv"]:
        fp = base / fn
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        df = df[df["feature"] == "close"]
        cols = _step_cols(df)
        for _, row in df.iterrows():
            paths.append(row[cols].values.astype(np.float64))
    return paths


def _is_conditional_layout(directory):
    base = Path(directory)
    return (base / "train_generated_close.csv").exists() or \
           (base / "val_generated_close.csv").exists()


if __name__ == "__main__":
    gen_dir = ROOT / GEN_DIRECTORY

    if _is_conditional_layout(gen_dir):
        print("Detected conditional (predict_close) layout")
        print(f"GEN_DIRECTORY : {GEN_DIRECTORY}\n")
        ref_paths = load_ref_conditional(gen_dir)
        gen_paths = load_gen_conditional(gen_dir)
        print(f"Reference (GT close) : {len(ref_paths)} windows loaded")
        print(f"Generated (Close)    : {len(gen_paths)} paths loaded\n")
    else:
        print("Detected unconditional / replication layout")
        print(f"REF_DIRECTORY : {REF_DIRECTORY}")
        print(f"GEN_DIRECTORY : {GEN_DIRECTORY}\n")
        ref_paths = load_ref_full(ROOT / REF_DIRECTORY)
        gen_paths = load_gen(gen_dir)
        print(f"Reference : {len(ref_paths)} full series loaded")
        print(f"Generated : {len(gen_paths)} paths loaded\n")

    for paths, label in [(ref_paths, "Reference"), (gen_paths, "Generated")]:
        incs = np.concatenate([p for p in paths])
        r = fit_powerlaw_fast(incs)
        print(f"{label}")
        print(f"  alpha    : {r['alpha']:.4f}")
        print(f"  xmin     : {r['xmin']:.6f}")
        print(f"  KS dist  : {r['ks']:.4f}")
        print(f"  n_total  : {r['n_total']:,}")
        print(f"  n_tail   : {r['n_tail']:,}")
        print()