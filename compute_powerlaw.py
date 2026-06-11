"""
compute_powerlaw.py
-------------------
Fit a power-law tail index (Hill / MLE estimator with optimal xmin via KS) on
generated Close log-returns.

SPLIT controls which generated split(s) to load and pool:
  - "val"  : val_generated_close.csv only
  - "both" : train + val generated splits, pooled

Only the positive tail is fitted.

Output
------
  - printed table for the generated series
  - CSV saved to OUTPUT_DIR: powerlaw_results_<appendix>.csv
"""

# ── USER INPUTS ───────────────────────────────────────────────────────────────
CHECKPOINT_STEM = (
    "csv10000_samples5_steps300_seed50_20260608_115703"
)  # sub-folder under GEN_DIR produced by generate_samples.py

GEN_DIR  = "../Master-Thesis/data/generated/final/edm"   # base output directory of generate_samples.py
SPLIT    = "val"   # "val" | "both"

# Image / results output
OUTPUT_DIR = "../Master-Thesis/images/final/edm/reference_vs_generated"
appendix   = "rep_Pm-1.4_Ps1.8_NFE_300_val"  # appended to output filenames
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "../Master-Thesis")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


# ── Data loading helpers (identical to notebook) ─────────────────────────────

def load_split(split, ckpt_dir):
    gen_path  = os.path.join(ckpt_dir, f"{split}_generated_close.csv")
    ohlc_path = os.path.join(ckpt_dir, f"{split}_gt_ohlc.csv")
    assert os.path.isfile(gen_path),  f"Not found: {gen_path}"
    assert os.path.isfile(ohlc_path), f"Not found: {ohlc_path}"
    df_gen  = pd.read_csv(gen_path)
    df_ohlc = pd.read_csv(ohlc_path)
    step_cols = [c for c in df_gen.columns if c.startswith("step_")]
    print(f"[{split}] windows={df_gen['window_idx'].nunique()}  "
          f"samples={df_gen['sample_idx'].nunique()}  seq_len={len(step_cols)}")
    return df_gen, df_ohlc, step_cols


def parse_arrays(df_gen, df_ohlc, step_cols):
    window_ids = sorted(df_gen["window_idx"].unique())
    n_windows  = len(window_ids)
    n_samples  = df_gen["sample_idx"].nunique()
    seq_len    = len(step_cols)
    gen_3d = np.zeros((n_windows, n_samples, seq_len), dtype=np.float32)
    for wi, wid in enumerate(window_ids):
        sub = df_gen[df_gen["window_idx"] == wid].sort_values("sample_idx")
        gen_3d[wi] = sub[step_cols].to_numpy(dtype=np.float32)
    def gt_feat(feat_name):
        sub = df_ohlc[df_ohlc["feature"] == feat_name].sort_values("window_idx")
        return sub[step_cols].to_numpy(dtype=np.float32)
    return gen_3d, gt_feat("close"), gt_feat("open"), gt_feat("high"), gt_feat("low"), window_ids


# ── Power-law fitting ─────────────────────────────────────────────────────────

def fit_powerlaw_fast(returns, max_sample=1_000_000, seed=42):
    x = np.abs(np.asarray(returns, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]
    n_total = len(x)

    if n_total > max_sample:
        x = np.random.default_rng(seed).choice(x, size=max_sample, replace=False)

    x = np.sort(x)
    n = len(x)
    log_x = np.log(x)

    # ── Hill estimator for every candidate xmin = x[i], all at once ──────────
    # alpha[i] = 1 + (n-i) / sum_{j>=i} log(x[j] / x[i])
    rev_cumlog  = np.cumsum(log_x[::-1])[::-1]
    n_tail_arr  = np.arange(n, 0, -1, dtype=np.float64)   # n - i per candidate
    sum_log_r   = rev_cumlog - n_tail_arr * log_x          # sum log(x[j]/x[i]) ∀j≥i
    valid       = sum_log_r > 1e-12
    safe_denom = np.where(valid, sum_log_r, 1.0)          # replace invalid zeros
    alpha_arr  = np.where(valid, 1.0 + n_tail_arr / safe_denom, np.inf)

    # ── KS statistic per candidate — vectorised inner body ────────────────────
    ks_arr = np.full(n, np.inf)
    for i in range(n - 1):
        if not valid[i]:
            continue
        m      = n - i
        a      = alpha_arr[i]
        # Pareto CDF:       F(t) = 1 - (xmin/t)^(alpha-1)
        cdf_th = 1.0 - (x[i] / x[i:]) ** (a - 1.0)
        # Empirical CDF at each sorted tail point
        cdf_em = np.arange(1, m + 1, dtype=np.float64) / m
        ks_arr[i] = np.max(np.abs(cdf_em - cdf_th))

    best = int(np.argmin(ks_arr))
    return {
        "alpha"    : float(alpha_arr[best]),
        "alpha_mle": float(alpha_arr[best]),   # identical to alpha — Hill IS MLE
        "xmin"     : float(x[best]),
        "ks"       : float(ks_arr[best]),
        "n_total"  : len(x),
        "n_tail"   : n - best,
    }


# ── Load generated data ───────────────────────────────────────────────────────

ckpt_dir = os.path.join(GEN_DIR, CHECKPOINT_STEM)
assert os.path.isdir(ckpt_dir), f"Directory not found: {ckpt_dir}"

splits_to_run = ["train", "val"] if SPLIT == "both" else [SPLIT]
parsed_gen = {}
for split in splits_to_run:
    try:
        df_gen, df_ohlc, step_cols = load_split(split, ckpt_dir)
        parsed_gen[split] = parse_arrays(df_gen, df_ohlc, step_cols)
    except AssertionError as e:
        print(f"Skipping {split}: {e}")

gen_3d_list = []
for g3d, *_ in parsed_gen.values():
    gen_3d_list.append(g3d)

gen_3d_all = np.concatenate(gen_3d_list, axis=0)   # (N_gen, n_samples, L)

n_gen_windows, n_samples, seq_len = gen_3d_all.shape
gen_flat = gen_3d_all.ravel()   # normalized generated log-returns

print(f"\nPooled generated: gen_3d_all={gen_3d_all.shape}")


# ── Fit power law — positive tail only ───────────────────────────────────────

def run_fits(flat_arr, label):
    pos_tail = flat_arr[flat_arr > 0]
    results = []
    print(f"\nFitting {label} — positive tail ({len(pos_tail):,} obs) …")
    res = fit_powerlaw_fast(pos_tail)
    res.update({"series": label, "tail": "positive"})
    print(f"  alpha={res['alpha']:.4f}  xmin={res['xmin']:.6f}  "
          f"ks={res['ks']:.4f}  n_tail={res['n_tail']:,} / {res['n_total']:,}")
    results.append(res)
    return results


all_results = []
all_results.extend(run_fits(gen_flat, "Generated"))

# ── Save CSV ──────────────────────────────────────────────────────────────────

df_out = pd.DataFrame(all_results)[
    ["series", "tail", "alpha", "alpha_mle", "xmin", "ks", "n_total", "n_tail"]
]
df_out.insert(0, "split", SPLIT)

out_csv = os.path.join(OUTPUT_DIR, f"powerlaw_results_{appendix}.csv")
df_out.to_csv(out_csv, index=False)
print(f"\nSaved → {out_csv}")

print("\n" + "=" * 70)
print("  POWER-LAW RESULTS")
print("=" * 70)
print(df_out.to_string(index=False))
