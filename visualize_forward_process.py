#!/usr/bin/env python
"""
visualize_forward_process.py  –  Forward (noising) process visualizations for EDM-CSDI.

Companion to visualize_denoising.py. Where that script runs the Heun sampler
*backwards* (sigma_max -> sigma_min) and captures the model's denoised estimate
at each step, this script runs the EDM forward corruption *forwards*
(sigma increasing) with NO model involved:

    x_sigma = x_0 + sigma * eps,     eps ~ N(0, I)

for a ground-truth Close log-return window x_0, at a sequence of monotonically
increasing sigma levels drawn from the same log-normal training schedule used
by EDMLoss (src/training/edm_loss.py): ln(sigma) ~ N(P_mean, P_std^2).

Purpose: show how the Close marginal distribution progressively flattens and
widens until it is indistinguishable from the diffusion prior N(0, sigma^2)
that the reverse sampler starts from — i.e. visually justify the EDM sampler's
"start from pure Gaussian noise at sigma_max" initialisation.

Only the Close channel is corrupted (predict_close mode: O/H/L are always
clean conditioning features and are never the diffusion target).

Design notes (see conversation / commit message for full rationale):
- Sigma levels are quantiles of the log-normal schedule, not an arbitrary grid,
  so the frames reflect noise levels actually seen during training.
- The marginal-KDE plot uses many *fresh* eps draws per sigma (statistical
  accuracy of the density estimate matters more than continuity).
- The path/price plots reuse the *same small set* of eps draws across every
  sigma (continuity matters: each line is one trajectory smoothly dissolving
  into noise as sigma grows, not an uncorrelated resample at each frame).
- Price paths are guarded against float overflow: at high sigma quantiles the
  cumulative corrupted log-return sum can be large enough that exp() overflows;
  the cumulative sum is clipped and a warning is printed when this triggers.

Frame plots are saved with zero-padded indices for easy GIF assembly:
    ffmpeg -pattern_type glob -framerate 2 -i 'figures/forward/forward_marginal_step_*.png' forward_marginal.gif

Usage:
    python visualize_forward_process.py \\
        --checkpoint_folder final/edm \\
        --checkpoint_name EDM_REPLICATION_....pt \\
        --split train --window_idx 0 \\
        --quantiles 0.01 0.05 0.25 0.5 0.75 0.95 0.999 \\
        --P_mean -1.4 --P_std 1.8 \\
        --num_kde_samples 1000 --num_path_samples 5 \\
        --norm_stats data/normalization_stats_norm_replication.csv \\
        --out_dir figures/forward --seed 42
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, norm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from visualize_denoising import (
    _save,
    _to_price_path,
    denorm,
    load_norm_stats,
    load_window,
)


# ── Plotting style (mirrors visualize_denoising.py) ──────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_COLORS = {
    "marginal":  "#9C27B0",   # purple — distinct from visualize_denoising's blue
    "prior":     "#607D8B",   # blue-grey — theoretical N(0, sigma^2) overlay
    "evolution": "#FF5722",   # deep orange
    "prices":    "#009688",   # teal
}

# exp() overflows in float64 around x ~ 709.78; keep a safety margin.
_MAX_CUMSUM_EXPONENT = 700.0


# ── Config loading (no model instantiation — forward process needs no network) ─

def load_config_only(checkpoint_folder: str, checkpoint_name: str):
    """
    Load just the checkpoint's config/metadata needed to reconstruct the
    training window (seq_len, feat_cols, close_idx). Skips building the
    CSDIModel and moving anything to a device, since the forward corruption
    process never calls the model.
    """
    ckpt_path = os.path.join("checkpoints", checkpoint_folder, checkpoint_name)
    print(f"Loading checkpoint config: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]

    seq_len   = int(config["train"]["seq_len"])
    columns   = tuple(config["data"].get("columns", ("date", "log_adj_close")))
    feat_cols = list(columns[1:])
    close_idx = feat_cols.index("close")

    print(f"  feat_cols={feat_cols}  close_idx={close_idx}  seq_len={seq_len}")
    return config, feat_cols, close_idx, seq_len


# ── Sigma schedule ────────────────────────────────────────────────────────────

def sigmas_from_quantiles(quantiles: list, P_mean: float, P_std: float) -> np.ndarray:
    """
    Map quantiles of ln(sigma) ~ N(P_mean, P_std^2) to sigma values.

    sigma(q) = exp(P_mean + P_std * Phi^-1(q))

    Using quantiles of the *training* schedule (rather than an arbitrary
    linear/log grid) means each frame corresponds to an actual noise band the
    model is trained on, weighted by how the log-normal schedule concentrates
    mass (Karras et al. 2022, Sec. 5).
    """
    q = np.sort(np.asarray(quantiles, dtype=np.float64))
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("--quantiles must all lie strictly between 0 and 1.")
    z = norm.ppf(q)
    sigmas = np.exp(P_mean + P_std * z)
    return sigmas, q


# ── Visualisation 1: Marginal distribution evolution toward the prior ───────

def plot_marginal_forward(gt_close, sigmas, quantiles, num_kde_samples, out_dir):
    """
    One PNG per sigma level: KDE of corrupted Close values vs. the reference
    KDE vs. the analytic diffusion prior N(0, sigma^2). Axes are fixed across
    all frames (using the largest sigma to set the range) so the reference
    distribution visibly shrinks to a spike relative to the widening corrupted
    density — that shrinkage *is* the "approaching the prior" effect.

    Uses fresh eps draws per sigma: many independent samples give a much
    cleaner KDE than the handful reused for the path/price plots, and there is
    no continuity requirement for a per-sigma density snapshot.
    """
    os.makedirs(out_dir, exist_ok=True)
    L = len(gt_close)
    sigma_last = float(sigmas[-1])

    ref_lo = np.quantile(gt_close, 0.001)
    ref_hi = np.quantile(gt_close, 0.999)
    x_lo = min(ref_lo, -4.0 * sigma_last)
    x_hi = max(ref_hi,  4.0 * sigma_last)
    x_grid = np.linspace(x_lo, x_hi, 400)

    ref_kde  = gaussian_kde(np.clip(gt_close, ref_lo, ref_hi))
    ref_dens = ref_kde(x_grid)

    # Pre-compute all frame data to fix the y-axis before plotting.
    frame_data = []
    for sigma, q in zip(sigmas, quantiles):
        eps = np.random.randn(num_kde_samples, L)
        x_sigma = gt_close[None, :] + sigma * eps
        vals = np.clip(x_sigma.ravel(), x_lo, x_hi)
        try:
            dens = gaussian_kde(vals)(x_grid)
        except np.linalg.LinAlgError:
            dens = np.zeros_like(x_grid)
        prior_dens = norm.pdf(x_grid, loc=0.0, scale=sigma)
        frame_data.append((sigma, q, dens, prior_dens))

    y_max = max(
        ref_dens.max(),
        *(d.max() for _, _, d, _ in frame_data),
        *(p.max() for _, _, _, p in frame_data),
    ) * 1.15

    for idx, (sigma, q, dens, prior_dens) in enumerate(frame_data):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_grid, ref_dens, "k--", lw=1.8, alpha=0.85, label="Reference (sigma=0)")
        ax.fill_between(x_grid, dens, alpha=0.30, color=_COLORS["marginal"])
        ax.plot(x_grid, dens, color=_COLORS["marginal"], lw=1.6,
                label=f"Corrupted  (sigma = {sigma:.3f})")
        ax.plot(x_grid, prior_dens, color=_COLORS["prior"], lw=1.4, ls=":",
                label=r"Prior  $\mathcal{N}(0,\sigma^2)$")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, y_max)
        ax.set_xlabel("Normalised close log-return")
        ax.set_ylabel("Density")
        ax.set_title(f"Forward process — quantile {q:.3f}  (sigma = {sigma:.3f})")
        ax.legend()
        _save(fig, os.path.join(out_dir, f"forward_marginal_step_{idx:04d}.png"))


# ── Visualisation 2: Sample path evolution toward noise ──────────────────────

def plot_evolution_forward(gt_close, sigmas, quantiles, fixed_eps, out_dir):
    """
    One PNG per sigma level: reference sequence (fixed dashed) + a small set of
    corrupted paths x_sigma = x_0 + sigma*eps, reusing the *same* eps vectors
    across every sigma so each coloured line is one trajectory continuously
    dissolving into noise, not an independent resample per frame.
    """
    os.makedirs(out_dir, exist_ok=True)
    L  = len(gt_close)
    xs = np.arange(L)
    sigma_last = float(sigmas[-1])

    y_lo = min(float(gt_close.min()), -4.0 * sigma_last)
    y_hi = max(float(gt_close.max()),  4.0 * sigma_last)
    pad  = 0.10 * (y_hi - y_lo)
    y_lo -= pad; y_hi += pad

    num_path_samples = fixed_eps.shape[0]
    for idx, (sigma, q) in enumerate(zip(sigmas, quantiles)):
        x_sigma = gt_close[None, :] + sigma * fixed_eps   # (num_path_samples, L)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(xs, gt_close, "k--", lw=1.8, alpha=0.75, label="Reference (true close)")
        for s in range(num_path_samples):
            ax.plot(xs, x_sigma[s], color=_COLORS["evolution"], lw=1.1, alpha=0.65,
                    label="Corrupted paths" if s == 0 else None)
        ax.set_xlim(0, L - 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalised log-return")
        ax.set_title(f"Forward process — quantile {q:.3f}  (sigma = {sigma:.3f})")
        ax.legend(loc="upper right")
        _save(fig, os.path.join(out_dir, f"forward_evolution_step_{idx:04d}.png"))


# ── Visualisation 3: Price path evolution toward noise ───────────────────────

def plot_prices_forward(gt_close, sigmas, quantiles, fixed_eps, norm_stats,
                        close_feat, out_dir):
    """
    One PNG per sigma level: relative price paths (S_0 = 1) built from the
    denormalised corrupted Close returns, reusing the same fixed_eps paths as
    plot_evolution_forward for consistency. Skipped if norm_stats lack the
    close feature.

    At high sigma the cumulative corrupted log-return sum can be large enough
    that exp() overflows to inf; the cumulative sum is clipped to
    +/-_MAX_CUMSUM_EXPONENT before exponentiating and a warning is printed
    when clipping is triggered. This is itself informative: it marks the
    point at which the "log-return path" interpretation breaks down and the
    corrupted sequence is just noise.
    """
    if close_feat not in norm_stats:
        print(f"  [prices] skipping — '{close_feat}' not in norm_stats")
        return
    mean, std = norm_stats[close_feat]
    os.makedirs(out_dir, exist_ok=True)

    gt_denorm = denorm(gt_close, mean, std)
    gt_price  = _to_price_path(gt_denorm)
    L1 = len(gt_price)
    xs = np.arange(L1)

    num_path_samples = fixed_eps.shape[0]

    def _safe_price_path(returns: np.ndarray) -> tuple:
        cumsum = np.concatenate([[0.0], np.cumsum(returns)])
        clipped = np.any(np.abs(cumsum) > _MAX_CUMSUM_EXPONENT)
        cumsum = np.clip(cumsum, -_MAX_CUMSUM_EXPONENT, _MAX_CUMSUM_EXPONENT)
        return np.exp(cumsum), clipped

    frame_prices = []
    all_prices = [gt_price]
    any_clipped = False
    for sigma, q in zip(sigmas, quantiles):
        x_sigma = gt_close[None, :] + sigma * fixed_eps        # (num_path_samples, L)
        prices = []
        for s in range(num_path_samples):
            gen_denorm = denorm(x_sigma[s], mean, std)
            gen_price, clipped = _safe_price_path(gen_denorm)
            prices.append(gen_price)
            any_clipped = any_clipped or clipped
        frame_prices.append((sigma, q, prices))
        all_prices.extend(prices)

    if any_clipped:
        print("  [prices] warning: cumulative log-return sum clipped to avoid "
              "float overflow at high sigma — price path no longer meaningful "
              "at those levels (expected: this is the noise-dominated regime).")

    y_lo = min(float(p.min()) for p in all_prices)
    y_hi = max(float(p.max()) for p in all_prices)
    pad  = 0.10 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    y_lo -= pad; y_hi += pad

    for idx, (sigma, q, prices) in enumerate(frame_prices):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(xs, gt_price, "k--", lw=1.8, alpha=0.75, label="Reference price path")
        for s, price in enumerate(prices):
            ax.plot(xs, price, color=_COLORS["prices"], lw=1.1, alpha=0.65,
                    label="Corrupted price paths" if s == 0 else None)
        ax.set_xlim(0, L1 - 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Relative price  ($S_0 = 1$)")
        ax.set_title(f"Forward process — quantile {q:.3f}  (sigma = {sigma:.3f})")
        ax.legend(loc="upper right")
        _save(fig, os.path.join(out_dir, f"forward_prices_step_{idx:04d}.png"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Forward (noising) process visualizations for EDM-CSDI."
    )
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)
    p.add_argument("--split",             type=str, default="val",
                   choices=["train", "val"])
    p.add_argument("--window_idx",        type=int, default=0,
                   help="Index into the chosen split (default: 0).")
    p.add_argument("--quantiles",         type=float, nargs="+",
                   default=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999],
                   help="Quantiles (0,1) of the log-normal training schedule "
                        "used to pick monotonically increasing sigma levels.")
    p.add_argument("--P_mean",            type=float, default=-1.4,
                   help="Mean of ln(sigma) ~ N(P_mean, P_std^2) (EDM training schedule).")
    p.add_argument("--P_std",             type=float, default=1.8,
                   help="Std of ln(sigma) ~ N(P_mean, P_std^2) (EDM training schedule).")
    p.add_argument("--num_kde_samples",   type=int, default=1000,
                   help="Independent eps draws per sigma for the marginal KDE (default: 1000).")
    p.add_argument("--num_path_samples",  type=int, default=5,
                   help="Fixed eps draws (reused across all sigma) for the "
                        "path/price evolution plots (default: 5).")
    p.add_argument("--out_dir",           type=str, default="figures/forward")
    p.add_argument("--norm_stats",        type=str, default=None,
                   help="Path to normalization_stats.csv for price-path "
                        "denormalization (optional).")
    p.add_argument("--date_format",       type=str, default="%d/%m/%Y")
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load config metadata only (no model, no device — forward process
    #    never calls the network) ───────────────────────────────────────────
    config, feat_cols, close_idx, seq_len = load_config_only(
        args.checkpoint_folder, args.checkpoint_name
    )
    close_feat = feat_cols[close_idx]

    # ── Load one window (reuse visualize_denoising's loader for the exact
    #    same train/val reconstruction & window selection logic) ─────────────
    repo_root = os.path.abspath(os.path.dirname(__file__))
    device = torch.device("cpu")
    _, _, _, gt_close, window_info = load_window(
        config, args.split, args.window_idx, repo_root,
        device, num_samples=1, close_idx=close_idx, feat_cols=feat_cols,
        seq_len=seq_len, date_format=args.date_format,
    )
    gt_close = np.asarray(gt_close, dtype=np.float64)

    # ── Sigma schedule from log-normal quantiles ──────────────────────────────
    sigmas, quantiles = sigmas_from_quantiles(args.quantiles, args.P_mean, args.P_std)
    print(f"\nSigma levels (P_mean={args.P_mean}, P_std={args.P_std}):")
    for q, s in zip(quantiles, sigmas):
        print(f"  quantile {q:6.3f}  ->  sigma = {s:.4f}")

    # ── Save chosen window/run info for later reuse ──────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    run_info = {
        **window_info,
        "checkpoint_folder": args.checkpoint_folder,
        "checkpoint_name":   args.checkpoint_name,
        "P_mean":            args.P_mean,
        "P_std":             args.P_std,
        "quantiles":         quantiles.tolist(),
        "sigmas":            sigmas.tolist(),
        "num_kde_samples":   args.num_kde_samples,
        "num_path_samples":  args.num_path_samples,
        "seed":              args.seed,
    }
    run_info_path = os.path.join(args.out_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run/window info -> {run_info_path}")

    # ── Fixed eps draws for path/price plots (continuity across sigma) ───────
    fixed_eps = np.random.randn(args.num_path_samples, len(gt_close))

    # ── Load normalisation stats (optional, needed for price paths) ──────────
    norm_stats: dict = {}
    if args.norm_stats:
        if os.path.isfile(args.norm_stats):
            norm_stats = load_norm_stats(args.norm_stats, feat_cols)
            print(f"Loaded norm stats for: {list(norm_stats.keys())}")
        else:
            print(f"[warn] norm_stats not found: {args.norm_stats}  "
                  f"— price-path visualisation will be skipped")

    # ── Run all three visualisations ──────────────────────────────────────────
    print("\n-- Vis 1: Marginal distribution evolution toward the prior --")
    plot_marginal_forward(gt_close, sigmas, quantiles, args.num_kde_samples, args.out_dir)

    print("\n-- Vis 2: Sample path evolution toward noise --")
    plot_evolution_forward(gt_close, sigmas, quantiles, fixed_eps, args.out_dir)

    print("\n-- Vis 3: Price path evolution toward noise --")
    plot_prices_forward(gt_close, sigmas, quantiles, fixed_eps, norm_stats,
                        close_feat, args.out_dir)

    print(f"\nDone. All outputs in: {args.out_dir}")
    print("GIF assembly example:")
    print("  ffmpeg -pattern_type glob -framerate 2 "
          f"-i '{args.out_dir}/forward_marginal_step_*.png' forward_marginal.gif")


if __name__ == "__main__":
    main()
