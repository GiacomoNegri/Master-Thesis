#!/usr/bin/env python
"""
visualize_price_evolution.py  –  Multi-sample price path evolution for EDM-CSDI.

Standalone variant of visualize_denoising.py's "price path evolution" plot
(vis 3), extended to overlay all `num_samples` generated paths (instead of a
single `sample_idx`) against the reference price path, one PNG per selected
denoising step.

Frame plots are saved with zero-padded step indices for easy GIF assembly:
    ffmpeg -pattern_type glob -framerate 3 -i 'figures/price_evolution/prices_multi_step_*.png' prices_multi.gif

Usage:
    python visualize_price_evolution.py \\
        --checkpoint_folder ohlc_conditional \\
        --checkpoint_name MY.pt \\
        --split val \\
        --window_idx 0 \\
        --num_samples 5 \\
        --num_steps 50 \\
        --snapshot_steps 0 5 15 30 49 \\
        --out_dir figures/price_evolution \\
        --norm_stats data/fake_fts_processed/normalization_stats.csv \\
        --seed 42
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from visualize_denoising import (
    load_model,
    load_window,
    run_sampler_with_snapshots,
    load_norm_stats,
    denorm,
    _to_price_path,
    _save,
)


# ── Visualisation: Prior noise vs reference (unnormalized price path) ────────

def plot_prior_vs_reference_prices(gt_close, close_idx, num_samples, shape, sigma_max,
                                   device, norm_stats, close_feat, out_dir):
    """
    One PNG: raw prior noise sample(s) x ~ N(0, sigma_max^2 I) — the sampler's
    literal starting point — denormalized and converted to relative price
    paths (S_0 = 1), plotted against the reference (unnormalized) price path.

    This does NOT run the model or the sampler; it draws the same prior
    (`torch.randn(*shape) * sigma_max`) used at the top of `edm_sampler`.
    """
    if close_feat not in norm_stats:
        print(f"  [prior] skipping — '{close_feat}' not in norm_stats")
        return
    mean, std = norm_stats[close_feat]
    os.makedirs(out_dir, exist_ok=True)

    gt_denorm = denorm(gt_close, mean, std)
    gt_price  = _to_price_path(gt_denorm)
    L1 = len(gt_price)
    xs = np.arange(L1)

    x_prior     = torch.randn(*shape, device=device) * sigma_max
    prior_close = x_prior[:, close_idx, :].cpu().numpy()   # (num_samples, L)

    colors = cm.tab10(np.linspace(0.0, 0.9, num_samples))

    prior_prices = []
    for s in range(num_samples):
        prior_denorm = denorm(prior_close[s], mean, std)
        prior_prices.append(_to_price_path(prior_denorm))

    all_prices = [gt_price] + prior_prices
    y_lo = min(float(p.min()) for p in all_prices)
    y_hi = max(float(p.max()) for p in all_prices)
    pad  = 0.10 * (y_hi - y_lo)
    y_lo -= pad;  y_hi += pad

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xs, gt_price, "k--", lw=1.8, alpha=0.8, label="Reference price path")
    for s, (path, color) in enumerate(zip(prior_prices, colors)):
        ax.plot(xs, path, color=color, lw=1.0, alpha=0.6, label=f"Prior sample {s}")
    ax.set_xlim(0, L1 - 1)
    ax.set_ylim(y_lo, y_hi)
    _save(fig, os.path.join(out_dir, "prior_vs_reference_prices.png"))


# ── Visualisation: Multi-sample price path evolution ─────────────────────────

def plot_price_evolution(snapshots, gt_close, close_idx, num_samples, selected_steps,
                         norm_stats, close_feat, out_dir):
    """
    One PNG per selected step: reference price path (dashed) plus all
    `num_samples` generated price paths (relative, S_0 = 1), overlaid.
    Y-axis fixed across frames.
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

    colors = cm.tab10(np.linspace(0.0, 0.9, num_samples))

    # Pre-compute all price paths to determine fixed y-axis.
    step_prices = {}
    all_prices  = [gt_price]
    for step, sigma, D_x in snapshots:
        if step not in selected_steps:
            continue
        paths = []
        for s in range(num_samples):
            gen_denorm = denorm(D_x[s, close_idx, :].numpy(), mean, std)
            paths.append(_to_price_path(gen_denorm))
        step_prices[step] = (sigma, paths)
        all_prices.extend(paths)

    if not step_prices:
        print("  [prices] no matching snapshot steps — skipping")
        return

    y_lo = min(float(p.min()) for p in all_prices)
    y_hi = max(float(p.max()) for p in all_prices)
    pad  = 0.10 * (y_hi - y_lo)
    y_lo -= pad;  y_hi += pad

    for step, (sigma, paths) in sorted(step_prices.items()):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(xs, gt_price, "k--", lw=1.8, alpha=0.8, label="Reference price path")
        for s, (path, color) in enumerate(zip(paths, colors)):
            ax.plot(xs, path, color=color, lw=1.2, alpha=0.8, label=f"Sample {s}")
        ax.set_xlim(0, L1 - 1)
        ax.set_ylim(y_lo, y_hi)
        # ax.set_xlabel("Time step")
        # ax.set_ylabel("Relative price  ($S_0 = 1$)")
        ax.set_title(f"Step {step}")
        _save(fig, os.path.join(out_dir, f"prices_multi_step_{step:04d}.png"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-sample price path evolution for EDM-CSDI."
    )
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)
    p.add_argument("--split",             type=str, default="val",
                   choices=["train", "val"])
    p.add_argument("--window_idx",        type=int, default=0,
                   help="Index into the chosen split (default: 0).")
    p.add_argument("--num_samples",       type=int, default=5,
                   help="Number of independent generated paths to overlay.")
    p.add_argument("--num_steps",         type=int, default=50,
                   help="Heun denoising steps (default: 50).")
    p.add_argument("--sigma_max",         type=float, default=None,
                   help="Override the sampler's sigma_max (top of the noise "
                        "schedule). Default: the checkpoint's own sigma_max.")
    p.add_argument("--snapshot_steps",    type=int, nargs="+", default=None,
                   help="Step indices to render as frames. "
                        "Default: 5 evenly-spaced steps including first and last.")
    p.add_argument("--out_dir",           type=str, default="figures/price_evolution")
    p.add_argument("--norm_stats",        type=str, required=True,
                   help="Path to normalization_stats.csv for price-path "
                        "denormalization.")
    p.add_argument("--date_format",       type=str, default="%d/%m/%Y")
    p.add_argument("--seed",              type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (model, config, processes, feat_cols, close_idx,
     seq_len, rho, sigma_min, sigma_max) = load_model(args, device)
    close_feat = feat_cols[close_idx]

    if args.sigma_max is not None:
        print(f"Overriding sigma_max: {sigma_max} → {args.sigma_max}")
        sigma_max = args.sigma_max

    repo_root = os.path.abspath(os.path.dirname(__file__))
    obs, cond_mask, observed_tp, gt_close, window_info = load_window(
        config, args.split, args.window_idx, repo_root,
        device, args.num_samples, close_idx, feat_cols, seq_len,
        date_format=args.date_format,
    )

    # ── Save chosen window/run info for later reuse ──────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    run_info = {
        **window_info,
        "checkpoint_folder": args.checkpoint_folder,
        "checkpoint_name":   args.checkpoint_name,
        "num_samples":       args.num_samples,
        "num_steps":         args.num_steps,
        "sigma_max":         sigma_max,
        "seed":              args.seed,
    }
    run_info_path = os.path.join(args.out_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run/window info → {run_info_path}")

    if not os.path.isfile(args.norm_stats):
        raise FileNotFoundError(f"norm_stats not found: {args.norm_stats}")
    norm_stats = load_norm_stats(args.norm_stats, feat_cols)
    print(f"Loaded norm stats for: {list(norm_stats.keys())}")

    print("\n── Prior noise vs reference (unnormalized price path) ──")
    plot_prior_vs_reference_prices(gt_close, close_idx, args.num_samples, tuple(obs.shape),
                                   sigma_max, device, norm_stats, close_feat, args.out_dir)

    print(f"\nRunning EDM sampler ({args.num_steps} steps, "
          f"{args.num_samples} samples)…")
    snapshots = run_sampler_with_snapshots(
        processes, model, obs, cond_mask, observed_tp,
        args.num_steps, rho, device,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    if args.snapshot_steps is None:
        idxs = np.linspace(0, args.num_steps - 1, 5, dtype=int).tolist()
    else:
        idxs = [s for s in args.snapshot_steps if 0 <= s < args.num_steps]
    if not idxs:
        raise ValueError("No valid snapshot_steps in range [0, num_steps).")
    selected_steps = set(idxs)
    print(f"\nFrame steps: {sorted(selected_steps)}")

    print("\n── Multi-sample price path evolution ──")
    plot_price_evolution(snapshots, gt_close, close_idx, args.num_samples,
                         selected_steps, norm_stats, close_feat, args.out_dir)

    print(f"\nDone. All outputs in: {args.out_dir}")
    print("GIF assembly example:")
    print("  ffmpeg -pattern_type glob -framerate 3 "
          f"-i '{args.out_dir}/prices_multi_step_*.png' prices_multi.gif")


if __name__ == "__main__":
    main()
