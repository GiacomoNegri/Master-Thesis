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
        --split train \\
        --window_idx 0 \\
        --num_samples 5 \\
        --num_steps 50 \\
        --snapshot_steps 0 5 15 30 49 \\
        --out_dir figures/price_evolution \\
        --norm_stats data/fake_fts_processed/normalization_stats.csv \\
        --seed 42

To reference the exact same window as visualize_forward_process.py, pass the
same --checkpoint_folder, --checkpoint_name, --split, and --window_idx (window
selection in both scripts is a pure function of those four values plus
--date_format — see load_window in visualize_denoising.py):
    python visualize_price_evolution.py \\
        --checkpoint_folder final/edm \\
        --checkpoint_name EDM_REPLICATION_CLOS_ep-500_step-44500_lr-8e-04_ch-128_layers-6_nheads-4_diffemb-256_sd-1.0_Pm--1.4_Ps-1.8_20260602_201915.pt \\
        --split train --window_idx 0 \\
        --num_samples 5 --num_steps 50 \\
        --norm_stats data/general/normalization_stats_norm_replication.csv \\
        --out_dir figures/price_evolution --seed 42
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
    _array_signature,
)


# ── Visualisation: Prior noise vs reference (unnormalized price path) ────────

def compute_prior_prices(gt_close, close_idx, num_samples, shape, sigma_max,
                         device, norm_stats, close_feat):
    """
    Draw the sampler's literal prior x ~ N(0, sigma_max^2 I) once, and
    denormalize it plus the reference into relative price paths (S_0 = 1).
    Returns (gt_price, prior_prices) or None if `close_feat` isn't in
    `norm_stats`. Kept separate from rendering so the same random draw can
    be plotted at more than one y-axis scale without redrawing noise.
    """
    if close_feat not in norm_stats:
        print(f"  [prior] skipping — '{close_feat}' not in norm_stats")
        return None
    mean, std = norm_stats[close_feat]

    gt_denorm = denorm(gt_close, mean, std)
    gt_price  = _to_price_path(gt_denorm)

    x_prior     = torch.randn(*shape, device=device) * sigma_max
    prior_close = x_prior[:, close_idx, :].cpu().numpy()   # (num_samples, L)

    prior_prices = [
        _to_price_path(denorm(prior_close[s], mean, std))
        for s in range(num_samples)
    ]
    return gt_price, prior_prices


def plot_prior_vs_reference_prices(gt_price, prior_prices, num_samples, ylim, out_dir, filename):
    """
    One PNG: reference price path (dashed) vs. the (already computed) prior
    noise price paths, at the given fixed `ylim`.
    """
    os.makedirs(out_dir, exist_ok=True)
    L1 = len(gt_price)
    xs = np.arange(L1)
    colors = cm.tab10(np.linspace(0.0, 0.9, num_samples))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xs, gt_price, "k--", lw=1.8, alpha=0.8, label="Reference price path")
    for s, (path, color) in enumerate(zip(prior_prices, colors)):
        ax.plot(xs, path, color=color, lw=1.0, alpha=0.6, label=f"Prior sample {s}")
    ax.set_xlim(0, L1 - 1)
    ax.set_ylim(*ylim)
    _save(fig, os.path.join(out_dir, filename))


# ── Visualisation: Multi-sample price path evolution ─────────────────────────

def plot_price_evolution(snapshots, gt_close, close_idx, num_samples, selected_steps,
                         norm_stats, close_feat, out_dir):
    """
    One PNG per selected step: reference price path (dashed) plus all
    `num_samples` generated price paths (relative, S_0 = 1), overlaid.
    Y-axis fixed across frames. Returns the (y_lo, y_hi) used, or None if
    skipped, so callers can reuse the same axis elsewhere (e.g. the prior
    noise vs reference plot).
    """
    if close_feat not in norm_stats:
        print(f"  [prices] skipping — '{close_feat}' not in norm_stats")
        return None
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
        return None

    y_lo = min(float(p.min()) for p in all_prices)
    y_hi = max(float(p.max()) for p in all_prices)
    pad  = 0.10 * (y_hi - y_lo)
    ylim = (y_lo - pad, y_hi + pad)

    ref_sig = _array_signature(gt_price)
    print(f"  [prices] mean={mean}  std={std}  reference price signature: {ref_sig}")
    with open(os.path.join(out_dir, "reference_signature.json"), "w") as f:
        json.dump({"mean": mean, "std": std, "gt_price_signature": ref_sig}, f, indent=2)

    for step, (sigma, paths) in sorted(step_prices.items()):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(xs, gt_price, "k--", lw=1.8, alpha=0.8, label="Reference price path")
        for s, (path, color) in enumerate(zip(paths, colors)):
            ax.plot(xs, path, color=color, lw=1.2, alpha=0.8, label=f"Sample {s}")
        ax.set_xlim(0, L1 - 1)
        ax.set_ylim(*ylim)
        # ax.set_xlabel("Time step")
        # ax.set_ylabel("Relative price  ($S_0 = 1$)")
        ax.set_title(f"Step {step}")
        _save(fig, os.path.join(out_dir, f"prices_multi_step_{step:04d}.png"))

    return ylim


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
                   help="Index into the chosen split (default: 0). Ignored if "
                        "--window_file is given.")
    p.add_argument("--window_file",       type=str, default=None,
                   help="Select the window by CSV basename (e.g. OKE_...csv) "
                        "instead of --window_idx. Invariant to how many other "
                        "files are in the data directory, so it resolves to the "
                        "same window across machines whose data dirs differ.")
    p.add_argument("--window_start",      type=int, default=0,
                   help="Start row within --window_file (default: 0). "
                        "Only used with --window_file.")
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
        window_file=args.window_file, window_start=args.window_start,
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
    ylim = plot_price_evolution(snapshots, gt_close, close_idx, args.num_samples,
                                selected_steps, norm_stats, close_feat, args.out_dir)

    print("\n── Prior noise vs reference (unnormalized price path) ──")
    prior_data = compute_prior_prices(gt_close, close_idx, args.num_samples, tuple(obs.shape),
                                      sigma_max, device, norm_stats, close_feat)
    if prior_data is not None:
        gt_price_prior, prior_prices = prior_data

        # Version 1: matched to the step frames' shared y-axis.
        if ylim is not None:
            plot_prior_vs_reference_prices(
                gt_price_prior, prior_prices, args.num_samples, ylim,
                args.out_dir, "prior_vs_reference_prices_matched.png",
            )

        # Version 2: own axis, scaled so all prior noise samples are visible.
        all_prices = [gt_price_prior] + prior_prices
        y_lo = min(float(p.min()) for p in all_prices)
        y_hi = max(float(p.max()) for p in all_prices)
        pad  = 0.10 * (y_hi - y_lo)
        full_ylim = (y_lo - pad, y_hi + pad)
        plot_prior_vs_reference_prices(
            gt_price_prior, prior_prices, args.num_samples, full_ylim,
            args.out_dir, "prior_vs_reference_prices_full.png",
        )

    print(f"\nDone. All outputs in: {args.out_dir}")
    print("GIF assembly example:")
    print("  ffmpeg -pattern_type glob -framerate 3 "
          f"-i '{args.out_dir}/prices_multi_step_*.png' prices_multi.gif")


if __name__ == "__main__":
    main()
