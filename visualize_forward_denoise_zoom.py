#!/usr/bin/env python
"""
visualize_forward_denoise_zoom.py — Paired forward/denoise ZOOM marginals over a
shared pooled multi-window reference, for EDM-CSDI. Single-GPU / one-shot.

Motivation
----------
`visualize_forward_process.py` shows the *noising* zoom (a window's Close
marginal spreading into the diffusion prior) and
`visualize_denoising_multi_window.py` shows the *denoising* marginal (the model's
denoised estimate converging back toward the empirical data marginal). To narrate
a clean "we noise this distribution, then denoise the SAME distribution" story,
both sides must reference the *identical* target marginal.

This script does exactly that: it selects ONE pooled set of `--num_windows`
distinct windows (same selection rule as visualize_denoising_multi_window.py:
same `--split`, `--seed`, count) and builds a single reference KDE from that pool.
That same reference — computed the same way (unclipped gaussian_kde on the pooled
Close values, evaluated on each frame's own x-grid) — is drawn on every frame of
both series, so the dashed reference curve is the same function on both sides.

Two independent zoom series are produced (per the chosen design):
  - Forward (noising):  sigma levels from log-normal training-schedule quantiles
                        (`--quantiles`, `--P_mean`, `--P_std`); NO model involved;
                        each frame x-window sized from the *corrupted* pool's
                        quantiles at that sigma (mirrors plot_marginal_forward_zoomed).
  - Denoise:            the EDM Heun sampler run over `--num_steps`; each snapshot
                        step's D_x pooled over (windows x seq_len); each frame
                        x-window sized from that step's D_x quantiles.

Only the zoom marginals are produced — no full-range marginal, path, or price
plots. Frames are zero-padded for GIF assembly:
    ffmpeg -pattern_type glob -framerate 2 -i 'figures/zoom/forward_marginal_zoom_*.png' forward_zoom.gif
    ffmpeg -pattern_type glob -framerate 3 -i 'figures/zoom/denoise_marginal_zoom_*.png' denoise_zoom.gif

Usage
-----
    python visualize_forward_denoise_zoom.py \\
        --checkpoint_folder final/edm \\
        --checkpoint_name EDM_REPLICATION_....pt \\
        --split train --num_windows 100 --seed 42 \\
        --quantiles 0.001 0.05 0.25 0.5 0.75 0.95 0.999 \\
        --P_mean -1.4 --P_std 1.8 --num_kde_samples 200 \\
        --num_steps 200 --snapshot_steps 0 20 40 60 80 100 120 140 160 180 195 199 \\
        --out_dir figures/zoom
"""

import argparse
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

# Reuse the exact sigma-schedule + corrupted-bounds logic from the forward
# script, and the exact model/window/sampler logic from the multi-window
# denoise script, so this paired view can never drift from either.
from visualize_forward_process import sigmas_from_quantiles, _corrupted_quantile_bounds
from visualize_denoising_multi_window import (
    load_model,
    load_windows,
    run_sampler_with_snapshots,
)


# ── Plotting style (mirrors both source scripts) ─────────────────────────────
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

_FORWARD_COLOR = "#E64A19"   # deep orange — noising (adding noise)
_DENOISE_COLOR = "#1E88E5"   # blue        — denoising (model estimate)
_PRIOR_COLOR   = "#607D8B"   # blue-grey   — analytic N(0, sigma^2) overlay


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _zoom_window(vals: np.ndarray, lo_q: float, hi_q: float):
    """Per-frame x-window from the [lo_q, hi_q] quantiles of `vals`, guarded
    against a degenerate (zero-width) range."""
    x_lo = float(np.quantile(vals, lo_q))
    x_hi = float(np.quantile(vals, hi_q))
    if not (np.isfinite(x_lo) and np.isfinite(x_hi)) or x_hi <= x_lo:
        c = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.0
        x_lo, x_hi = c - 1e-6, c + 1e-6
    return x_lo, x_hi


def _plot_zoom_frame(x_grid, x_lo, x_hi, dens, ref_dens, title, out_path,
                     color, series_label, prior_dens=None):
    """One zoom-marginal PNG: the current density (filled) + the shared pooled
    reference (dashed black) + optional analytic prior (dotted). The reference
    is drawn on every frame so convergence/divergence from it is visible."""
    y_candidates = [dens.max(), ref_dens.max()]
    if prior_dens is not None:
        y_candidates.append(prior_dens.max())
    y_max = max(y_candidates) * 1.15

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_grid, ref_dens, "k--", lw=1.8, alpha=0.85, label="Pooled reference")
    ax.fill_between(x_grid, dens, alpha=0.30, color=color)
    ax.plot(x_grid, dens, color=color, lw=1.6, label=series_label)
    if prior_dens is not None:
        ax.plot(x_grid, prior_dens, color=_PRIOR_COLOR, lw=1.4, ls=":",
                label=r"Prior $\mathcal{N}(0,\sigma^2)$")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("Normalised close log-return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    _save(fig, out_path)


# ── Forward (noising) zoom series ────────────────────────────────────────────

def plot_forward_zoom(pooled_ref, ref_kde, sigmas, quantiles, num_kde_samples,
                      num_windows, out_dir, lo_q, hi_q):
    """Per-frame zoom of the corrupted pooled Close marginal at each schedule
    sigma. Each frame's x-window comes from the corrupted pool's own quantiles
    at that sigma (mirrors visualize_forward_process.plot_marginal_forward_zoomed).
    """
    L = len(pooled_ref)
    for idx, (sigma, q) in enumerate(zip(sigmas, quantiles)):
        sigma = float(sigma)
        x_lo, x_hi = _corrupted_quantile_bounds(pooled_ref, sigma, num_kde_samples,
                                                lo_q=lo_q, hi_q=hi_q)
        x_grid = np.linspace(x_lo, x_hi, 400)

        eps  = np.random.randn(num_kde_samples, L)
        vals = (pooled_ref[None, :] + sigma * eps).ravel()
        try:
            dens = gaussian_kde(vals)(x_grid)
        except np.linalg.LinAlgError:
            dens = np.zeros_like(x_grid)
        prior_dens = norm.pdf(x_grid, loc=0.0, scale=sigma)
        ref_dens   = ref_kde(x_grid)

        _plot_zoom_frame(
            x_grid, x_lo, x_hi, dens, ref_dens,
            title=f"Forward noising (zoom, {num_windows} windows) — "
                  f"quantile {float(q):.3f}  (σ = {sigma:.3f})",
            out_path=os.path.join(out_dir, f"forward_marginal_zoom_{idx:04d}.png"),
            color=_FORWARD_COLOR,
            series_label=f"Corrupted  (σ = {sigma:.3f})",
            prior_dens=prior_dens,
        )


# ── Denoise zoom series ──────────────────────────────────────────────────────

def plot_denoise_zoom(snapshots, ref_kde, close_idx, selected_steps,
                      num_windows, out_dir, lo_q, hi_q):
    """Per-frame zoom of the pooled denoised estimate D_x at each selected
    sampler step. Each frame's x-window comes from that step's D_x quantiles.
    No analytic prior overlay: D_x is the model's denoised (MMSE) estimate, not
    the noisy latent, so N(0, sigma^2) is not the distribution it approaches."""
    for step, sigma, D_x in sorted(snapshots, key=lambda t: t[0]):
        if step not in selected_steps:
            continue
        vals = D_x[:, close_idx, :].numpy().ravel()   # pool over windows AND time
        x_lo, x_hi = _zoom_window(vals, lo_q, hi_q)
        x_grid = np.linspace(x_lo, x_hi, 400)
        try:
            dens = gaussian_kde(vals)(x_grid)
        except np.linalg.LinAlgError:
            dens = np.zeros_like(x_grid)
        ref_dens = ref_kde(x_grid)

        _plot_zoom_frame(
            x_grid, x_lo, x_hi, dens, ref_dens,
            title=f"Denoising (zoom, {num_windows} windows) — "
                  f"step {step}  (σ = {float(sigma):.3f})",
            out_path=os.path.join(out_dir, f"denoise_marginal_zoom_{step:04d}.png"),
            color=_DENOISE_COLOR,
            series_label=f"Denoised D_x  (step {step})",
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Paired forward/denoise zoom marginals over a shared pooled "
                    "multi-window reference (EDM-CSDI)."
    )
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)
    p.add_argument("--split",             type=str, default="val",
                   choices=["train", "val"])
    p.add_argument("--num_windows",       type=int, default=50,
                   help="Number of DISTINCT windows to pool for the shared "
                        "reference. Selected exactly like "
                        "visualize_denoising_multi_window.py's --num_samples "
                        "(same --split/--seed).")
    # Forward (noising) schedule ------------------------------------------------
    p.add_argument("--quantiles",         type=float, nargs="+",
                   default=[0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999],
                   help="Quantiles (0,1) of ln(sigma)~N(P_mean,P_std^2) picking "
                        "the forward noising sigma levels.")
    p.add_argument("--P_mean",            type=float, default=-1.4,
                   help="Mean of ln(sigma)~N(P_mean,P_std^2) (EDM training schedule).")
    p.add_argument("--P_std",             type=float, default=1.8,
                   help="Std of ln(sigma)~N(P_mean,P_std^2) (EDM training schedule).")
    p.add_argument("--num_kde_samples",   type=int, default=200,
                   help="Independent eps draws per sigma for the forward KDE.")
    # Denoise schedule ----------------------------------------------------------
    p.add_argument("--num_steps",         type=int, default=50,
                   help="Heun denoising steps.")
    p.add_argument("--sigma_max",         type=float, default=None,
                   help="Override the sampler's sigma_max. Default: checkpoint's own.")
    p.add_argument("--snapshot_steps",    type=int, nargs="+", default=None,
                   help="Sampler step indices to render as denoise zoom frames. "
                        "Default: 6 evenly-spaced steps including first and last.")
    # Shared --------------------------------------------------------------------
    p.add_argument("--lo_q",              type=float, default=0.001,
                   help="Lower quantile for every frame's x-window (default 0.001).")
    p.add_argument("--hi_q",              type=float, default=0.999,
                   help="Upper quantile for every frame's x-window (default 0.999).")
    p.add_argument("--out_dir",           type=str, default="figures/zoom")
    p.add_argument("--date_format",       type=str, default="%d/%m/%Y")
    p.add_argument("--seed",              type=int, default=42,
                   help="Seeds torch/sampler RNG, numpy RNG, and window selection.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model + config (needed for the denoise sampler) ──────────────────
    (model, config, processes, feat_cols, close_idx,
     seq_len, rho, sigma_min, sigma_max) = load_model(args, device)

    if args.sigma_max is not None:
        print(f"Overriding sigma_max: {sigma_max} → {args.sigma_max}")
        sigma_max = args.sigma_max

    # ── Select ONE pooled set of windows — shared reference for both series ───
    repo_root = os.path.abspath(os.path.dirname(__file__))
    obs, cond_mask, observed_tp, gt_close_batch, windows_info = load_windows(
        config, args.split, args.num_windows, repo_root, device,
        close_idx, feat_cols, seq_len, args.seed,
        date_format=args.date_format,
    )

    # Shared reference: unclipped KDE on the pooled Close values. Both series
    # evaluate THIS SAME kde on their per-frame grids, so the dashed reference
    # curve is the same function on the forward and denoise sides.
    pooled_ref = gt_close_batch.astype(np.float64).ravel()
    ref_kde    = gaussian_kde(pooled_ref)
    print(f"Pooled reference: {args.num_windows} windows × {seq_len} "
          f"= {pooled_ref.size} Close values")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Forward noising sigma schedule ────────────────────────────────────────
    sigmas, quantiles = sigmas_from_quantiles(args.quantiles, args.P_mean, args.P_std)
    print(f"\nForward sigma levels (P_mean={args.P_mean}, P_std={args.P_std}):")
    for q, s in zip(quantiles, sigmas):
        print(f"  quantile {q:6.3f}  ->  sigma = {s:.4f}")

    # ── Run the denoise sampler, capturing every step ─────────────────────────
    print(f"\nRunning EDM sampler ({args.num_steps} steps, "
          f"{args.num_windows} distinct windows)…")
    snapshots = run_sampler_with_snapshots(
        processes, model, obs, cond_mask, observed_tp,
        args.num_steps, rho, device,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    if args.snapshot_steps is None:
        idxs = np.linspace(0, args.num_steps - 1, 6, dtype=int).tolist()
    else:
        idxs = [s for s in args.snapshot_steps if 0 <= s < args.num_steps]
    if not idxs:
        raise ValueError("No valid snapshot_steps in range [0, num_steps).")
    selected_steps = set(idxs)
    print(f"Denoise frame steps: {sorted(selected_steps)}")

    # ── Reproducibility metadata ──────────────────────────────────────────────
    run_info = {
        "split":             args.split,
        "num_windows":       args.num_windows,
        "windows":           windows_info,
        "checkpoint_folder": args.checkpoint_folder,
        "checkpoint_name":   args.checkpoint_name,
        "quantiles":         quantiles.tolist(),
        "sigmas":            sigmas.tolist(),
        "P_mean":            args.P_mean,
        "P_std":             args.P_std,
        "num_kde_samples":   args.num_kde_samples,
        "num_steps":         args.num_steps,
        "sigma_max":         sigma_max,
        "snapshot_steps":    sorted(selected_steps),
        "lo_q":              args.lo_q,
        "hi_q":              args.hi_q,
        "seed":              args.seed,
    }
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run/window info → {os.path.join(args.out_dir, 'run_info.json')}")

    # ── Forward zoom series ───────────────────────────────────────────────────
    print("\n── Forward (noising) zoom marginal ──")
    plot_forward_zoom(pooled_ref, ref_kde, sigmas, quantiles,
                      args.num_kde_samples, args.num_windows, args.out_dir,
                      args.lo_q, args.hi_q)

    # ── Denoise zoom series ───────────────────────────────────────────────────
    print("\n── Denoising zoom marginal ──")
    plot_denoise_zoom(snapshots, ref_kde, close_idx, selected_steps,
                      args.num_windows, args.out_dir, args.lo_q, args.hi_q)

    print(f"\nDone. All outputs in: {args.out_dir}")
    print("GIF assembly examples:")
    print("  ffmpeg -pattern_type glob -framerate 2 "
          f"-i '{args.out_dir}/forward_marginal_zoom_*.png' forward_zoom.gif")
    print("  ffmpeg -pattern_type glob -framerate 3 "
          f"-i '{args.out_dir}/denoise_marginal_zoom_*.png' denoise_zoom.gif")


if __name__ == "__main__":
    main()
