#!/usr/bin/env python
"""
visualize_denoising.py  –  Denoising process visualizations for EDM-CSDI.

Generates six sets of plots illustrating how the EDM Heun sampler progressively
transforms pure noise into Close log-return sequences conditioned on the OHL
context of a single training/validation window.

Frame plots are saved with zero-padded step indices for easy GIF assembly:
    ffmpeg -pattern_type glob -framerate 3 -i 'figures/denoising/marginal_step_*.png' marginal.gif

Usage:
    python visualize_denoising.py \\
        --checkpoint_folder ohlc_conditional \\
        --checkpoint_name MY.pt \\
        --split val \\
        --window_idx 0 \\
        --num_samples 5 \\
        --num_steps 50 \\
        --snapshot_steps 0 5 15 30 49 \\
        --out_dir figures/denoising \\
        --norm_stats data/fake_fts_processed/normalization_stats.csv \\
        --seed 42

python visualize_denoising.py --checkpoint_folder final/edm --checkpoint_name EDM_REPLICATION_CLOS_ep-500_step-44500_lr-8e-04_ch-128_layers-6_nheads-4_diffemb-256_sd-1.0_Pm--1.4_Ps-1.8_20260602_201915.pt --split train --window_idx 0 --num_samples 10 --num_steps 200 --snapshot_steps 0 20 40 60 80 100 120 140 160 180 195 199 --norm_stats data/normalization_stats_norm_replication.csv --seed 42 --out_dir figures/denoising --seed 42
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
import pandas as pd
import torch
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.model_core import CSDIModel
from src.utils.WIP_processes import Diffusion_Processes
from generate_samples import reconstruct_split


# ── Plotting style ────────────────────────────────────────────────────────────
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
    "marginal":  "#2196F3",   # blue
    "evolution": "#E91E63",   # pink
    "prices":    "#4CAF50",   # green
    "acf":       "#FF9800",   # orange
}


# ── Utility functions ─────────────────────────────────────────────────────────

def _acf_numpy(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation at lags 1..max_lag via numpy (O(n log n))."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    var = np.var(x)
    if var < 1e-12 or n <= max_lag:
        return np.zeros(max_lag)
    # Full correlation then normalise; c[0] = 1 (lag 0), c[1] = lag 1, etc.
    c = np.correlate(x, x, mode="full")
    c = c[n - 1:] / (var * n)
    return c[1: max_lag + 1]   # lags 1..max_lag


def _to_price_path(returns: np.ndarray) -> np.ndarray:
    """Log-returns → relative price path with S_0 = 1."""
    return np.exp(np.concatenate([[0.0], np.cumsum(returns)]))


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _array_signature(arr: np.ndarray) -> dict:
    """
    Plain-value fingerprint of an array (first/last few entries, sum, mean,
    std) — enough to eyeball or text-diff whether the same quantity (e.g.
    gt_close, or the denormalised reference price path) came out identical
    across two scripts/machines that are supposed to compute it the same way.
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return {
        "n":      int(arr.size),
        "first3": [round(float(v), 8) for v in arr[:3]],
        "last3":  [round(float(v), 8) for v in arr[-3:]],
        "sum":    round(float(arr.sum()), 8),
        "mean":   round(float(arr.mean()), 8),
        "std":    round(float(arr.std()), 8),
    }


# ── Normalization helpers ─────────────────────────────────────────────────────

def load_norm_stats(path: str, feat_cols: list) -> dict:
    """
    Load per-channel stats written by ohlc_to_returns.py.

    The CSV has the feature name as the row index and columns 'mean' / 'std'.
    Returns {feature: (mean, std)}.
    """
    df = pd.read_csv(path, index_col=0)
    return {
        feat: (float(df.loc[feat, "mean"]), float(df.loc[feat, "std"]))
        for feat in feat_cols
        if feat in df.index
    }


def denorm(x_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x_norm * std + mean


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(args, device):
    ckpt_path = os.path.join("checkpoints", args.checkpoint_folder, args.checkpoint_name)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    target_dim = int(config["data"]["target_dim"])
    seq_len    = int(config["train"]["seq_len"])
    columns    = tuple(config["data"].get("columns", ("date", "log_adj_close")))
    feat_cols  = list(columns[1:])
    close_idx  = feat_cols.index("close")

    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    processes = Diffusion_Processes(config["process"])
    rho       = float(config.get("edm", {}).get("rho", 7.0))

    try:
        sigma_min = float(processes.sde.sigma_schedule.sigma_min)
        sigma_max = float(processes.sde.sigma_schedule.sigma_max)
    except AttributeError:
        sigma_min = float(config["process"].get("sigma_min", 0.002))
        sigma_max = float(config["process"].get("sigma_max", 80.0))

    print(f"  feat_cols={feat_cols}  close_idx={close_idx}  seq_len={seq_len}  "
          f"σ=[{sigma_min}, {sigma_max}]  rho={rho}")
    print(f"  data_root={config['train'].get('data_root')}  "
          f"stride={config['train'].get('stride')}  "
          f"seed={config['train'].get('seed')}  "
          f"val_split_ratio={config['train'].get('val_split_ratio')}")
    return model, config, processes, feat_cols, close_idx, seq_len, rho, sigma_min, sigma_max


# ── Window loading ────────────────────────────────────────────────────────────

def load_window(config, split, window_idx, repo_root, device, num_samples,
                close_idx, feat_cols, seq_len, date_format="%d/%m/%Y"):
    """
    Reconstruct the train/val split and return tensors for one window.

    Returns
    -------
    obs          : (num_samples, K, L) tensor on device
    cond_mask    : (num_samples, K, L) tensor on device — 1=OHL, 0=Close
    observed_tp  : (num_samples, L)   tensor on device — global date indices
    gt_close     : (L,) numpy float32 — ground-truth normalised close log-returns
    window_info  : dict — metadata identifying the chosen window (for reproducibility)
    """
    files, flat_index, train_indices, val_indices, date_to_idx, date_col = \
        reconstruct_split(config, repo_root, date_format=date_format)

    split_indices = train_indices if split == "train" else val_indices
    if not split_indices:
        raise ValueError(f"No {split} windows found in checkpoint config.")
    if window_idx >= len(split_indices):
        raise IndexError(
            f"window_idx={window_idx} out of range for {split} split "
            f"(size={len(split_indices)})."
        )

    global_idx = split_indices[window_idx]
    fi, start  = flat_index[global_idx]
    df = pd.read_csv(files[fi])

    win    = df[feat_cols].to_numpy(dtype=np.float32)[start: start + seq_len]   # (L, K)
    gt_win = win.T                                                                # (K, L)
    dates  = df[date_col].iloc[start: start + seq_len].tolist()
    tp_arr = np.array([date_to_idx[d] for d in dates], dtype=np.float32)        # (L,)

    K   = len(feat_cols)
    obs = (
        torch.from_numpy(gt_win).unsqueeze(0).expand(num_samples, -1, -1).to(device)
    )                                                                            # (B, K, L)
    cond_mask = torch.ones(num_samples, K, seq_len, device=device)
    cond_mask[:, close_idx, :] = 0.0

    observed_tp = (
        torch.from_numpy(tp_arr).to(device).unsqueeze(0).expand(num_samples, -1)
    )                                                                            # (B, L)

    gt_close = gt_win[close_idx]                                                 # (L,)
    gt_close_sig = _array_signature(gt_close)
    print(f"Window [{split}][{window_idx}]  file={os.path.basename(files[fi])}  "
          f"start={start}  dates={dates[0]}..{dates[-1]}")
    print(f"  gt_close signature: {gt_close_sig}")

    window_info = {
        "split":             split,
        "window_idx":        window_idx,
        "file":              os.path.basename(files[fi]),
        "start":             int(start),
        "seq_len":           int(seq_len),
        "date_start":        dates[0],
        "date_end":          dates[-1],
        "gt_close_signature": gt_close_sig,
    }
    return obs, cond_mask, observed_tp, gt_close, window_info


# ── Sampling with full snapshot capture ──────────────────────────────────────

def run_sampler_with_snapshots(processes, model, obs, cond_mask, observed_tp,
                               num_steps, rho, device, sigma_min=None, sigma_max=None):
    """
    Run the Heun sampler capturing D_x at every step.

    sigma_min / sigma_max default to the checkpoint's own values (via
    processes.sde.sigma_schedule) when left as None; pass explicit values
    to override the noise range used by the sampler.

    Returns
    -------
    snapshots : list of (step: int, sigma: float, D_x: cpu Tensor (B, K, L))
                from step 0 (σ_max, pure noise) to step num_steps-1 (σ_min).
    """
    shape = tuple(obs.shape)
    _, snapshots = processes.edm_sampler(
        model          = model,
        shape          = shape,
        observed_data  = obs,
        cond_mask      = cond_mask,
        observed_tp    = observed_tp,
        num_steps      = num_steps,
        rho            = rho,
        sigma_min      = sigma_min,
        sigma_max      = sigma_max,
        device         = device,
        snapshot_steps = list(range(num_steps)),
    )
    print(f"Captured {len(snapshots)} snapshots.")
    return snapshots


# ── Visualisation 0: Prior noise vs reference ────────────────────────────────

def plot_prior_vs_reference(gt_close, close_idx, num_samples, shape, sigma_max,
                            device, out_dir):
    """
    One PNG: raw prior noise sample(s) x ~ N(0, sigma_max^2 I) — the sampler's
    literal starting point, before any denoising step — plotted directly
    against the reference (normalised) close log-returns.

    This does NOT run the model or the sampler; it draws the same prior
    (`torch.randn(*shape) * sigma_max`) used at the top of `edm_sampler`.
    """
    os.makedirs(out_dir, exist_ok=True)
    L  = len(gt_close)
    xs = np.arange(L)

    x_prior     = torch.randn(*shape, device=device) * sigma_max
    prior_close = x_prior[:, close_idx, :].cpu().numpy()   # (num_samples, L)

    colors = cm.tab10(np.linspace(0.0, 0.9, num_samples))

    y_lo = min(float(gt_close.min()), float(prior_close.min()))
    y_hi = max(float(gt_close.max()), float(prior_close.max()))
    pad  = 0.10 * (y_hi - y_lo)
    y_lo -= pad;  y_hi += pad

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xs, gt_close, "k--", lw=1.8, alpha=0.8, label="Reference (true close)")
    for s in range(num_samples):
        ax.plot(xs, prior_close[s], color=colors[s], lw=1.0, alpha=0.6,
                label=f"Prior sample {s}")
    ax.set_xlim(0, L - 1)
    ax.set_ylim(y_lo, y_hi)
    # ax.set_xlabel("Time step")
    # ax.set_ylabel("Normalised log-return")
    # ax.set_title(f"Prior noise vs reference  ($\\sigma_{{max}}$ = {sigma_max:.3f})")
    _save(fig, os.path.join(out_dir, "prior_vs_reference.png"))


# ── Visualisation 1: Marginal distribution panels ────────────────────────────

def plot_marginals(snapshots, gt_close, close_idx, selected_steps, out_dir):
    """
    One PNG per selected step: KDE of generated Close vs. reference KDE.
    Axes are fixed across all frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_ref = gt_close.ravel()
    x_lo = np.quantile(all_ref, 0.001)
    x_hi = np.quantile(all_ref, 0.999)
    x_grid = np.linspace(x_lo, x_hi, 300)

    ref_kde  = gaussian_kde(np.clip(all_ref, x_lo, x_hi))
    ref_dens = ref_kde(x_grid)

    # Pre-compute all snapshot densities to fix the y-axis before plotting.
    snap_data = {}
    for step, sigma, D_x in snapshots:
        if step not in selected_steps:
            continue
        vals = np.clip(D_x[:, close_idx, :].numpy().ravel(), x_lo, x_hi)
        try:
            dens = gaussian_kde(vals)(x_grid)
        except np.linalg.LinAlgError:
            dens = np.zeros_like(x_grid)
        snap_data[step] = (sigma, dens)

    if not snap_data:
        print("  [marginals] no matching snapshot steps — skipping")
        return

    y_max = max(ref_dens.max(), *(d.max() for _, d in snap_data.values())) * 1.15

    for step, (sigma, dens) in sorted(snap_data.items()):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_grid, ref_dens, "k--", lw=1.8, alpha=0.85, label="Reference")
        ax.fill_between(x_grid, dens, alpha=0.35, color=_COLORS["marginal"])
        ax.plot(x_grid, dens, color=_COLORS["marginal"], lw=1.6,
                label=f"Step {step}  (σ = {sigma:.3f})")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, y_max)
        # ax.set_xlabel("Normalised close log-return")
        # ax.set_ylabel("Density")
        ax.set_title(f"Step {step}  (σ = {sigma:.3f})")
        _save(fig, os.path.join(out_dir, f"marginal_step_{step:04d}.png"))


# ── Visualisation 2: Sample path evolution ───────────────────────────────────

def plot_evolution(snapshots, gt_close, close_idx, sample_idx, selected_steps,
                   sigma_max, out_dir):
    """
    One PNG per selected step: reference sequence (fixed dashed) + denoised path.
    Y-axis spans [reference range] ∪ [initial noise range] and is fixed across frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    L  = len(gt_close)
    xs = np.arange(L)

    y_lo = min(float(gt_close.min()), -3.0 * sigma_max)
    y_hi = max(float(gt_close.max()),  3.0 * sigma_max)
    pad  = 0.10 * (y_hi - y_lo)
    y_lo -= pad;  y_hi += pad

    rendered = 0
    for step, sigma, D_x in snapshots:
        if step not in selected_steps:
            continue
        gen = D_x[sample_idx, close_idx, :].numpy()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(xs, gt_close, "k--", lw=1.8, alpha=0.75, label="Reference (true close)")
        ax.plot(xs, gen, color=_COLORS["evolution"], lw=1.4, alpha=0.9,
                label=f"Denoised  (step {step}, σ = {sigma:.3f})")
        ax.set_xlim(0, L - 1)
        ax.set_ylim(y_lo, y_hi)
        # ax.set_xlabel("Time step")
        # ax.set_ylabel("Normalised log-return")
        ax.set_title(f"Step {step}")
        _save(fig, os.path.join(out_dir, f"evolution_step_{step:04d}.png"))
        rendered += 1

    if rendered == 0:
        print("  [evolution] no matching snapshot steps — skipping")


# ── Visualisation 3: Price path evolution ────────────────────────────────────

def plot_prices(snapshots, gt_close, close_idx, sample_idx, selected_steps,
                norm_stats, close_feat, out_dir):
    """
    One PNG per selected step: relative price paths (S_0 = 1).
    Skipped if normalization stats for the close feature are unavailable.
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

    # Pre-compute all price paths to determine fixed y-axis.
    step_prices = {}
    all_prices  = [gt_price]
    for step, sigma, D_x in snapshots:
        if step not in selected_steps:
            continue
        gen_denorm = denorm(D_x[sample_idx, close_idx, :].numpy(), mean, std)
        gen_price  = _to_price_path(gen_denorm)
        step_prices[step] = (sigma, gen_price)
        all_prices.append(gen_price)

    if not step_prices:
        print("  [prices] no matching snapshot steps — skipping")
        return

    y_lo = min(float(p.min()) for p in all_prices)
    y_hi = max(float(p.max()) for p in all_prices)
    pad  = 0.10 * (y_hi - y_lo)
    y_lo -= pad;  y_hi += pad

    for step, (sigma, gen_price) in sorted(step_prices.items()):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(xs, gt_price, "k--", lw=1.8, alpha=0.75, label="Reference price path")
        ax.plot(xs, gen_price, color=_COLORS["prices"], lw=1.4, alpha=0.9,
                label=f"Denoised  (step {step}, σ = {sigma:.3f})")
        ax.set_xlim(0, L1 - 1)
        ax.set_ylim(y_lo, y_hi)
        # ax.set_xlabel("Time step")
        # ax.set_ylabel("Relative price  ($S_0 = 1$)")
        ax.set_title(f"Step {step}")
        _save(fig, os.path.join(out_dir, f"prices_step_{step:04d}.png"))


# ── Visualisation 4: ACF evolution ───────────────────────────────────────────

def plot_acf_evolution(snapshots, gt_close, close_idx, sample_idx, selected_steps,
                       max_lag, out_dir):
    """
    One PNG per selected step: ACF of |returns| for the denoised path vs. reference.
    Y-axis fixed across frames.  Volatile-clustering pattern should emerge as σ → 0.
    """
    os.makedirs(out_dir, exist_ok=True)
    max_lag = min(max_lag, len(gt_close) - 2)
    lags    = np.arange(1, max_lag + 1)

    ref_acf = _acf_numpy(np.abs(gt_close), max_lag)

    step_acfs = {}
    all_acfs  = [ref_acf]
    for step, sigma, D_x in snapshots:
        if step not in selected_steps:
            continue
        seq = D_x[sample_idx, close_idx, :].numpy()
        a   = _acf_numpy(np.abs(seq), max_lag)
        step_acfs[step] = (sigma, a)
        all_acfs.append(a)

    if not step_acfs:
        print("  [acf] no matching snapshot steps — skipping")
        return

    y_lo = min(float(a.min()) for a in all_acfs) - 0.03
    y_hi = max(float(a.max()) for a in all_acfs) * 1.15

    for step, (sigma, acf_vals) in sorted(step_acfs.items()):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lags, ref_acf, "k--", lw=1.8, alpha=0.75, label="Reference")
        ax.plot(lags, acf_vals, color=_COLORS["acf"], lw=1.6, alpha=0.9,
                label=f"Denoised  (step {step}, σ = {sigma:.3f})")
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.set_xlim(1, max_lag)
        ax.set_ylim(y_lo, y_hi)
        # ax.set_xlabel("Lag")
        # ax.set_ylabel("ACF of |return|")
        ax.set_title(f"Step {step}")
        _save(fig, os.path.join(out_dir, f"acf_step_{step:04d}.png"))


# ── Visualisation 5: Preconditioning coefficients ────────────────────────────

def plot_preconditioning(sigma_min: float, sigma_max: float,
                         sigma_data: float, out_dir: str):
    """
    c_skip(σ), c_out(σ), c_in(σ) vs σ on a log scale.
    Pure math — no model evaluation needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    sigmas = np.logspace(np.log10(sigma_min * 0.5), np.log10(sigma_max * 2.0), 500)

    c_skip = sigma_data ** 2 / (sigmas ** 2 + sigma_data ** 2)
    c_out  = sigmas * sigma_data / np.sqrt(sigmas ** 2 + sigma_data ** 2)
    c_in   = 1.0 / np.sqrt(sigmas ** 2 + sigma_data ** 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigmas, c_skip, lw=2.2, label=r"$c_\mathrm{skip}(\sigma)$")
    ax.plot(sigmas, c_out,  lw=2.2, label=r"$c_\mathrm{out}(\sigma)$")
    ax.plot(sigmas, c_in,   lw=2.2, label=r"$c_\mathrm{in}(\sigma)$")
    ax.axvline(sigma_data, color="dimgray", ls="--", lw=1.4,
               label=rf"$\sigma_{{\mathrm{{data}}}} = {sigma_data}$  (crossover)")
    ax.axhline(0.5, color="lightgray", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(sigmas[0], sigmas[-1])
    ax.set_ylim(-0.05, 1.1)
    # ax.set_xlabel(r"Noise level $\sigma$")
    # ax.set_ylabel("Coefficient value")
    # ax.set_title("EDM preconditioning coefficients")
    _save(fig, os.path.join(out_dir, "preconditioning.png"))


# ── Visualisation 6: Phase-space trajectory ──────────────────────────────────

def plot_phase_space(snapshots, gt_close, close_idx, num_samples,
                     t0, out_dir):
    """
    For one time position t0, plot the denoised estimate x̂_0(σ) for each
    sample across all σ levels.  Shows convergence from scattered (high σ)
    to clustered near the ground truth (low σ).
    X-axis is inverted so high σ is on the left (denoising goes left → right).
    """
    os.makedirs(out_dir, exist_ok=True)

    if t0 is None:
        t0 = int(np.argmax(np.abs(gt_close)))

    gt_val = float(gt_close[t0])
    colors = cm.tab10(np.linspace(0.0, 0.7, num_samples))

    # Collect (sigma, value) pairs per sample across all captured steps.
    sigmas_all = [sigma for _, sigma, _ in snapshots]
    traces     = [
        [float(D_x[s, close_idx, t0]) for _, _, D_x in snapshots]
        for s in range(num_samples)
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, (trace, color) in enumerate(zip(traces, colors)):
        ax.plot(sigmas_all, trace, "-o", markersize=2.5, lw=0.9, alpha=0.75,
                color=color, label=f"Sample {s}")

    ax.axhline(gt_val, color="black", ls="--", lw=1.8, alpha=0.9,
               label=f"Reference  (t = {t0})")
    ax.set_xscale("log")
    ax.invert_xaxis()   # high σ on left → denoising goes left-to-right
    # ax.set_xlabel(r"Noise level $\sigma$   (← denoising direction)")
    # ax.set_ylabel(f"Denoised estimate  $\\hat{{x}}_0(\\sigma)$ at $t = {t0}$")
    # ax.set_title("Phase-space trajectory — convergence of denoised estimates")
    _save(fig, os.path.join(out_dir, "phase_space.png"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Denoising process visualizations for EDM-CSDI."
    )
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)
    p.add_argument("--split",             type=str, default="val",
                   choices=["train", "val"])
    p.add_argument("--window_idx",        type=int, default=0,
                   help="Index into the chosen split (default: 0).")
    p.add_argument("--num_samples",       type=int, default=5,
                   help="Number of independent generated paths.")
    p.add_argument("--num_steps",         type=int, default=50,
                   help="Heun denoising steps (default: 50).")
    p.add_argument("--sigma_max",         type=float, default=None,
                   help="Override the sampler's sigma_max (top of the noise "
                        "schedule). Default: the checkpoint's own sigma_max.")
    p.add_argument("--snapshot_steps",    type=int, nargs="+", default=None,
                   help="Step indices to use for frame-based plots. "
                        "Default: 5 evenly-spaced steps including first and last.")
    p.add_argument("--sample_idx",        type=int, default=0,
                   help="Which of the num_samples paths to use for "
                        "evolution / ACF / price plots (default: 0).")
    p.add_argument("--acf_max_lag",       type=int, default=100,
                   help="Maximum ACF lag (default: 100).")
    p.add_argument("--phase_space_t0",    type=int, default=None,
                   help="Time position for phase-space plot. "
                        "Default: position of max |gt_close|.")
    p.add_argument("--out_dir",           type=str, default="figures/denoising")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model and config ─────────────────────────────────────────────────
    (model, config, processes, feat_cols, close_idx,
     seq_len, rho, sigma_min, sigma_max) = load_model(args, device)
    close_feat = feat_cols[close_idx]
    sigma_data = 1.0   # per-channel z-score normalisation ensures σ_data = 1

    if args.sigma_max is not None:
        print(f"Overriding sigma_max: {sigma_max} → {args.sigma_max}")
        sigma_max = args.sigma_max

    # Validate sample_idx
    if args.sample_idx >= args.num_samples:
        raise ValueError(
            f"--sample_idx={args.sample_idx} must be < --num_samples={args.num_samples}"
        )

    # ── Load one window ───────────────────────────────────────────────────────
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
        "sample_idx":        args.sample_idx,
        "num_steps":         args.num_steps,
        "sigma_max":         sigma_max,
        "seed":              args.seed,
    }
    run_info_path = os.path.join(args.out_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run/window info → {run_info_path}")

    print("\n── Vis 0: Prior noise vs reference ──")
    plot_prior_vs_reference(gt_close, close_idx, args.num_samples, tuple(obs.shape),
                            sigma_max, device, args.out_dir)

    # ── Run sampler — capture all steps ──────────────────────────────────────
    print(f"\nRunning EDM sampler ({args.num_steps} steps, "
          f"{args.num_samples} samples)…")
    snapshots = run_sampler_with_snapshots(
        processes, model, obs, cond_mask, observed_tp,
        args.num_steps, rho, device,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    # ── Determine which steps to render as per-frame PNGs ────────────────────
    if args.snapshot_steps is None:
        idxs = np.linspace(0, args.num_steps - 1, 5, dtype=int).tolist()
    else:
        idxs = [s for s in args.snapshot_steps if 0 <= s < args.num_steps]
    if not idxs:
        raise ValueError("No valid snapshot_steps in range [0, num_steps).")
    selected_steps = set(idxs)
    print(f"\nFrame steps for vis 1-4: {sorted(selected_steps)}")

    # ── Load normalisation stats (optional, needed for price paths) ───────────
    norm_stats: dict = {}
    if args.norm_stats:
        if os.path.isfile(args.norm_stats):
            norm_stats = load_norm_stats(args.norm_stats, feat_cols)
            print(f"Loaded norm stats for: {list(norm_stats.keys())}")
        else:
            print(f"[warn] norm_stats not found: {args.norm_stats}  "
                  f"— price-path visualisation will be skipped")

    # ── Run all six visualisations ────────────────────────────────────────────
    print("\n── Vis 1: Marginal distributions ──")
    plot_marginals(snapshots, gt_close, close_idx, selected_steps, args.out_dir)

    print("\n── Vis 2: Sample path evolution ──")
    plot_evolution(snapshots, gt_close, close_idx, args.sample_idx,
                   selected_steps, sigma_max, args.out_dir)

    print("\n── Vis 3: Price path evolution ──")
    plot_prices(snapshots, gt_close, close_idx, args.sample_idx,
                selected_steps, norm_stats, close_feat, args.out_dir)

    # print("\n── Vis 4: ACF (volatility clustering) evolution ──")
    # plot_acf_evolution(snapshots, gt_close, close_idx, args.sample_idx,
    #                    selected_steps, args.acf_max_lag, args.out_dir)

    # print("\n── Vis 5: EDM preconditioning coefficients ──")
    # plot_preconditioning(sigma_min, sigma_max, sigma_data, args.out_dir)

    # print("\n── Vis 6: Phase-space trajectory ──")
    # plot_phase_space(snapshots, gt_close, close_idx, args.num_samples,
    #                  args.phase_space_t0, args.out_dir)

    print(f"\nDone. All outputs in: {args.out_dir}")
    print("GIF assembly example:")
    print("  ffmpeg -pattern_type glob -framerate 3 "
          f"-i '{args.out_dir}/marginal_step_*.png' marginal.gif")


if __name__ == "__main__":
    main()