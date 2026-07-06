#!/usr/bin/env python
"""
visualize_denoising_multi_window.py – Pooled marginal-distribution evolution
across many independent windows, for EDM-CSDI.

Unlike visualize_denoising.py (which repeats ONE window's OHL conditioning
`num_samples` times to draw `num_samples` denoised paths for that single
window), this script treats `--num_samples` as the number of DISTINCT
windows to draw from the split. Each window keeps its own OHL conditioning
and its own dates; one denoised path is generated per window. The marginal
KDE at each snapshot step is then pooled over (num_samples windows × seq_len
time steps), giving an aggregate view of how the generated Close marginal
converges toward the empirical marginal of real data, rather than how one
specific window's conditional distribution converges.

Only the marginal-distribution visualization is produced. The per-sequence
plots in visualize_denoising.py (sample evolution, price paths, ACF,
phase-space) assume a single shared reference sequence and don't carry over
once every batch element has a different underlying window/ground truth.

Frame plots are saved with zero-padded step indices for easy GIF assembly:
    ffmpeg -pattern_type glob -framerate 3 -i 'figures/denoising_multi_window/marginal_step_*.png' marginal.gif

Usage:
    python visualize_denoising_multi_window.py \\
        --checkpoint_folder final/edm \\
        --checkpoint_name EDM_REPLICATION_....pt \\
        --split val \\
        --num_samples 100 \\
        --num_steps 200 \\
        --snapshot_steps 0 20 40 60 80 100 120 140 160 180 195 199 \\
        --out_dir figures/denoising_multi_window \\
        --seed 42
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

_MARGINAL_COLOR = "#2196F3"   # blue


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


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
    return model, config, processes, feat_cols, close_idx, seq_len, rho, sigma_min, sigma_max


# ── Multi-window loading ──────────────────────────────────────────────────────

def load_windows(config, split, num_samples, repo_root, device,
                 close_idx, feat_cols, seq_len, seed, date_format="%d/%m/%Y"):
    """
    Reconstruct the train/val split and return tensors for `num_samples`
    DISTINCT windows, each with its own OHL conditioning and dates.

    Returns
    -------
    obs           : (num_samples, K, L) tensor on device — window i's own OHLC
    cond_mask     : (num_samples, K, L) tensor on device — 1=OHL, 0=Close
    observed_tp   : (num_samples, L)   tensor on device — window i's own date indices
    gt_close_batch: (num_samples, L)   numpy float32 — window i's own normalised close
    windows_info  : list of dict — metadata for every selected window (for reproducibility)
    """
    files, flat_index, train_indices, val_indices, date_to_idx, date_col = \
        reconstruct_split(config, repo_root, date_format=date_format)

    split_indices = train_indices if split == "train" else val_indices
    if not split_indices:
        raise ValueError(f"No {split} windows found in checkpoint config.")
    if num_samples > len(split_indices):
        raise ValueError(
            f"--num_samples={num_samples} exceeds the number of available "
            f"{split} windows ({len(split_indices)})."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(split_indices), size=num_samples, replace=False)

    K = len(feat_cols)
    obs_list, tp_list, gt_list, windows_info = [], [], [], []

    for global_idx in chosen:
        fi, start = flat_index[int(global_idx)]
        df = pd.read_csv(files[fi])

        win    = df[feat_cols].to_numpy(dtype=np.float32)[start: start + seq_len]  # (L, K)
        gt_win = win.T                                                              # (K, L)
        dates  = df[date_col].iloc[start: start + seq_len].tolist()
        tp_arr = np.array([date_to_idx[d] for d in dates], dtype=np.float32)       # (L,)

        obs_list.append(gt_win)
        tp_list.append(tp_arr)
        gt_list.append(gt_win[close_idx])
        windows_info.append({
            "global_idx":  int(global_idx),
            "file":        os.path.basename(files[fi]),
            "start":       int(start),
            "date_start":  dates[0],
            "date_end":    dates[-1],
        })

    obs_np = np.stack(obs_list, axis=0)   # (num_samples, K, L)
    tp_np  = np.stack(tp_list, axis=0)    # (num_samples, L)
    gt_close_batch = np.stack(gt_list, axis=0)  # (num_samples, L)

    obs = torch.from_numpy(obs_np).to(device)
    cond_mask = torch.ones(num_samples, K, seq_len, device=device)
    cond_mask[:, close_idx, :] = 0.0
    observed_tp = torch.from_numpy(tp_np).to(device)

    print(f"Selected {num_samples} distinct {split} windows "
          f"(global indices: {sorted(int(i) for i in chosen)[:10]}"
          f"{'...' if num_samples > 10 else ''})")
    return obs, cond_mask, observed_tp, gt_close_batch, windows_info


# ── Sampling with full snapshot capture ──────────────────────────────────────

def run_sampler_with_snapshots(processes, model, obs, cond_mask, observed_tp,
                               num_steps, rho, device, sigma_min=None, sigma_max=None):
    """
    Run the Heun sampler capturing D_x at every step.

    Returns
    -------
    snapshots : list of (step: int, sigma: float, D_x: cpu Tensor (B, K, L))
                from step 0 (σ_max, pure noise) to step num_steps-1 (σ_min).
                Here B = num_samples = number of distinct windows.
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


# ── Visualisation: Pooled marginal distribution panels ───────────────────────

def plot_marginals_pooled(snapshots, gt_close_batch, close_idx, selected_steps,
                          num_windows, out_dir):
    """
    One PNG per selected step: KDE of generated Close, pooled over all
    (num_windows × seq_len) values, vs. a reference KDE pooled the same way
    over all windows' ground-truth Close sequences.

    Axes are fixed across all frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_ref = gt_close_batch.ravel()   # (num_windows * L,)
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
        # D_x: (num_windows, K, L) → pool over windows AND time.
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
        ax.plot(x_grid, ref_dens, "k--", lw=1.8, alpha=0.85,
                label=f"Reference ({num_windows} windows pooled)")
        ax.fill_between(x_grid, dens, alpha=0.35, color=_MARGINAL_COLOR)
        ax.plot(x_grid, dens, color=_MARGINAL_COLOR, lw=1.6,
                label=f"Step {step}  (σ = {sigma:.3f})")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, y_max)
        ax.set_xlabel("Normalised close log-return")
        ax.set_ylabel("Density")
        ax.set_title(f"Pooled marginal distribution ({num_windows} windows) — "
                     f"denoising step {step}  (σ = {sigma:.3f})")
        ax.legend()
        _save(fig, os.path.join(out_dir, f"marginal_step_{step:04d}.png"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pooled marginal-distribution evolution across many "
                    "independent windows, for EDM-CSDI."
    )
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)
    p.add_argument("--split",             type=str, default="val",
                   choices=["train", "val"])
    p.add_argument("--num_samples",       type=int, default=50,
                   help="Number of DISTINCT windows to draw (each contributes "
                        "one denoised path, conditioned on its own OHL).")
    p.add_argument("--num_steps",         type=int, default=50,
                   help="Heun denoising steps (default: 50).")
    p.add_argument("--sigma_max",         type=float, default=None,
                   help="Override the sampler's sigma_max (top of the noise "
                        "schedule). Default: the checkpoint's own sigma_max.")
    p.add_argument("--snapshot_steps",    type=int, nargs="+", default=None,
                   help="Step indices to render as marginal frames. "
                        "Default: 5 evenly-spaced steps including first and last.")
    p.add_argument("--out_dir",           type=str, default="figures/denoising_multi_window")
    p.add_argument("--date_format",       type=str, default="%d/%m/%Y")
    p.add_argument("--seed",              type=int, default=42,
                   help="Seeds both torch/sampler RNG and the window selection RNG.")
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

    if args.sigma_max is not None:
        print(f"Overriding sigma_max: {sigma_max} → {args.sigma_max}")
        sigma_max = args.sigma_max

    # ── Load num_samples distinct windows ─────────────────────────────────────
    repo_root = os.path.abspath(os.path.dirname(__file__))
    obs, cond_mask, observed_tp, gt_close_batch, windows_info = load_windows(
        config, args.split, args.num_samples, repo_root, device,
        close_idx, feat_cols, seq_len, args.seed,
        date_format=args.date_format,
    )

    # ── Save chosen windows/run info for later reuse ──────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    run_info = {
        "split":              args.split,
        "num_windows":        args.num_samples,
        "windows":            windows_info,
        "checkpoint_folder":  args.checkpoint_folder,
        "checkpoint_name":    args.checkpoint_name,
        "num_steps":          args.num_steps,
        "sigma_max":          sigma_max,
        "seed":               args.seed,
    }
    run_info_path = os.path.join(args.out_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run/window info → {run_info_path}")

    # ── Run sampler — capture all steps ──────────────────────────────────────
    print(f"\nRunning EDM sampler ({args.num_steps} steps, "
          f"{args.num_samples} distinct windows)…")
    snapshots = run_sampler_with_snapshots(
        processes, model, obs, cond_mask, observed_tp,
        args.num_steps, rho, device,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )

    # ── Determine which steps to render as marginal frames ────────────────────
    if args.snapshot_steps is None:
        idxs = np.linspace(0, args.num_steps - 1, 5, dtype=int).tolist()
    else:
        idxs = [s for s in args.snapshot_steps if 0 <= s < args.num_steps]
    if not idxs:
        raise ValueError("No valid snapshot_steps in range [0, num_steps).")
    selected_steps = set(idxs)
    print(f"\nFrame steps for pooled marginal: {sorted(selected_steps)}")

    print("\n── Pooled marginal distribution across windows ──")
    plot_marginals_pooled(snapshots, gt_close_batch, close_idx, selected_steps,
                          args.num_samples, args.out_dir)

    print(f"\nDone. All outputs in: {args.out_dir}")
    print("GIF assembly example:")
    print("  ffmpeg -pattern_type glob -framerate 3 "
          f"-i '{args.out_dir}/marginal_step_*.png' marginal.gif")


if __name__ == "__main__":
    main()
