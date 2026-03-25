"""
generate_samples.py — unconditional generation of log_adj_close time-series.

Run:
    python generate_samples.py \
        --checkpoint_folder replication \
        --checkpoint_name   MY.pt \
        --n_samples         250 \
        --years_per_sample  39 \
        --seed              42

Output CSVs land in:
    <out_dir>/<checkpoint_name_without_ext>/

W&B logging (optional — omit --wandb_project to skip):
    python generate_samples.py ... \
        --wandb_project generation-gbm --wandb_entity thesis-giacomo-negri

On a headless HPC node matplotlib is forced to the 'Agg' backend so that
diagnostic plots produced inside Diffusion_Processes.reverse_process are
saved to PNG files instead of being displayed interactively.
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # must come before any other matplotlib import
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.model_core import CSDIModel
from src.utils.WIP_processes import Diffusion_Processes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic log_adj_close time-series with a trained CSDI checkpoint."
    )

    # Checkpoint
    p.add_argument("--checkpoint_folder", type=str, required=True,
                   help="Sub-folder inside checkpoints/ that holds the .pt file.")
    p.add_argument("--checkpoint_name", type=str, required=True,
                   help="Filename of the checkpoint (.pt).")

    # Kept for .sh compatibility — ignored at runtime
    p.add_argument("--mask_mode",     type=str, default="unconditional")
    p.add_argument("--cond_data_dir", type=str, default=None)

    # Volume
    p.add_argument("--n_samples", type=int, default=100,
                   help="Number of synthetic time-series to generate.")

    # Date range for output CSVs
    p.add_argument("--start_date", type=str, default="1986-01-01")
    p.add_argument("--end_date",   type=str, default="2025-12-31")
    p.add_argument("--years_per_sample", type=int, default=1,
                   help="Number of years of business days each sample covers.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for start-date sampling (-1 for a random seed).")

    # Reverse diffusion
    p.add_argument("--num_reverse_steps", type=int, default=None,
                   help="Override reverse diffusion steps (default: value in checkpoint config).")

    # Output base dir — checkpoint subfolder is appended automatically
    p.add_argument("--out_dir", type=str,
                   default=os.path.join("data", "generated", "replication"),
                   help="Base directory; a sub-folder named after the checkpoint is created here.")

    # W&B (all optional)
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Output dir = base_dir / checkpoint_stem ───────────────────────────────
    ckpt_stem = os.path.splitext(args.checkpoint_name)[0]
    out_dir   = os.path.join(args.out_dir, ckpt_stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint_path = os.path.join("checkpoints", args.checkpoint_folder, args.checkpoint_name)
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint loaded successfully.")
    config = ckpt["config"]

    # CHECK PRINTING
    print("Checkpoint config:")
    for section, params in config.items():
        print(f"  {section}:")
        for key, value in params.items():
            print(f"    {key}: {value}")

    target_dim = int(config["data"]["target_dim"])   # expected: 1 (log_adj_close)
    seq_len    = int(config["train"]["seq_len"])

    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Checkpoint : {args.checkpoint_name}")
    print(f"target_dim : {target_dim}  |  seq_len : {seq_len}")
    print(f"SDE type   : {config['process']['sde_type']}")
    print(f"Noise sched.: {config['process']['noise_schedule']} with N={config['process']['N']} steps")

    # ── W&B init (optional) ───────────────────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        run_name = args.wandb_run_name or f"gen_unconditional_{ckpt_stem}"
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_name,
            config  = {
                "checkpoint_folder":  args.checkpoint_folder,
                "checkpoint_name":    args.checkpoint_name,
                "mask_mode":          "unconditional",
                "n_samples":          args.n_samples,
                "num_reverse_steps":  args.num_reverse_steps,
                "start_date":         args.start_date,
                "end_date":           args.end_date,
                "sde_type":           config["process"]["sde_type"],
                "noise_schedule":     config["process"]["noise_schedule"],
                "sde_N":              config["process"]["N"],
                "model_steps":        config["process"]["model_steps"],
                "target_dim":         target_dim,
                "seq_len":            seq_len,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    # ── Diffusion processes ───────────────────────────────────────────────────
    processes = Diffusion_Processes(config["process"])
    num_reverse_steps = args.num_reverse_steps if args.num_reverse_steps is not None else processes.N
    print(f"Diffusion_Processes — SDE: {processes.sde_type}, N: {processes.N}, "
          f"model_steps: {processes.model_steps}")
    print(f"Reverse steps to use: {num_reverse_steps}")

    # ── Unconditional: all-zero observed data and mask ────────────────────────
    N_SAMPLES     = args.n_samples
    print(f"Number of samples: {N_SAMPLES}")
    observed_data = torch.zeros(N_SAMPLES, target_dim, seq_len, device=device)
    cond_mask     = torch.zeros(N_SAMPLES, target_dim, seq_len, device=device)

    # ── Run reverse diffusion ─────────────────────────────────────────────────
    observed_tp = (
        torch.linspace(0.0, 1.0, seq_len, device=device)
             .unsqueeze(0)
             .expand(N_SAMPLES, -1)
    )

    samples = processes.reverse_process(
        model            = model,
        shape            = (N_SAMPLES, target_dim, seq_len),
        observed_data    = observed_data,
        cond_mask        = cond_mask,
        observed_tp      = observed_tp,
        num_steps        = num_reverse_steps,
        probability_flow = False,
        device           = device,
    )  # → (N_SAMPLES, 1, L)
    print("Reverse diffusion completed. Samples shape:", samples.shape)
    print('Samples head:\n', samples[:2, 0, :5])  # print first 5 values of first 2 samples

    samples_global_mean = samples.mean().item()
    samples_global_std  = samples.std().item()
    samples_global_min  = samples.min().item()
    samples_global_max  = samples.max().item()

    print(f"\nGenerated tensor shape : {samples.shape}")
    print(f"Mean  : {samples_global_mean:.4f}")
    print(f"Std   : {samples_global_std:.4f}")
    print(f"Range : [{samples_global_min:.4f}, {samples_global_max:.4f}]")

    # ── Save diagnostic figures ───────────────────────────────────────────────
    wandb_images = {}
    for fig_num in plt.get_fignums():
        label    = "denoising_trajectory" if fig_num == 1 else "generated_samples"
        fig_path = os.path.join(out_dir, f"diagnostic_{label}.png")
        plt.figure(fig_num).savefig(fig_path, dpi=100, bbox_inches="tight")
        print(f"Saved diagnostic figure → {fig_path}")
        if use_wandb:
            wandb_images[label] = wandb.Image(fig_path)
    plt.close("all")

    # ── Build date pool ───────────────────────────────────────────────────────
    all_bdays    = pd.bdate_range(start=args.start_date, end=args.end_date)
    cutoff_date  = pd.Timestamp(args.end_date) - relativedelta(years=args.years_per_sample)
    valid_starts = all_bdays[all_bdays <= cutoff_date]
    print(f"Number of valid starts: {len(valid_starts)} (from {valid_starts[0].date()} to {valid_starts[-1].date()})")
    if len(valid_starts) == 0:
        raise ValueError(
            f"No valid start dates: end_date ({args.end_date}) minus "
            f"{args.years_per_sample} year(s) is before start_date ({args.start_date}). "
            "Widen the window or reduce --years_per_sample."
        )

    rng_seed      = None if args.seed == -1 else args.seed
    rng           = np.random.default_rng(rng_seed)
    start_indices = rng.integers(0, len(valid_starts), size=N_SAMPLES)

    samples_np = samples.detach().cpu().numpy()  # (N_SAMPLES, 1, L)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    for i in range(N_SAMPLES):
        series = samples_np[i, 0, :]  # (L,) — log_adj_close values

        sample_start = valid_starts[start_indices[i]]
        dates        = pd.bdate_range(start=sample_start, periods=seq_len)

        df = pd.DataFrame({
            "Date":          dates.strftime("%d/%m/%Y"),
            "log_adj_close": series,
        })

        filename = (
            f"FAKE_{i+1:04d}_"
            f"{dates[0].strftime('%Y%m%d')}_"
            f"{dates[-1].strftime('%Y%m%d')}_generated.csv"
        )
        df.to_csv(os.path.join(out_dir, filename), index=False)

    print(f"\nSaved {N_SAMPLES} CSV files → {out_dir}")

    # ── W&B — log summary metrics and diagnostic plots ────────────────────────
    if use_wandb:
        log_dict = {
            "gen/global_mean": samples_global_mean,
            "gen/global_std":  samples_global_std,
            "gen/global_min":  samples_global_min,
            "gen/global_max":  samples_global_max,
            "gen/n_samples":   N_SAMPLES,
        }
        log_dict.update(wandb_images)
        wandb.log(log_dict)
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()