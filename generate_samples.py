"""
generate_samples.py — unconditional generation of log_adj_close time-series.

Run:
    python generate_samples.py \
        --checkpoint_folder replication \
        --checkpoint_name   MY.pt \
        --n_samples         250 \
        # --years_per_sample  39 \
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
    # p.add_argument("--years_per_sample", type=int, default=1,
    #                help="Number of years of business days each sample covers.")
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

    # Debug mode: enables ODE reverse-step consistency diagnostic
    p.add_argument("--debug", action="store_true", default=False,
                   help="Run ODE reverse-step consistency diagnostic and log to W&B.")
    p.add_argument("--probability_flow", action="store_true", default=False,
                   help="Use probability-flow ODE sampler (required for consistency diagnostic).")
    p.add_argument("--use_heun_ode", action="store_true", default=False,
                   help="Use Heun's method ODE sampler (automatically implies --probability_flow).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.use_heun_ode:
        _sampler_label = "HEUN_ODE"
    elif args.probability_flow:
        _sampler_label = "ODE"
    else:
        _sampler_label = "SDE"

    # ── Seed (before any stochastic operation) ────────────────────────────────
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # ── Output dir = base_dir / checkpoint_stem ───────────────────────────────
    ckpt_stem  = os.path.splitext(args.checkpoint_name)[0]
    _parts     = ckpt_stem.split("_")
    _noise_idx = next((i for i, p in enumerate(_parts) if p.startswith("noise-")), len(_parts) - 3)
    _short_stem = "_".join(_parts[:_noise_idx + 1]) + "_" + "_".join(_parts[-2:])
    out_dir   = os.path.join(args.out_dir, f"{_sampler_label}_{_short_stem}")
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
                "mask_mode":          "unconditional", # hardcoded because of replication attempt
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

    use_ode   = args.probability_flow
    use_heun  = args.use_heun_ode
    run_debug = args.debug and (use_ode or use_heun)
    disc_out  = [] if run_debug else None

    samples = processes.reverse_process(
        model            = model,
        shape            = (N_SAMPLES, target_dim, seq_len),
        observed_data    = observed_data,
        cond_mask        = cond_mask,
        observed_tp      = observed_tp,
        num_steps        = num_reverse_steps,
        probability_flow = use_ode or use_heun,
        device           = device,
        debug            = run_debug,
        disc_out         = disc_out,
        use_heun         = use_heun,
    )  # → (N_SAMPLES, 1, L)
    print("Reverse diffusion completed. Samples shape:", samples.shape)
    print('Samples head:\n', samples[:2, 0, :5])  # print first 5 values of first 2 samples

    # ── ODE consistency diagnostic summary ───────────────────────────────────
    if run_debug and disc_out:
        steps    = np.array([d["step"]       for d in disc_out])
        ts_diag  = np.array([d["t"]          for d in disc_out])
        l2s      = np.array([d["l2"]         for d in disc_out])
        maes     = np.array([d["mae"]        for d in disc_out])
        max_aes  = np.array([d["max_ae"]     for d in disc_out])
        cum_l2s  = np.array([d["cum_l2_sum"] for d in disc_out])  # Σ L2_i up to each check
        cum_rmss = np.array([d["cum_rms"]    for d in disc_out])  # running RMS of L2s

        summary = {
            "consistency/l2_mean":         float(l2s.mean()),
            "consistency/l2_p50":          float(np.percentile(l2s,  50)),
            "consistency/l2_p95":          float(np.percentile(l2s,  95)),
            "consistency/l2_max":          float(l2s.max()),
            "consistency/mae_mean":        float(maes.mean()),
            "consistency/mae_p50":         float(np.percentile(maes, 50)),
            "consistency/mae_p95":         float(np.percentile(maes, 95)),
            "consistency/mae_max":         float(maes.max()),
            "consistency/max_ae_mean":     float(max_aes.mean()),
            "consistency/max_ae_p50":      float(np.percentile(max_aes, 50)),
            "consistency/max_ae_p95":      float(np.percentile(max_aes, 95)),
            "consistency/max_ae_max":      float(max_aes.max()),
            # Cumulative trajectory stability
            "consistency/cum_l2_final":    float(cum_l2s[-1]),   # total accumulated L2
            "consistency/cum_rms_final":   float(cum_rmss[-1]),  # RMS of L2s at end of path
        }

        print("\n=== Reverse-step ODE consistency summary ===")
        for metric, val in summary.items():
            print(f"  {metric}: {val:.4e}")

        # Per-step trajectory: local + cumulative metrics (step = reverse step index)
        for d in disc_out:
            if wandb.run is not None:
                wandb.log({
                    "consistency/step_l2":       d["l2"],
                    "consistency/step_mae":      d["mae"],
                    "consistency/step_max_ae":   d["max_ae"],
                    "consistency/step_cum_l2":   d["cum_l2_sum"],
                    "consistency/step_cum_rms":  d["cum_rms"],
                    "consistency/t":             d["t"],
                }, step=d["step"])

        # Summary scalars
        if wandb.run is not None:
            wandb.log(summary)

        # ── Cumulative trajectory stability plot ─────────────────────────────
        # Two panels: (top) cumulative L2 sum, (bottom) running RMS.
        # A bounded/flat curve → local errors stay controlled across the path.
        # A steadily rising curve → errors accumulate and may corrupt the sample.
        fig_ct, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        ax_top.plot(steps, cum_l2s, marker="o", markersize=3, color="steelblue")
        ax_top.set_ylabel("Cumulative L2 sum  Σ L2_i")
        ax_top.set_title("ODE reverse-step trajectory stability")
        ax_top.grid(True, linewidth=0.4)

        ax_bot.plot(steps, cum_rmss, marker="o", markersize=3, color="darkorange")
        ax_bot.set_ylabel("Running RMS of L2")
        ax_bot.set_xlabel("Reverse step index  (0 = start at T)")
        ax_bot.grid(True, linewidth=0.4)

        # Secondary x-axis showing diffusion time t (decreasing)
        ax_top2 = ax_top.twiny()
        ax_top2.set_xlim(ax_top.get_xlim())
        tick_idx = np.linspace(0, len(steps) - 1, min(6, len(steps)), dtype=int)
        ax_top2.set_xticks(steps[tick_idx])
        ax_top2.set_xticklabels([f"{ts_diag[j]:.3f}" for j in tick_idx], fontsize=7)
        ax_top2.set_xlabel("Diffusion time t  (T → ε)", fontsize=8)

        plt.tight_layout()
        traj_path = os.path.join(out_dir, "consistency_trajectory.png")
        fig_ct.savefig(traj_path, dpi=120)
        print(f"  [consistency] Trajectory plot saved to {traj_path}")
        if wandb.run is not None:
            wandb.log({"consistency/trajectory_plot": wandb.Image(traj_path)})
        plt.close(fig_ct)

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
    # Cutoff: latest start such that seq_len business days fit before end_date.
    all_bdays    = pd.bdate_range(start=args.start_date, end=args.end_date)
    valid_starts = all_bdays[:-seq_len + 1] if seq_len > 1 else all_bdays
    print(f"Number of valid starts: {len(valid_starts)} (from {valid_starts[0].date()} to {valid_starts[-1].date()})")
    if len(valid_starts) == 0:
        raise ValueError(
            f"No valid start dates: the date range {args.start_date}–{args.end_date} "
            f"is shorter than seq_len={seq_len} business days. Widen the range."
        )
    rng_seed      = None if args.seed == -1 else args.seed
    rng           = np.random.default_rng(rng_seed)
    start_indices = rng.integers(0, len(valid_starts), size=N_SAMPLES)

    samples_np = samples.detach().cpu().numpy()  # (N_SAMPLES, 1, L)

    # ── Save single wide-format CSV ───────────────────────────────────────────
    step_cols = [f"step_{t:03d}" for t in range(seq_len)]
    rows = []
    for i in range(N_SAMPLES):
        series       = samples_np[i, 0, :]  # (L,)
        sample_start = valid_starts[start_indices[i]]
        dates        = pd.bdate_range(start=sample_start, periods=seq_len)
        row = {
            "sample_idx": i,
            "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date":   dates[-1].strftime("%Y-%m-%d"),
        }
        row.update(zip(step_cols, series))
        rows.append(row)

    df_gen   = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "generated_samples.csv")
    df_gen.to_csv(csv_path, index=False)
    print(f"\nSaved {N_SAMPLES} samples → {csv_path}")

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

#python generate_samples.py --checkpoint_folder replication --checkpoint_name NO_NO_REPL_PRIC_WIN_UNCO_ep-3000_sde-vp_noise-linear_lr-1e-04_NOAN_channels-64_layers-4_nheads-4_diffemb-128_seq-64_stride-64_20260412_185321.pt --n_samples 32 --seed 42