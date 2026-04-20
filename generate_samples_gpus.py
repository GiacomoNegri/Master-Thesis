"""
generate_samples_gpus.py — multi-GPU unconditional generation via DDP.

Launch with torchrun, e.g.:
    torchrun --standalone --nproc_per_node=4 generate_samples_gpus.py \
        --checkpoint_folder replication \
        --checkpoint_name   MY.pt \
        --n_samples         250 \
        --seed              42

Each GPU generates a disjoint slice of the requested samples.
Rank 0 merges the per-rank CSVs into a single generated_samples.csv and
removes the temporaries.  Everything else (CLI, output path, CSV format,
W&B logging) is identical to generate_samples.py.

On a headless HPC node matplotlib is forced to the 'Agg' backend so that
diagnostic plots produced inside Diffusion_Processes.reverse_process are
saved to PNG files instead of being displayed interactively.
"""

import argparse
import os
import sys
import warnings

import matplotlib
from datetime import timedelta
matplotlib.use("Agg")          # must come before any other matplotlib import
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.model_core import CSDIModel
from src.utils.WIP_processes import Diffusion_Processes


# ---------------------------------------------------------------------------
# CLI — identical to generate_samples.py
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU generation of synthetic log_adj_close time-series."
    )

    # Checkpoint
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)

    # Kept for .sh compatibility — ignored at runtime
    p.add_argument("--mask_mode",     type=str, default="unconditional")
    p.add_argument("--cond_data_dir", type=str, default=None)

    # Volume
    p.add_argument("--n_samples", type=int, default=100)

    # Date range
    p.add_argument("--start_date", type=str, default="1986-01-01")
    p.add_argument("--end_date",   type=str, default="2025-12-31")
    p.add_argument("--seed", type=int, default=42)

    # Reverse diffusion
    p.add_argument("--num_reverse_steps", type=int, default=None)

    # Output
    p.add_argument("--out_dir", type=str,
                   default=os.path.join("data", "generated", "replication"))

    # W&B (all optional)
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    # Debug / ODE
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--probability_flow", action="store_true", default=False)
    p.add_argument("--use_heun_ode", action="store_true", default=False,
                   help="Use Heun's method ODE sampler (automatically implies --probability_flow).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── DDP init ──────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    is_main    = (local_rank == 0)

    args = parse_args()

    if args.use_heun_ode:
        _sampler_label = "HEUN_ODE"
    elif args.probability_flow:
        _sampler_label = "ODE"
    else:
        _sampler_label = "SDE"

    # ── Per-rank seed so each GPU draws different initial noise ───────────────
    # Date sampling still uses the original seed (deterministic, rank-independent).
    if args.seed != -1:
        rank_seed = args.seed + local_rank
        torch.manual_seed(rank_seed)
        np.random.seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

    # ── Output dir (rank 0 creates; others wait at barrier) ──────────────────
    ckpt_stem   = os.path.splitext(args.checkpoint_name)[0]
    _parts      = ckpt_stem.split("_")
    _noise_idx  = next((i for i, p in enumerate(_parts) if p.startswith("noise-")), len(_parts) - 3)
    _short_stem = "_".join(_parts[:_noise_idx + 1]) + "_" + "_".join(_parts[-2:])
    out_dir     = os.path.join(args.out_dir, f"{_sampler_label}_{_short_stem}")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{local_rank}")
    if is_main:
        print(f"World size: {world_size}  |  Device (rank 0): {device}")

    # ── Load checkpoint — each rank loads independently ───────────────────────
    checkpoint_path = os.path.join("checkpoints", args.checkpoint_folder, args.checkpoint_name)
    if is_main:
        print(f"Loading checkpoint: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    if is_main:
        print("Checkpoint config:")
        for section, params in config.items():
            print(f"  {section}:")
            for key, value in params.items():
                print(f"    {key}: {value}")

    target_dim = int(config["data"]["target_dim"])
    seq_len    = int(config["train"]["seq_len"])

    # No DDP wrapper needed — inference only, no gradients, no all-reduce.
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if is_main:
        print(f"Checkpoint : {args.checkpoint_name}")
        print(f"target_dim : {target_dim}  |  seq_len : {seq_len}")
        print(f"SDE type   : {config['process']['sde_type']}")
        print(f"Noise sched.: {config['process']['noise_schedule']} with N={config['process']['N']} steps")

    # ── W&B init (rank 0 only) ────────────────────────────────────────────────
    use_wandb = (args.wandb_project is not None) and is_main
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
                "world_size":         world_size,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    # ── Diffusion processes ───────────────────────────────────────────────────
    processes = Diffusion_Processes(config["process"])
    num_reverse_steps = args.num_reverse_steps if args.num_reverse_steps is not None else processes.N
    if is_main:
        print(f"Diffusion_Processes — SDE: {processes.sde_type}, N: {processes.N}, "
              f"model_steps: {processes.model_steps}")
        print(f"Reverse steps to use: {num_reverse_steps}")

    # ── Split N_SAMPLES across ranks ──────────────────────────────────────────
    # Distribute as evenly as possible.  The first (N_TOTAL % world_size) ranks
    # get one extra sample so every sample is generated exactly once.
    N_TOTAL        = args.n_samples
    base           = N_TOTAL // world_size
    extra          = N_TOTAL % world_size
    rank_n_samples = base + (1 if local_rank < extra else 0)
    rank_offset    = local_rank * base + min(local_rank, extra)  # global idx of first sample

    if is_main:
        print(f"Total samples: {N_TOTAL}  |  World size: {world_size}")
    print(f"[rank {local_rank}] generating {rank_n_samples} samples "
          f"(global idx {rank_offset}–{rank_offset + rank_n_samples - 1})")

    # ── Unconditional inputs ──────────────────────────────────────────────────
    observed_data = torch.zeros(rank_n_samples, target_dim, seq_len, device=device)
    cond_mask     = torch.zeros(rank_n_samples, target_dim, seq_len, device=device)

    observed_tp = (
        torch.linspace(0.0, 1.0, seq_len, device=device)
             .unsqueeze(0)
             .expand(rank_n_samples, -1)
    )

    # ── Run reverse diffusion ─────────────────────────────────────────────────
    use_ode   = args.probability_flow
    use_heun  = args.use_heun_ode
    # ODE consistency diagnostic only on rank 0 (same behaviour as original)
    run_debug = args.debug and (use_ode or use_heun) and is_main
    disc_out  = [] if run_debug else None

    samples = processes.reverse_process(
        model            = model,
        shape            = (rank_n_samples, target_dim, seq_len),
        observed_data    = observed_data,
        cond_mask        = cond_mask,
        observed_tp      = observed_tp,
        num_steps        = num_reverse_steps,
        probability_flow = use_ode or use_heun,
        device           = device,
        debug            = run_debug,
        disc_out         = disc_out,
        use_heun         = use_heun,
    )  # → (rank_n_samples, 1, L)

    print(f"[rank {local_rank}] generation done. shape={samples.shape}")
    if is_main:
        print('Samples head:\n', samples[:2, 0, :5])

    # ── ODE consistency diagnostic (rank 0 only — identical to original) ──────
    if run_debug and disc_out:
        steps    = np.array([d["step"]       for d in disc_out])
        ts_diag  = np.array([d["t"]          for d in disc_out])
        l2s      = np.array([d["l2"]         for d in disc_out])
        maes     = np.array([d["mae"]        for d in disc_out])
        max_aes  = np.array([d["max_ae"]     for d in disc_out])
        cum_l2s  = np.array([d["cum_l2_sum"] for d in disc_out])
        cum_rmss = np.array([d["cum_rms"]    for d in disc_out])

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
            "consistency/cum_l2_final":    float(cum_l2s[-1]),
            "consistency/cum_rms_final":   float(cum_rmss[-1]),
        }

        print("\n=== Reverse-step ODE consistency summary ===")
        for metric, val in summary.items():
            print(f"  {metric}: {val:.4e}")

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

        if wandb.run is not None:
            wandb.log(summary)

        fig_ct, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax_top.plot(steps, cum_l2s, marker="o", markersize=3, color="steelblue")
        ax_top.set_ylabel("Cumulative L2 sum  Σ L2_i")
        ax_top.set_title("ODE reverse-step trajectory stability")
        ax_top.grid(True, linewidth=0.4)
        ax_bot.plot(steps, cum_rmss, marker="o", markersize=3, color="darkorange")
        ax_bot.set_ylabel("Running RMS of L2")
        ax_bot.set_xlabel("Reverse step index  (0 = start at T)")
        ax_bot.grid(True, linewidth=0.4)
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

    # ── Save diagnostic figures (rank 0 saves; all ranks close) ──────────────
    wandb_images = {}
    for fig_num in plt.get_fignums():
        if is_main:
            label    = "denoising_trajectory" if fig_num == 1 else "generated_samples"
            fig_path = os.path.join(out_dir, f"diagnostic_{label}.png")
            plt.figure(fig_num).savefig(fig_path, dpi=100, bbox_inches="tight")
            print(f"Saved diagnostic figure → {fig_path}")
            if use_wandb:
                wandb_images[label] = wandb.Image(fig_path)
    plt.close("all")

    if is_main:
        samples_local = samples.detach().cpu()
        print(f"\nRank-0 tensor shape : {samples_local.shape}")
        print(f"Mean  : {samples_local.mean().item():.4f}")
        print(f"Std   : {samples_local.std().item():.4f}")
        print(f"Range : [{samples_local.min().item():.4f}, {samples_local.max().item():.4f}]")

    # ── Build date pool — all ranks, same rng, same global assignment ─────────
    # Re-compute the full N_TOTAL start_indices with the original seed (not the
    # rank-offset seed) so the per-sample dates are identical to a single-GPU run.
    all_bdays    = pd.bdate_range(start=args.start_date, end=args.end_date)
    valid_starts = all_bdays[:-seq_len + 1] if seq_len > 1 else all_bdays
    if is_main:
        print(f"Number of valid starts: {len(valid_starts)} "
              f"(from {valid_starts[0].date()} to {valid_starts[-1].date()})")
    if len(valid_starts) == 0:
        raise ValueError(
            f"No valid start dates: the date range {args.start_date}–{args.end_date} "
            f"is shorter than seq_len={seq_len} business days. Widen the range."
        )

    rng_seed           = None if args.seed == -1 else args.seed
    rng                = np.random.default_rng(rng_seed)
    all_start_indices  = rng.integers(0, len(valid_starts), size=N_TOTAL)
    rank_start_indices = all_start_indices[rank_offset : rank_offset + rank_n_samples]

    # ── Write per-rank temporary CSV ─────────────────────────────────────────
    samples_np = samples.detach().cpu().numpy()  # (rank_n_samples, 1, L)
    step_cols  = [f"step_{t:03d}" for t in range(seq_len)]
    rows = []
    for i in range(rank_n_samples):
        series       = samples_np[i, 0, :]
        sample_start = valid_starts[rank_start_indices[i]]
        dates        = pd.bdate_range(start=sample_start, periods=seq_len)
        row = {
            "sample_idx": rank_offset + i,        # global index — used for merge sort
            "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date":   dates[-1].strftime("%Y-%m-%d"),
        }
        row.update(zip(step_cols, series))
        rows.append(row)

    df_rank  = pd.DataFrame(rows)
    rank_csv = os.path.join(out_dir, f"_tmp_rank{local_rank}.csv")
    df_rank.to_csv(rank_csv, index=False)
    print(f"[rank {local_rank}] saved {rank_n_samples} rows → {rank_csv}")

    # ── Barrier: wait for every rank to finish writing ────────────────────────
    dist.barrier()

    # ── Rank 0: merge, sort, clean up, log ────────────────────────────────────
    if is_main:
        dfs = [
            pd.read_csv(os.path.join(out_dir, f"_tmp_rank{r}.csv"))
            for r in range(world_size)
        ]
        df_gen   = pd.concat(dfs, ignore_index=True).sort_values("sample_idx").reset_index(drop=True)
        csv_path = os.path.join(out_dir, "generated_samples.csv")
        df_gen.to_csv(csv_path, index=False)
        print(f"\nMerged {N_TOTAL} samples → {csv_path}")

        for r in range(world_size):
            os.remove(os.path.join(out_dir, f"_tmp_rank{r}.csv"))

        if use_wandb:
            step_data = df_gen[[c for c in df_gen.columns if c.startswith("step_")]].values
            log_dict = {
                "gen/global_mean": float(step_data.mean()),
                "gen/global_std":  float(step_data.std()),
                "gen/global_min":  float(step_data.min()),
                "gen/global_max":  float(step_data.max()),
                "gen/n_samples":   N_TOTAL,
            }
            log_dict.update(wandb_images)
            wandb.log(log_dict)
            wandb.finish()
            print("W&B run finished.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
