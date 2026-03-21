"""
generate_samples.py — CLI version of notebooks/Reverse_samples_generation.ipynb

Run unconditional generation:
    python generate_samples.py --checkpoint_folder toy --checkpoint_name MY.pt \
        --mask_mode unconditional --n_samples 100

Run conditional (predict_close) generation:
    python generate_samples.py --checkpoint_folder toy --checkpoint_name MY.pt \
        --mask_mode predict_close --n_samples 100 \
        --cond_data_dir data/fake_individual_gbm

W&B logging (optional — omit --wandb_project to skip):
    python generate_samples.py ... \
        --wandb_project csdi-gbm --wandb_entity thesis-giacomo-negri

On a headless HPC node matplotlib is forced to the 'Agg' backend so that the
diagnostic plots produced inside Diffusion_Processes.reverse_process are saved
to PNG files instead of being displayed interactively.
"""

import argparse
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # must come before any other matplotlib import
import matplotlib.pyplot as plt  # noqa: E402 — kept here to apply the backend

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
        description="Generate synthetic time-series with a trained CSDI checkpoint."
    )

    # Checkpoint
    p.add_argument("--checkpoint_folder", type=str, required=True,
                   help="Sub-folder inside checkpoints/ that holds the .pt file.")
    p.add_argument("--checkpoint_name", type=str, required=True,
                   help="Filename of the checkpoint (.pt).")

    # Generation mode
    p.add_argument("--mask_mode", type=str, default="unconditional",
                   choices=["unconditional", "predict_close"],
                   help="'unconditional': generate all features from noise. "
                        "'predict_close': condition on O,H,L,V and predict Close.")

    # Volume
    p.add_argument("--n_samples", type=int, default=100,
                   help="Number of synthetic time-series to generate.")

    # Conditioning data (predict_close only)
    p.add_argument("--cond_data_dir", type=str,
                   default=os.path.join("data", "fake_individual_gbm"),
                   help="Directory of processed CSVs used as conditioning context "
                        "(only used when --mask_mode predict_close).")
    p.add_argument("--cond_stats_path", type=str, default=None,
                   help="Optional path to a pre-computed stats pickle "
                        "{ticker: {mean, std}}. Computed on-the-fly if omitted.")

    # Date range for output CSVs
    p.add_argument("--start_date", type=str, default="1986-01-01",
                   help="Start of the business-day date range used to label rows.")
    p.add_argument("--end_date", type=str, default="2025-12-31",
                   help="End of the business-day date range.")

    # Reverse diffusion
    p.add_argument("--num_reverse_steps", type=int, default=None,
                   help="Override the number of reverse diffusion steps "
                        "(default: value stored in the checkpoint config).")

    # Output
    p.add_argument("--out_dir", type=str,
                   default=os.path.join("data", "generated", "toy"),
                   help="Directory where generated CSVs and stats pickle are saved.")

    # W&B (all optional — omit --wandb_project to disable logging entirely)
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project name. W&B logging is disabled when omitted.")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity (team or username).")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Human-readable run name. Auto-generated from checkpoint if omitted.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint_path = os.path.join("checkpoints", args.checkpoint_folder, args.checkpoint_name)
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    target_dim = int(config["data"]["target_dim"])   # K features
    seq_len    = int(config["train"]["seq_len"])      # L time steps

    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Checkpoint : {args.checkpoint_name}")
    print(f"target_dim : {target_dim}  |  seq_len : {seq_len}")
    print(f"SDE type   : {config['process']['sde_type']}")

    # ── W&B init (optional) ───────────────────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        run_name = args.wandb_run_name or (
            f"gen_{args.mask_mode}_{args.checkpoint_name.replace('.pt', '')}"
        )
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_name,
            config  = {
                # generation settings
                "checkpoint_folder":  args.checkpoint_folder,
                "checkpoint_name":    args.checkpoint_name,
                "mask_mode":          args.mask_mode,
                "n_samples":          args.n_samples,
                "num_reverse_steps":  args.num_reverse_steps,
                "start_date":         args.start_date,
                "end_date":           args.end_date,
                # model / process settings from the checkpoint
                "sde_type":           config["process"]["sde_type"],
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

    # ── Feature bookkeeping ───────────────────────────────────────────────────
    FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
    CLOSE_IDX    = FEATURE_COLS.index("Close")                              # 3
    COND_INDICES = [i for i, f in enumerate(FEATURE_COLS) if f != "Close"]  # [0,1,2,4]

    norm_stats = {}  # filled only in predict_close mode

    # ── Build observed_data and cond_mask ─────────────────────────────────────
    N_SAMPLES  = args.n_samples
    mask_mode  = args.mask_mode

    if mask_mode == "unconditional":
        observed_data = torch.zeros(N_SAMPLES, target_dim, seq_len, device=device)
        cond_mask     = torch.zeros(N_SAMPLES, target_dim, seq_len, device=device)
        cond_tickers  = [f"FAKE_{i+1:04d}" for i in range(N_SAMPLES)]

    elif mask_mode == "predict_close":
        csv_files = sorted([f for f in os.listdir(args.cond_data_dir) if f.endswith(".csv")])
        if len(csv_files) < N_SAMPLES:
            raise ValueError(
                f"Need {N_SAMPLES} CSV files but found only {len(csv_files)} "
                f"in {args.cond_data_dir}"
            )
        selected_files = csv_files[:N_SAMPLES]

        # Load optional pre-computed stats
        precomp_stats = {}
        if args.cond_stats_path is not None:
            with open(args.cond_stats_path, "rb") as fh:
                precomp_stats = pickle.load(fh)

        obs_np       = np.zeros((N_SAMPLES, target_dim, seq_len), dtype=np.float32)
        cond_tickers = []

        for idx, fname in enumerate(selected_files):
            ticker = fname.split("_processed")[0]
            cond_tickers.append(ticker)

            df   = pd.read_csv(os.path.join(args.cond_data_dir, fname))
            vals = df[FEATURE_COLS].values[:seq_len].T.astype(np.float32)  # (K, L)

            if ticker in precomp_stats:
                mu  = precomp_stats[ticker]["mean"].reshape(-1, 1)
                std = precomp_stats[ticker]["std"].reshape(-1, 1)
            else:
                mu  = vals.mean(axis=1, keepdims=True)
                std = vals.std(axis=1, keepdims=True).clip(min=1e-8)

            norm_stats[idx] = {"mean": mu.squeeze(), "std": std.squeeze(), "ticker": ticker}
            obs_np[idx] = (vals - mu) / std

        observed_data = torch.from_numpy(obs_np).to(device)

        cond_mask = torch.zeros(N_SAMPLES, target_dim, seq_len, device=device)
        cond_mask[:, COND_INDICES, :] = 1.0

    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode!r}")

    print(f"mask_mode     : {mask_mode}")
    print(f"observed frac : {cond_mask.mean().item():.2f}")

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
    )  # → (N_SAMPLES, K, L)

    samples_global_mean = samples.mean().item()
    samples_global_std  = samples.std().item()
    samples_global_min  = samples.min().item()
    samples_global_max  = samples.max().item()

    print(f"\nGenerated tensor shape : {samples.shape}")
    print(f"Mean  : {samples_global_mean:.4f}")
    print(f"Std   : {samples_global_std:.4f}")
    print(f"Range : [{samples_global_min:.4f}, {samples_global_max:.4f}]")

    # ── Save diagnostic figures and optionally upload to W&B ─────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    wandb_images = {}
    for fig_num in plt.get_fignums():
        label    = "denoising_trajectory" if fig_num == 1 else "generated_samples"
        fig_path = os.path.join(args.out_dir, f"diagnostic_{label}.png")
        plt.figure(fig_num).savefig(fig_path, dpi=100, bbox_inches="tight")
        print(f"Saved diagnostic figure → {fig_path}")
        if use_wandb:
            wandb_images[label] = wandb.Image(fig_path)
    plt.close("all")

    # ── Save results ──────────────────────────────────────────────────────────
    dates = pd.bdate_range(start=args.start_date, end=args.end_date)
    if len(dates) < seq_len:
        raise ValueError(
            f"Date range {args.start_date} → {args.end_date} yields only "
            f"{len(dates)} business days but seq_len = {seq_len}. "
            "Please extend --end_date."
        )
    dates = dates[:seq_len]

    samples_np  = samples.detach().cpu().numpy()         # (N_SAMPLES, K, L)
    obs_np_cpu  = observed_data.detach().cpu().numpy() if mask_mode == "predict_close" else None

    stats_dict = {}

    for i in range(N_SAMPLES):
        ticker = cond_tickers[i]
        gen    = samples_np[i]   # (K, L) in normalised space

        if mask_mode == "unconditional":
            series = gen
            mu  = series.mean(axis=1).astype(np.float32)
            std = series.std(axis=1).clip(min=1e-8).astype(np.float32)

        else:  # predict_close
            mu_arr  = norm_stats[i]["mean"]   # (K,)
            std_arr = norm_stats[i]["std"]    # (K,)

            series_denorm = gen * std_arr[:, None] + mu_arr[:, None]
            orig = obs_np_cpu[i] * std_arr[:, None] + mu_arr[:, None]
            series_denorm[COND_INDICES, :] = orig[COND_INDICES, :]

            series = series_denorm
            mu  = series.mean(axis=1).astype(np.float32)
            std = series.std(axis=1).clip(min=1e-8).astype(np.float32)

        df = pd.DataFrame(series.T, columns=FEATURE_COLS)
        df.insert(0, "Date", dates.strftime("%d/%m/%Y"))

        suffix   = "predicted_close" if mask_mode == "predict_close" else "generated"
        filename = f"{ticker}_{args.start_date}_{args.end_date}_{suffix}.csv"
        df.to_csv(os.path.join(args.out_dir, filename), index=False)

        stats_dict[ticker] = {"mean": mu, "std": std}

    stats_path = os.path.join(args.out_dir, "fake_stats_generated.pkl")
    with open(stats_path, "wb") as fh:
        pickle.dump(stats_dict, fh)

    print(f"\nmask_mode : {mask_mode}")
    print(f"Saved {N_SAMPLES} CSV files  →  {args.out_dir}")
    print(f"Saved stats pickle          →  {stats_path}")

    # ── W&B — log summary metrics, per-feature stats, and plots ──────────────
    if use_wandb:
        FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
        samples_np_final = samples.detach().cpu().numpy()  # (N, K, L)

        # Global stats
        log_dict = {
            "gen/global_mean": samples_global_mean,
            "gen/global_std":  samples_global_std,
            "gen/global_min":  samples_global_min,
            "gen/global_max":  samples_global_max,
            "gen/n_samples":   N_SAMPLES,
        }

        # Per-feature mean and std (averaged across samples and time steps)
        for k_idx, feat in enumerate(FEATURE_COLS):
            feat_vals = samples_np_final[:, k_idx, :]   # (N, L)
            log_dict[f"gen/feat_{feat}_mean"] = float(feat_vals.mean())
            log_dict[f"gen/feat_{feat}_std"]  = float(feat_vals.std())

        # Diagnostic plots
        log_dict.update(wandb_images)

        # W&B artifact — upload the stats pickle
        artifact = wandb.Artifact(
            name=f"generated_samples_{wandb.run.id}",
            type="generation_output",
            description=f"{mask_mode} generation from {args.checkpoint_name}",
        )
        artifact.add_file(stats_path)
        wandb.log_artifact(artifact)

        wandb.log(log_dict)
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
