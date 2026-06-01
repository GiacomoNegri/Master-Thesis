"""
generate_samples.py — conditional generation of Close time-series
given Open/High/Low context, sampled from training and validation windows.

Reconstructs the exact train/val split from the checkpoint config (same
deterministic logic used during training), then runs the reverse diffusion
conditioned on OHL for each selected window.

Run:
    python generate_samples.py \
        --checkpoint_folder ohlc_conditional \
        --checkpoint_name   MY.pt \
        --num_csv           16 \
        --num_samples       10 \
        --seed              42

python generate_samples.py --checkpoint_folder edm --checkpoint_name <checkpoint_name>.pt --num_csv 10 --num_samples 10 --seed 42

Outputs (under <out_dir>/<checkpoint_stem>/):
    train_generated_close.csv  — generated Close paths for training windows
    train_gt_ohlc.csv          — ground-truth OHLC context for training windows
    val_generated_close.csv    — generated Close paths for validation windows
    val_gt_ohlc.csv            — ground-truth OHLC context for validation windows

CSV schemas
-----------
*_generated_close.csv:
    window_idx | sample_idx | file | window_start | step_000 … step_{L-1}

*_gt_ohlc.csv:
    window_idx | feature | file | window_start | step_000 … step_{L-1}

Reload example (in a notebook or analysis script):
    import pandas as pd
    df_gen  = pd.read_csv(".../<ckpt>/train_generated_close.csv")
    df_ohlc = pd.read_csv(".../<ckpt>/train_gt_ohlc.csv")
    step_cols = [c for c in df_gen.columns if c.startswith("step_")]
    # all generated Close paths for window 0:
    window0 = df_gen[df_gen.window_idx == 0][step_cols].to_numpy()

W&B logging (optional — omit --wandb_project to skip):
    python generate_samples.py ... \
        --wandb_project generation-ohlc --wandb_entity thesis-giacomo-negri

On a headless HPC node matplotlib is forced to the 'Agg' backend so that
any diagnostic plots produced inside Diffusion_Processes.reverse_process are
saved to PNG files instead of being displayed interactively.
"""

import argparse
import glob
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
        description="Generate conditional Close samples from a trained OHLC CSDI checkpoint."
    )

    # Checkpoint
    p.add_argument("--checkpoint_folder", type=str, required=True,
                   help="Sub-folder inside checkpoints/ that holds the .pt file.")
    p.add_argument("--checkpoint_name", type=str, required=True,
                   help="Filename of the checkpoint (.pt).")

    # Volume
    p.add_argument("--num_csv", type=int, required=True,
                   help="Number of windows to use from each split (train and val).")
    p.add_argument("--num_samples", type=int, default=10,
                   help="Number of Close paths to generate per OHL context window.")

    # Reverse diffusion
    p.add_argument("--num_reverse_steps", type=int, default=None,
                   help="Override reverse diffusion steps (default: value in checkpoint config).")

    # Stochastic sampler parameters (S_churn=0 gives deterministic Heun)
    p.add_argument("--S_churn", type=float, default=0.0,
                   help="Stochastic noise budget per step. 0 = deterministic Heun sampler.")
    p.add_argument("--S_tmin",  type=float, default=0.05,
                   help="Lower sigma bound for stochastic noise injection.")
    p.add_argument("--S_tmax",  type=float, default=float('inf'),
                   help="Upper sigma bound for stochastic noise injection.")
    p.add_argument("--S_noise", type=float, default=1.0,
                   help="Noise amplification factor for stochastic injection.")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Output base dir — checkpoint subfolder is appended automatically
    p.add_argument("--out_dir", type=str,
                   default=os.path.join("data", "generated", "conditional"),
                   help="Base directory; a sub-folder named after the checkpoint is created here.")

    # Date format — must match the format used in the training CSVs so that
    # date_to_idx is built with the same sort order as SP500WindowDataset.
    p.add_argument("--date_format", type=str, default="%d/%m/%Y",
                   help="strptime format of the date column in the training CSVs "
                        "(default: %%d/%%m/%%Y). Must match the format used at training time.")

    # W&B (all optional)
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset split reconstruction (mirrors SP500WindowDataset + training script)
# ---------------------------------------------------------------------------

def reconstruct_split(config, repo_root, date_format="%d/%m/%Y"):
    """
    Rebuild the exact train/val window indices from the checkpoint config.

    Replicates the index-building and permutation logic from
    csdi_train_modified.py and SP500WindowDataset.__init__ so that the
    returned indices are identical to those used during training.

    Also builds date_to_idx — the same global calendar-date → integer mapping
    that SP500WindowDataset constructs, required for correct observed_tp values.

    Returns
    -------
    files         : sorted list of absolute CSV paths
    flat_index    : list of (file_idx, start_row) for every valid window
    train_indices : list of dataset indices used for training
    val_indices   : list of dataset indices used for validation (may be empty)
    date_to_idx   : dict mapping date string → global integer index (matches training)
    date_col      : name of the date column in the CSVs
    """
    # ── Config fields ─────────────────────────────────────────────────────────
    seq_len      = int(config["train"]["seq_len"])
    stride       = int(config["train"]["stride"])
    seed         = int(config["train"]["seed"])
    data_root    = config["train"]["data_root"]
    columns      = tuple(config["data"].get("columns", ("date", "log_adj_close")))
    subset_size  = config["train"].get("train_subset_size", None)
    subset_ratio = config["train"].get("train_subset_ratio", None)
    val_ratio    = config["train"].get("val_split_ratio", None)

    date_col = columns[0]

    # ── Resolve data_root relative to repo root ───────────────────────────────
    abs_root = os.path.normpath(os.path.join(repo_root, data_root))
    files    = sorted(glob.glob(os.path.join(abs_root, "*.csv")))
    assert len(files) > 0, f"No CSVs found in {abs_root}"
    print(f"Data root   : {abs_root}  ({len(files)} files)")

    # ── Single pass: collect all dates + file lengths (mirrors SP500WindowDataset.__init__) ──
    all_dates_set = set()
    file_lengths  = []
    for fp in files:
        df = pd.read_csv(fp, usecols=[date_col])
        all_dates_set.update(df[date_col].tolist())
        file_lengths.append(len(df))

    # Identical sort to SP500WindowDataset.date_to_idx
    all_dates_sorted = sorted(
        all_dates_set,
        key=lambda s: pd.to_datetime(s, format='%d/%m/%Y'),
    )
    date_to_idx = {d: i for i, d in enumerate(all_dates_sorted)}
    print(f"Global date range: {all_dates_sorted[0]} → {all_dates_sorted[-1]}"
          f"  ({len(all_dates_sorted)} unique dates)")

    # ── Build flat index (reuses file_lengths already collected above) ────────
    flat_index = []
    for fi, T in enumerate(file_lengths):
        max_start = T - seq_len
        if max_start < 0:
            continue
        for s in range(0, max_start + 1, stride):
            flat_index.append((fi, s))

    dataset_size = len(flat_index)
    print(f"Total windows: {dataset_size}")

    # ── Permute with the same seed used during training ───────────────────────
    rng_torch   = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(dataset_size, generator=rng_torch).tolist()

    # ── Val split ─────────────────────────────────────────────────────────────
    if val_ratio is not None and 0 < val_ratio < 1:
        val_size    = max(1, int(dataset_size * val_ratio))
        val_indices = all_indices[:val_size]
        train_pool  = all_indices[val_size:]
        print(f"Val windows : {val_size}")
    else:
        val_indices = []
        train_pool  = all_indices
        print("Val windows : 0  (no val_split_ratio in config)")

    # ── Train subset ──────────────────────────────────────────────────────────
    if subset_size is not None:
        n_train = min(int(subset_size), len(train_pool))
    elif subset_ratio is not None:
        n_train = max(1, int(dataset_size * subset_ratio))
        n_train = min(n_train, len(train_pool))
    else:
        n_train = len(train_pool)

    train_indices = train_pool[:n_train]
    print(f"Train windows: {n_train}")

    return files, flat_index, train_indices, val_indices, date_to_idx, date_col


# ---------------------------------------------------------------------------
# Conditional generation + save for one split
# ---------------------------------------------------------------------------

def run_split(
    split_name, window_indices, flat_index, files,
    feat_cols, close_idx, K, seq_len,
    processes, model, num_samples, num_steps,
    device, out_dir, rho,
    date_to_idx, date_col,
    S_churn=0.0, S_tmin=0.05, S_tmax=float('inf'), S_noise=1.0,
):
    """
    For each window in window_indices:
      - load the ground-truth OHLC window from CSV
      - run reverse process conditioned on OHL (Close channel is masked)
      - collect num_samples generated Close paths

    Saves two CSVs under out_dir:
      {split_name}_generated_close.csv
      {split_name}_gt_ohlc.csv

    Returns the two DataFrames for optional W&B logging.
    """
    n_windows = len(window_indices)
    step_cols = [f"step_{t:03d}" for t in range(seq_len)]

    # ── Load ground-truth windows + calendar time positions ──────────────────
    gt_windows = []
    gt_tp      = []   # list of (L,) float32 arrays of global date indices
    for idx in window_indices:
        fi, start = flat_index[idx]
        df  = pd.read_csv(files[fi])
        win = df[feat_cols].to_numpy(dtype=np.float32)[start : start + seq_len]  # (L, K)
        gt_windows.append(win.T)   # (K, L)
        dates_win = df[date_col].iloc[start : start + seq_len].tolist()
        gt_tp.append(np.array([date_to_idx[d] for d in dates_win], dtype=np.float32))
    gt_windows = np.stack(gt_windows)   # (n_windows, K, L)

    # ── Reverse process (conditioned on OHL, masking Close) ───────────────────
    gen_close_list = []   # list of (num_samples, L) arrays, one per window
    with torch.no_grad():
        for w_idx in range(n_windows):
            # Repeat the same OHL context num_samples times
            obs = torch.from_numpy(gt_windows[w_idx]).unsqueeze(0)    # (1, K, L)
            obs = obs.expand(num_samples, -1, -1).to(device)          # (B, K, L)

            # cond_mask: 1 = observed (OHL), 0 = to generate (Close)
            cond_mask = torch.ones(num_samples, K, seq_len, device=device)
            cond_mask[:, close_idx, :] = 0.0

            observed_tp = (
                torch.from_numpy(gt_tp[w_idx])
                     .to(device)
                     .unsqueeze(0)
                     .expand(num_samples, -1)
            )  # (B, L) — global calendar indices, identical to SP500WindowDataset

            # EDM EDITING: replaced legacy reverse_process (score-based Euler-Maruyama)
            # with the EDM deterministic Heun sampler.
            samples = processes.edm_sampler(
                model         = model,
                shape         = (num_samples, K, seq_len),
                observed_data = obs,
                cond_mask     = cond_mask,
                observed_tp   = observed_tp,
                num_steps     = num_steps,
                rho           = rho,
                device        = device,
                S_churn       = S_churn,
                S_tmin        = S_tmin,
                S_tmax        = S_tmax,
                S_noise       = S_noise,
            )   # (B, K, L)

            gen_close_list.append(samples[:, close_idx, :].cpu().numpy())   # (B, L)

            log_every = max(1, n_windows // 5)
            if (w_idx + 1) % log_every == 0 or w_idx == n_windows - 1:
                print(f"  [{split_name}] {w_idx + 1}/{n_windows} windows done")

    # ── Save generated Close ──────────────────────────────────────────────────
    gen_rows = []
    for w_idx, dataset_idx in enumerate(window_indices):
        fi, start = flat_index[dataset_idx]
        fname = os.path.basename(files[fi])
        for s_idx in range(num_samples):
            row = {
                "window_idx":   w_idx,
                "sample_idx":   s_idx,
                "file":         fname,
                "window_start": start,
            }
            row.update(zip(step_cols, gen_close_list[w_idx][s_idx]))
            gen_rows.append(row)

    df_gen   = pd.DataFrame(gen_rows)
    gen_path = os.path.join(out_dir, f"{split_name}_generated_close.csv")
    df_gen.to_csv(gen_path, index=False)
    print(f"Saved {split_name} generated Close → {gen_path}  "
          f"({len(df_gen)} rows = {n_windows} windows × {num_samples} samples)")

    # ── Save ground-truth OHLC context ────────────────────────────────────────
    ohlc_rows = []
    for w_idx, dataset_idx in enumerate(window_indices):
        fi, start = flat_index[dataset_idx]
        fname = os.path.basename(files[fi])
        for feat_idx, feat_name in enumerate(feat_cols):
            row = {
                "window_idx":   w_idx,
                "feature":      feat_name,
                "file":         fname,
                "window_start": start,
            }
            row.update(zip(step_cols, gt_windows[w_idx, feat_idx, :]))
            ohlc_rows.append(row)

    df_ohlc   = pd.DataFrame(ohlc_rows)
    ohlc_path = os.path.join(out_dir, f"{split_name}_gt_ohlc.csv")
    df_ohlc.to_csv(ohlc_path, index=False)
    print(f"Saved {split_name} GT OHLC     → {ohlc_path}  "
          f"({n_windows} windows × {K} features)")

    return df_gen, df_ohlc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    config = ckpt["config"]

    print("Checkpoint config:")
    for section, params in config.items():
        print(f"  {section}:")
        for key, value in params.items():
            print(f"    {key}: {value}")

    target_dim = int(config["data"]["target_dim"])
    seq_len    = int(config["train"]["seq_len"])
    columns    = tuple(config["data"].get("columns", ("date", "log_adj_close")))
    feat_cols  = list(columns[1:])          # drop the date column
    K          = len(feat_cols)             # = target_dim
    close_idx  = feat_cols.index("close")  # typically 3 for OHLC

    print(f"Checkpoint : {args.checkpoint_name}")
    print(f"target_dim : {target_dim}  |  seq_len : {seq_len}")
    print(f"columns    : {columns}")
    print(f"feat_cols  : {feat_cols}  (close_idx={close_idx})")
    print(f"SDE type   : {config['process']['sde_type']}")
    print(f"Noise sched.: {config['process']['noise_schedule']} "
          f"with N={config['process']['N']} steps")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Diffusion processes ───────────────────────────────────────────────────
    processes = Diffusion_Processes(config["process"])
    num_steps = args.num_reverse_steps if args.num_reverse_steps is not None else processes.N
    rho       = float(config.get("edm", {}).get("rho", 7.0))
    print(f"Diffusion_Processes — SDE: {processes.sde_type}, N: {processes.N}, "
          f"model_steps: {processes.model_steps}")
    print(f"Reverse steps to use: {num_steps}  |  rho: {rho}")

    # ── Reconstruct train/val split from checkpoint config ────────────────────
    repo_root = os.path.abspath(os.path.dirname(__file__))
    files, flat_index, train_indices, val_indices, date_to_idx, date_col = reconstruct_split(
        config, repo_root, date_format=args.date_format,
    )

    # ── Select num_csv windows from each split ────────────────────────────────
    n_train_use = min(args.num_csv, len(train_indices))
    n_val_use   = min(args.num_csv, len(val_indices))

    if n_train_use < args.num_csv:
        print(f"Warning: only {n_train_use} training windows available "
              f"(requested {args.num_csv}).")
    if len(val_indices) == 0:
        print("Warning: no validation windows in checkpoint config — skipping val generation.")
    elif n_val_use < args.num_csv:
        print(f"Warning: only {n_val_use} validation windows available "
              f"(requested {args.num_csv}).")

    selected_train = train_indices[:n_train_use]
    selected_val   = val_indices[:n_val_use]

    print(f"\nWindows to generate:")
    print(f"  Train : {n_train_use}  ×  {args.num_samples} samples  "
          f"= {n_train_use * args.num_samples} rows in CSV")
    if selected_val:
        print(f"  Val   : {n_val_use}  ×  {args.num_samples} samples  "
              f"= {n_val_use * args.num_samples} rows in CSV")

    # ── W&B init (optional) ───────────────────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        run_name = args.wandb_run_name or f"gen_conditional_{ckpt_stem}"
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_name,
            config  = {
                "checkpoint_folder": args.checkpoint_folder,
                "checkpoint_name":   args.checkpoint_name,
                "num_csv":           args.num_csv,
                "num_samples":       args.num_samples,
                "num_reverse_steps": num_steps,
                "sde_type":          config["process"]["sde_type"],
                "noise_schedule":    config["process"]["noise_schedule"],
                "sde_N":             config["process"]["N"],
                "target_dim":        target_dim,
                "seq_len":           seq_len,
                "close_idx":         close_idx,
                "S_churn":           args.S_churn,
                "S_tmin":            args.S_tmin,
                "S_tmax":            args.S_tmax,
                "S_noise":           args.S_noise,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    # ── Generate for TRAIN split ──────────────────────────────────────────────
    print("\n── Generating for TRAIN split ──")
    df_gen_train, _ = run_split(
        split_name     = "train",
        window_indices = selected_train,
        flat_index     = flat_index,
        files          = files,
        feat_cols      = feat_cols,
        close_idx      = close_idx,
        K              = K,
        seq_len        = seq_len,
        processes      = processes,
        model          = model,
        num_samples    = args.num_samples,
        num_steps      = num_steps,
        device         = device,
        out_dir        = out_dir,
        rho            = rho,
        date_to_idx    = date_to_idx,
        date_col       = date_col,
        S_churn        = args.S_churn,
        S_tmin         = args.S_tmin,
        S_tmax         = args.S_tmax,
        S_noise        = args.S_noise,
    )

    # ── Generate for VAL split ────────────────────────────────────────────────
    df_gen_val = None
    if selected_val:
        print("\n── Generating for VAL split ──")
        df_gen_val, _ = run_split(
            split_name     = "val",
            window_indices = selected_val,
            flat_index     = flat_index,
            files          = files,
            feat_cols      = feat_cols,
            close_idx      = close_idx,
            K              = K,
            seq_len        = seq_len,
            processes      = processes,
            model          = model,
            num_samples    = args.num_samples,
            num_steps      = num_steps,
            device         = device,
            out_dir        = out_dir,
            rho            = rho,
            date_to_idx    = date_to_idx,
            date_col       = date_col,
            S_churn        = args.S_churn,
            S_tmin         = args.S_tmin,
            S_tmax         = args.S_tmax,
            S_noise        = args.S_noise,
        )

    plt.close("all")

    # ── W&B summary ───────────────────────────────────────────────────────────
    if use_wandb:
        step_cols = [c for c in df_gen_train.columns if c.startswith("step_")]

        gen_np = df_gen_train[step_cols].to_numpy()
        log_dict = {
            "gen/n_train_windows": n_train_use,
            "gen/n_val_windows":   n_val_use,
            "gen/num_samples":     args.num_samples,
            "train/gen_mean":      gen_np.mean(),
            "train/gen_std":       gen_np.std(),
            "train/gen_min":       gen_np.min(),
            "train/gen_max":       gen_np.max(),
        }
        if df_gen_val is not None:
            gen_val_np = df_gen_val[step_cols].to_numpy()
            log_dict.update({
                "val/gen_mean": gen_val_np.mean(),
                "val/gen_std":  gen_val_np.std(),
                "val/gen_min":  gen_val_np.min(),
                "val/gen_max":  gen_val_np.max(),
            })

        wandb.log(log_dict)
        wandb.finish()
        print("W&B run finished.")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()

#python generate_samples.py --checkpoint_folder ohlc_conditional --checkpoint_name NO_NO_FAKE_FTS_PROC_CLOS_ep-100_step-0_sde-vp_noise-linear_lr-1e-03_N-1000_64_layers-4_nheads-4_diffemb-128_20260407_192135.pt --out_dir ./data/generated/conditional --num_csv 1 --num_samples 2 --seed 42