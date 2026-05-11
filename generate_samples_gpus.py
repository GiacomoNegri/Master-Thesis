"""
generate_samples_gpus.py — multi-GPU conditional generation of Close time-series
given Open/High/Low context, sampled from training and validation windows.

Drop-in multi-GPU replacement for generate_samples.py.
Windows are distributed across GPUs; each rank runs the EDM Heun sampler
on its own shard. Rank 0 gathers all results, sorts by window_idx, and
writes the same CSV layout as the single-GPU script.

Launch with torchrun:
    torchrun --standalone --nproc_per_node=4 generate_samples_gpus.py \
        --checkpoint_folder ohlc_conditional \
        --checkpoint_name   MY.pt \
        --num_csv           16 \
        --num_samples       10 \
        --seed              42

Outputs (under <out_dir>/<checkpoint_stem>/) — identical schema to
generate_samples.py:
    train_generated_close.csv
    train_gt_ohlc.csv
    val_generated_close.csv
    val_gt_ohlc.csv
"""

import argparse
from datetime import datetime
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb

# Flash Attention CUDA kernels require a minimum number of SMs that MIG GPU
# slices (e.g. 4g.40gb) do not satisfy. Mem-efficient SDP has a hard CUDA
# kernel limit of batch*heads <= 65535, which is violated when num_samples is
# large (num_samples * seq_len * nheads can exceed 500k+). Force math SDP.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.model_core import CSDIModel
from src.utils.WIP_processes import Diffusion_Processes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU conditional Close generation from a trained OHLC CSDI checkpoint."
    )
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)
    p.add_argument("--num_csv",           type=int, required=True,
                   help="Number of windows to use from each split (train and val).")
    p.add_argument("--num_samples",       type=int, default=10,
                   help="Number of Close paths to generate per OHL context window.")
    p.add_argument("--num_reverse_steps", type=int, default=None,
                   help="Override reverse diffusion steps (default: value in checkpoint config).")
    p.add_argument("--chunk_size",        type=int, default=10,
                   help="Number of windows batched into a single edm_sampler call. "
                        "Larger = fewer calls and better GPU utilisation, but more VRAM.")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--out_dir",           type=str,
                   default=os.path.join("data", "generated", "conditional"))
    p.add_argument("--wandb_project",     type=str, default=None)
    p.add_argument("--wandb_entity",      type=str, default=None)
    p.add_argument("--wandb_run_name",    type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset split reconstruction  (identical to generate_samples.py)
# ---------------------------------------------------------------------------

def reconstruct_split(config, repo_root):
    seq_len      = int(config["train"]["seq_len"])
    stride       = int(config["train"]["stride"])
    seed         = int(config["train"]["seed"])
    data_root    = config["train"]["data_root"]
    columns      = tuple(config["data"].get("columns", ("date", "log_adj_close")))
    subset_size  = config["train"].get("train_subset_size", None)
    subset_ratio = config["train"].get("train_subset_ratio", None)
    val_ratio    = config["train"].get("val_split_ratio", None)

    date_col = columns[0]

    abs_root = os.path.normpath(os.path.join(repo_root, data_root))
    files    = sorted(glob.glob(os.path.join(abs_root, "*.csv")))
    assert len(files) > 0, f"No CSVs found in {abs_root}"
    print(f"Data root   : {abs_root}  ({len(files)} files)")

    # Single pass: collect all dates + file lengths, identical to
    # SP500WindowDataset.__init__ so that date_to_idx matches training exactly.
    all_dates_set = set()
    file_lengths  = []
    for fp in files:
        df = pd.read_csv(fp, usecols=[date_col])
        all_dates_set.update(df[date_col].tolist())
        file_lengths.append(len(df))

    all_dates_sorted = sorted(
        all_dates_set,
        key=lambda s: pd.to_datetime(s, format='%d/%m/%Y'),
    )
    date_to_idx = {d: i for i, d in enumerate(all_dates_sorted)}
    print(f"Global date range: {all_dates_sorted[0]} → {all_dates_sorted[-1]}"
          f"  ({len(all_dates_sorted)} unique dates)")

    flat_index = []
    for fi, T in enumerate(file_lengths):
        max_start = T - seq_len
        if max_start < 0:
            continue
        for s in range(0, max_start + 1, stride):
            flat_index.append((fi, s))

    dataset_size = len(flat_index)
    print(f"Total windows: {dataset_size}")

    rng_torch   = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(dataset_size, generator=rng_torch).tolist()

    if val_ratio is not None and 0 < val_ratio < 1:
        val_size    = max(1, int(dataset_size * val_ratio))
        val_indices = all_indices[:val_size]
        train_pool  = all_indices[val_size:]
        print(f"Val windows : {val_size}")
    else:
        val_indices = []
        train_pool  = all_indices
        print("Val windows : 0  (no val_split_ratio in config)")

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
# Per-rank generation + gather to rank 0
# ---------------------------------------------------------------------------

def run_split_ddp(
    split_name, window_indices, flat_index, files,
    feat_cols, close_idx, K, seq_len,
    processes, model, num_samples, num_steps,
    chunk_size,
    device, out_dir, rho,
    date_to_idx, date_col,
    rank, world_size, is_main,
):
    """
    Distributes window_indices across ranks (round-robin), runs the EDM
    sampler on each rank's shard, then gathers rows to rank 0 which sorts
    and saves the CSVs.

    Returns (df_gen, df_ohlc) on rank 0, (None, None) on other ranks.
    """
    n_windows = len(window_indices)
    step_cols = [f"step_{t:03d}" for t in range(seq_len)]

    # ── Assign windows to this rank (round-robin keeps load balanced) ─────────
    local_global_idxs = list(range(rank, n_windows, world_size))   # positions in window_indices
    local_dataset_idxs = [window_indices[i] for i in local_global_idxs]
    n_local = len(local_dataset_idxs)

    # ── Load ground-truth windows + calendar time positions for this rank ─────
    gt_windows_local = []
    gt_tp_local = []   # list of (L,) float32 arrays of global date indices
    for idx in local_dataset_idxs:
        fi, start = flat_index[idx]
        df  = pd.read_csv(files[fi])
        win = df[feat_cols].to_numpy(dtype=np.float32)[start : start + seq_len]
        gt_windows_local.append(win.T)   # (K, L)
        dates_win = df[date_col].iloc[start : start + seq_len].tolist()
        gt_tp_local.append(np.array([date_to_idx[d] for d in dates_win], dtype=np.float32))

    if n_local > 0:
        gt_windows_local = np.stack(gt_windows_local)   # (n_local, K, L)
    else:
        gt_windows_local = np.zeros((0, K, seq_len), dtype=np.float32)

    # ── EDM Heun sampler — chunked over local windows ────────────────────────
    # Each chunk stacks `chunk_size` windows × `num_samples` into one batched
    # sampler call (batch = n_chunk * num_samples).  After the call the result
    # is reshaped back to (n_chunk, num_samples, L) and appended per-window,
    # so gen_close_local keeps the same layout as before.
    gen_close_local = []
    n_chunks = max(1, -(-n_local // chunk_size))   # ceil(n_local / chunk_size)
    with torch.no_grad():
        for chunk_idx, chunk_start in enumerate(range(0, max(n_local, 1), chunk_size)):
            if chunk_start >= n_local:
                break
            chunk_end = min(chunk_start + chunk_size, n_local)
            n_chunk   = chunk_end - chunk_start
            B         = n_chunk * num_samples   # effective batch size for this call

            # Each of the n_chunk windows is repeated num_samples times.
            # obs: (n_chunk, K, L) → expand → (n_chunk, num_samples, K, L)
            #      → reshape  → (B, K, L)
            obs = torch.from_numpy(gt_windows_local[chunk_start:chunk_end]).to(device)
            obs = (obs.unsqueeze(1)
                      .expand(n_chunk, num_samples, K, seq_len)
                      .reshape(B, K, seq_len))

            cond_mask = torch.ones(B, K, seq_len, device=device)
            cond_mask[:, close_idx, :] = 0.0

            # observed_tp: list slice → (n_chunk, L) → same expand/reshape
            chunk_tp    = np.stack(gt_tp_local[chunk_start:chunk_end])   # (n_chunk, L)
            observed_tp = (
                torch.from_numpy(chunk_tp).to(device)
                     .unsqueeze(1)
                     .expand(n_chunk, num_samples, seq_len)
                     .reshape(B, seq_len)
            )

            samples = processes.edm_sampler(
                model         = model,
                shape         = (B, K, seq_len),
                observed_data = obs,
                cond_mask     = cond_mask,
                observed_tp   = observed_tp,
                num_steps     = num_steps,
                rho           = rho,
                device        = device,
            )   # (B, K, L)

            # Split back into per-window arrays of shape (num_samples, L)
            close_preds = (samples[:, close_idx, :]
                           .cpu().numpy()
                           .reshape(n_chunk, num_samples, seq_len))
            for i in range(n_chunk):
                gen_close_local.append(close_preds[i])   # (num_samples, L)

            print(f"  [rank {rank} | {split_name}] chunk {chunk_idx + 1}/{n_chunks} done "
                  f"(windows {chunk_start + 1}–{chunk_end} / {n_local})")

    # ── Build local row lists (carry the original global window position) ─────
    local_gen_rows  = []
    local_ohlc_rows = []

    for li, (gw_idx, dataset_idx) in enumerate(zip(local_global_idxs, local_dataset_idxs)):
        fi, start = flat_index[dataset_idx]
        fname = os.path.basename(files[fi])

        # generated Close rows
        for s_idx in range(num_samples):
            row = {
                "window_idx":   gw_idx,
                "sample_idx":   s_idx,
                "file":         fname,
                "window_start": start,
            }
            row.update(zip(step_cols, gen_close_local[li][s_idx]))
            local_gen_rows.append(row)

        # ground-truth OHLC rows
        for feat_idx, feat_name in enumerate(feat_cols):
            row = {
                "window_idx":   gw_idx,
                "feature":      feat_name,
                "file":         fname,
                "window_start": start,
            }
            row.update(zip(step_cols, gt_windows_local[li, feat_idx, :]))
            local_ohlc_rows.append(row)

    # ── Gather all rows to rank 0 ─────────────────────────────────────────────
    gathered_gen  = [None] * world_size if is_main else None
    gathered_ohlc = [None] * world_size if is_main else None

    dist.gather_object(local_gen_rows,  gathered_gen,  dst=0)
    dist.gather_object(local_ohlc_rows, gathered_ohlc, dst=0)

    if not is_main:
        return None, None

    # ── Rank 0: merge, sort by window_idx, save ───────────────────────────────
    all_gen_rows  = [row for shard in gathered_gen  for row in shard]
    all_ohlc_rows = [row for shard in gathered_ohlc for row in shard]

    all_gen_rows.sort(key=lambda r: (r["window_idx"], r["sample_idx"]))
    all_ohlc_rows.sort(key=lambda r: (r["window_idx"], feat_cols.index(r["feature"])))

    df_gen   = pd.DataFrame(all_gen_rows)
    df_ohlc  = pd.DataFrame(all_ohlc_rows)

    os.makedirs(out_dir, exist_ok=True)

    gen_path  = os.path.join(out_dir, f"{split_name}_generated_close.csv")
    ohlc_path = os.path.join(out_dir, f"{split_name}_gt_ohlc.csv")

    df_gen.to_csv(gen_path,  index=False)
    df_ohlc.to_csv(ohlc_path, index=False)

    print(f"Saved {split_name} generated Close → {gen_path}  "
          f"({len(df_gen)} rows = {n_windows} windows × {num_samples} samples)")
    print(f"Saved {split_name} GT OHLC         → {ohlc_path}  "
          f"({n_windows} windows × {K} features)")

    return df_gen, df_ohlc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── DDP init ──────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = (rank == 0)
    device     = torch.device(f"cuda:{rank}")

    args = parse_args()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_main:
        print(f"DDP: {world_size} GPU(s) active")

    # ── Output dir ────────────────────────────────────────────────────────────
    ckpt_stem = os.path.splitext(args.checkpoint_name)[0]
    # Compute timestamp on rank 0 only and broadcast so all ranks agree on the same path.
    if is_main:
        ts_tensor = torch.tensor(
            [int(datetime.now().strftime("%Y%m%d%H%M%S"))], dtype=torch.long, device=device
        )
    else:
        ts_tensor = torch.zeros(1, dtype=torch.long, device=device)
    dist.broadcast(ts_tensor, src=0)
    run_timestamp = datetime.strptime(str(ts_tensor.item()), "%Y%m%d%H%M%S").strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        args.out_dir, ckpt_stem,
        f"csv{args.num_csv}_samples{args.num_samples}_seed{args.seed}_{run_timestamp}",
    )
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()   # fail fast if any rank is unhealthy before any real work

    # ── Load checkpoint (each rank loads to its own GPU) ──────────────────────
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
    columns    = tuple(config["data"].get("columns", ("date", "log_adj_close")))
    feat_cols  = list(columns[1:])
    K          = len(feat_cols)
    close_idx  = feat_cols.index("close")

    if is_main:
        print(f"Checkpoint : {args.checkpoint_name}")
        print(f"target_dim : {target_dim}  |  seq_len : {seq_len}")
        print(f"columns    : {columns}")
        print(f"feat_cols  : {feat_cols}  (close_idx={close_idx})")
        print(f"SDE type   : {config['process']['sde_type']}")
        print(f"Noise sched.: {config['process']['noise_schedule']} "
              f"with N={config['process']['N']} steps")

    # ── Model (no DDP needed for inference — each rank has its own copy) ───────
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Diffusion processes ───────────────────────────────────────────────────
    processes = Diffusion_Processes(config["process"])
    num_steps = args.num_reverse_steps if args.num_reverse_steps is not None else processes.N
    rho       = float(config.get("edm", {}).get("rho", 7.0))
    if is_main:
        print(f"Diffusion_Processes — SDE: {processes.sde_type}, N: {processes.N}, "
              f"model_steps: {processes.model_steps}")
        print(f"Reverse steps to use: {num_steps}  |  rho: {rho}")

    # ── Reconstruct train/val split (rank 0 only to avoid 4× concurrent I/O) ──
    repo_root = os.path.abspath(os.path.dirname(__file__))
    if is_main:
        _split = list(reconstruct_split(config, repo_root))
    else:
        _split = [None] * 6
    dist.broadcast_object_list(_split, src=0)
    files, flat_index, train_indices, val_indices, date_to_idx, date_col = _split

    n_train_use = min(args.num_csv, len(train_indices))
    n_val_use   = min(args.num_csv, len(val_indices))

    if is_main:
        if n_train_use < args.num_csv:
            print(f"Warning: only {n_train_use} training windows available "
                  f"(requested {args.num_csv}).")
        if len(val_indices) == 0:
            print("Warning: no validation windows in checkpoint config — skipping val generation.")
        elif n_val_use < args.num_csv:
            print(f"Warning: only {n_val_use} validation windows available "
                  f"(requested {args.num_csv}).")
        print(f"\nWindows to generate:")
        print(f"  Train : {n_train_use}  ×  {args.num_samples} samples  "
              f"= {n_train_use * args.num_samples} rows in CSV")
        if n_val_use:
            print(f"  Val   : {n_val_use}  ×  {args.num_samples} samples  "
                  f"= {n_val_use * args.num_samples} rows in CSV")

    selected_train = train_indices[:n_train_use]
    selected_val   = val_indices[:n_val_use]

    # ── W&B init (rank 0 only) ────────────────────────────────────────────────
    use_wandb = (args.wandb_project is not None) and is_main
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
                "world_size":        world_size,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    # ── Generate for TRAIN split ──────────────────────────────────────────────
    if is_main:
        print("\n── Generating for TRAIN split ──")
    dist.barrier()

    df_gen_train, _ = run_split_ddp(
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
        chunk_size     = args.chunk_size,
        device         = device,
        out_dir        = out_dir,
        rho            = rho,
        date_to_idx    = date_to_idx,
        date_col       = date_col,
        rank           = rank,
        world_size     = world_size,
        is_main        = is_main,
    )

    # ── Generate for VAL split ────────────────────────────────────────────────
    df_gen_val = None
    if selected_val:
        if is_main:
            print("\n── Generating for VAL split ──")
        dist.barrier()

        df_gen_val, _ = run_split_ddp(
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
            chunk_size     = args.chunk_size,
            device         = device,
            out_dir        = out_dir,
            rho            = rho,
            date_to_idx    = date_to_idx,
            date_col       = date_col,
            rank           = rank,
            world_size     = world_size,
            is_main        = is_main,
        )

    plt.close("all")

    # ── W&B summary (rank 0 only) ─────────────────────────────────────────────
    if use_wandb and df_gen_train is not None:
        step_cols = [c for c in df_gen_train.columns if c.startswith("step_")]
        gen_np = df_gen_train[step_cols].to_numpy()
        log_dict = {
            "gen/n_train_windows": n_train_use,
            "gen/n_val_windows":   n_val_use,
            "gen/num_samples":     args.num_samples,
            "gen/world_size":      world_size,
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
        if is_main:
            print("W&B run finished.")

    if is_main:
        print(f"\nDone. All outputs in: {out_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# torchrun --standalone --nproc_per_node=4 generate_samples_gpus.py \
#     --checkpoint_folder ohlc_conditional \
#     --checkpoint_name NO_NO_FAKE_FTS_PROC_CLOS_ep-100_step-0_sde-vp_noise-linear_lr-1e-03_N-1000_64_layers-4_nheads-4_diffemb-128_20260407_192135.pt \
#     --out_dir ./data/generated/conditional \
#     --num_csv 1 --num_samples 2 --seed 42