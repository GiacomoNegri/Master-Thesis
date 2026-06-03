"""
generate_samples_unconditional_gpus.py — multi-GPU generation of Close time-series
supporting both unconditional and conditional EDM checkpoints.

Extends generate_samples_gpus.py with the following additions (tagged CHANGE[N]):
  CHANGE[1]  --mask_mode  CLI arg (unconditional | predict_close)
  CHANGE[2]  is_unconditional parameter threaded into run_split_ddp
  CHANGE[3]  cond_mask explicitly all-zeros for unconditional mode
  CHANGE[4]  samples extraction branch on is_unconditional
  CHANGE[5]  output CSV named *_gt_features.csv (not *_gt_ohlc.csv) for unconditional
  CHANGE[6]  --n_samples canonical name; --num_samples kept as deprecated alias
  CHANGE[7]  consistency check: --mask_mode vs config["model"]["is_unconditional"]
  CHANGE[8]  --data_root CLI override applied before reconstruct_split
  CHANGE[9]  close_idx set to 0 without .index("close") for unconditional
  CHANGE[10] --edm_rho CLI arg; takes precedence over config["edm"]["rho"]
  CHANGE[11] --cond_data_dir / --probability_flow stubs for interface completeness
  CHANGE[12] --out_dir default changed to data/generated/unconditional

Everything not tagged CHANGE is identical to generate_samples_gpus.py.

Launch with torchrun:
    torchrun --standalone --nproc_per_node=4 generate_samples_unconditional_gpus.py \\
        --checkpoint_folder edm_unconditional         \\
        --checkpoint_name   MY_UNCONDITIONAL.pt       \\
        --mask_mode         unconditional             \\
        --num_csv           256                       \\
        --n_samples         10                        \\
        --seed              42

Outputs (under <out_dir>/<checkpoint_stem>/<run_tag>/):
    train_generated_close.csv   — generated Close paths for training windows
    train_gt_features.csv       — ground-truth features for training windows
    val_generated_close.csv     — generated Close paths for validation windows
    val_gt_features.csv         — ground-truth features for validation windows
"""

import argparse
from datetime import datetime, timedelta
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
# CHANGE[1]: Added --mask_mode, --n_samples, --data_root, --edm_rho,
#            --cond_data_dir, --probability_flow.
#            --num_samples retained as a deprecated alias for --n_samples.
# CHANGE[12]: --out_dir default changed to data/generated/unconditional so that
#             unconditional runs do not land in a folder named 'conditional'.
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU Close generation (unconditional or conditional) "
                    "from a trained EDM CSDI checkpoint."
    )
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--checkpoint_name",   type=str, required=True)

    # CHANGE[1a]: --mask_mode is the central switch between generation modes.
    # It is cross-checked against config["model"]["is_unconditional"] from the
    # checkpoint (CHANGE[7]) so that a mismatched call fails loudly before any
    # GPU memory is allocated rather than producing silently wrong outputs.
    p.add_argument(
        "--mask_mode", type=str, default="predict_close",
        choices=["predict_close", "unconditional"],
        help=(
            "Generation mode. "
            "'unconditional': cond_mask=0 everywhere, no observed context. "
            "'predict_close': condition on OHL, generate Close only. "
            "Must match the training mode recorded in the checkpoint."
        ),
    )

    p.add_argument("--num_csv", type=int, required=True,
                   help="Number of windows to use from each split (train and val).")

    # CHANGE[1b]: --n_samples is the canonical name matching the new job interface.
    # --num_samples is kept as a deprecated alias; --n_samples takes precedence
    # when both are supplied. Mutual exclusion is resolved in main() (CHANGE[6]).
    p.add_argument("--n_samples",   type=int, default=None,
                   help="Number of paths to generate per window.")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Deprecated alias for --n_samples.")

    p.add_argument("--num_reverse_steps", type=int, default=None,
                   help="Override reverse diffusion steps (default: value in checkpoint config).")
    p.add_argument("--chunk_size",        type=int, default=10,
                   help="Windows batched per edm_sampler call. Larger = better GPU "
                        "utilisation but more VRAM.")
    p.add_argument("--sigma_max",         type=float, default=None,
                   help="Override sigma_max from checkpoint config.")

    # CHANGE[1c]: --edm_rho separates the sampler's sigma-schedule curvature
    # from the training config. rho can be tuned at generation time without
    # retraining. Fallback chain: CLI → config["edm"]["rho"] → 7.0 (CHANGE[10]).
    p.add_argument("--edm_rho", type=float, default=None,
                   help="Override EDM rho (sigma-schedule curvature, default 7.0).")

    # CHANGE[1d]: --data_root overrides the path stored in the checkpoint config.
    # Needed when the generation machine mounts the training data at a different
    # path from training time (CHANGE[8]).
    p.add_argument("--data_root", type=str, default=None,
                   help="Override config['train']['data_root'] from the checkpoint. "
                        "Required when the data is not at its original training path.")

    # CHANGE[1e/11]: stubs for interface completeness with the unified generation job.
    # Neither argument is used in unconditional mode.
    p.add_argument("--cond_data_dir", type=str, default=None,
                   help="(predict_close only) Directory of conditioning OHL CSVs. "
                        "Unused in unconditional mode.")
    p.add_argument("--probability_flow", action="store_true",
                   help="(VP SDE only) Use the probability-flow ODE. "
                        "Unused for EDM checkpoints.")

    p.add_argument("--seed",    type=int, default=42)
    # CHANGE[12]: default changed from 'conditional' to 'unconditional'.
    p.add_argument("--out_dir", type=str,
                   default=os.path.join("data", "generated", "unconditional"))
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset split reconstruction — identical to generate_samples_gpus.py
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
    files    = sorted(f for f in glob.glob(os.path.join(abs_root, "*.csv"))
                      if os.path.basename(f) != "normalization_stats.csv")
    assert len(files) > 0, f"No CSVs found in {abs_root}"
    print(f"Data root   : {abs_root}  ({len(files)} files)")

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

    # Per-file temporal block hold-out — identical to csdi_train_gpus.py.
    if val_ratio is not None and 0 < val_ratio < 1:
        rng_np       = np.random.default_rng(seed)
        file_val_start = {}
        file_val_end   = {}
        n_skipped      = 0

        for fi, T in enumerate(file_lengths):
            val_block_size = max(seq_len, int(T * val_ratio))
            lo = seq_len
            hi = T - val_block_size
            if lo > hi:
                file_val_start[fi] = -1
                n_skipped += 1
                continue
            val_start              = int(rng_np.integers(lo, hi + 1))
            file_val_start[fi]     = val_start
            file_val_end[fi]       = val_start + val_block_size

        if n_skipped:
            warnings.warn(
                f"reconstruct_split: {n_skipped}/{len(file_lengths)} files too short "
                f"for both splits; all their windows go to training.",
            )

        train_pool  = []
        val_indices = []
        for i, (fi, start) in enumerate(flat_index):
            vs = file_val_start[fi]
            if vs == -1:
                train_pool.append(i)
                continue
            ve  = file_val_end[fi]
            end = start + seq_len
            if end <= vs:
                train_pool.append(i)
            elif start >= ve:
                train_pool.append(i)
            elif start >= vs and end <= ve:
                val_indices.append(i)
            # straddles boundary → discard (prevents data-point leakage)

        print(f"Val windows : {len(val_indices)}")
    else:
        val_indices = []
        train_pool  = list(range(dataset_size))
        print("Val windows : 0  (no val_split_ratio in config)")

    rng_torch      = torch.Generator().manual_seed(seed)
    shuffled_order = torch.randperm(len(train_pool), generator=rng_torch).tolist()
    train_pool     = [train_pool[i] for i in shuffled_order]

    if subset_size is not None:
        n_train = min(int(subset_size), len(train_pool))
    elif subset_ratio is not None:
        n_train = max(1, int(len(train_pool) * subset_ratio))
        n_train = min(n_train, len(train_pool))
    else:
        n_train = len(train_pool)

    train_indices = train_pool[:n_train]
    print(f"Train windows: {n_train}")

    return files, flat_index, train_indices, val_indices, date_to_idx, date_col


# ---------------------------------------------------------------------------
# Per-rank generation + gather to rank 0
# CHANGE[2]: Added is_unconditional parameter so that the function can serve
# both generation modes without duplicating any DDP/IO logic.
# ---------------------------------------------------------------------------

def run_split_ddp(
    split_name, window_indices, flat_index, files,
    feat_cols, close_idx, K, seq_len,
    processes, model, num_samples, num_steps,
    chunk_size,
    device, out_dir, rho,
    date_to_idx, date_col,
    rank, world_size, is_main,
    is_unconditional,   # CHANGE[2]: controls cond_mask, sample extraction, CSV name
):
    n_windows = len(window_indices)
    step_cols = [f"step_{t:03d}" for t in range(seq_len)]

    # ── Assign windows to this rank (round-robin) ─────────────────────────────
    local_global_idxs  = list(range(rank, n_windows, world_size))
    local_dataset_idxs = [window_indices[i] for i in local_global_idxs]
    n_local = len(local_dataset_idxs)

    # ── Load ground-truth windows + calendar time positions ───────────────────
    # Note for unconditional: the data loaded here is used solely to derive
    # observed_tp (absolute calendar positions fed to the time embedding).
    # observed_data itself is not consumed by the model (make_diff_input returns
    # x_t only when is_unconditional=True) and enforce_observed is a no-op
    # (cond_mask=0 → x = 0*obs + 1*x = x). Loading is still required to
    # maintain the correct tensor shapes expected by edm_sampler's signature.
    gt_windows_local = []
    gt_tp_local      = []
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

    # ── EDM Heun sampler — chunked over local windows ─────────────────────────
    gen_close_local = []
    n_chunks = max(1, -(-n_local // chunk_size))
    with torch.no_grad():
        for chunk_idx, chunk_start in enumerate(range(0, max(n_local, 1), chunk_size)):
            if chunk_start >= n_local:
                break
            chunk_end = min(chunk_start + chunk_size, n_local)
            n_chunk   = chunk_end - chunk_start
            B         = n_chunk * num_samples

            obs = torch.from_numpy(gt_windows_local[chunk_start:chunk_end]).to(device)
            obs = (obs.unsqueeze(1)
                      .expand(n_chunk, num_samples, K, seq_len)
                      .reshape(B, K, seq_len))

            # CHANGE[3]: cond_mask construction branches on is_unconditional.
            #
            # predict_close: 1 = observed (OHL channels), 0 = target (Close).
            # The model learns to generate Close given OHL context.
            #
            # unconditional: all zeros — no channels are observed; the full
            # sequence is generated from noise. This exactly mirrors the training
            # regime (mask_mode=unconditional sets cond_mask=0 for all channels
            # in csdi_train_gpus.py). Any 1s here would inject a conditioning
            # signal the model was never trained with, corrupting the output.
            #
            # The original code used ones(B,K,L) then zeroed col close_idx.
            # For K=1 that accidentally produced all-zeros (correct), but only
            # by coincidence. The explicit branch is correct for any K.
            if is_unconditional:
                cond_mask = torch.zeros(B, K, seq_len, device=device)
            else:
                cond_mask = torch.ones(B, K, seq_len, device=device)
                cond_mask[:, close_idx, :] = 0.0

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

            # CHANGE[4]: Sample extraction branches on is_unconditional.
            #
            # predict_close: extract only the Close channel (index close_idx).
            # unconditional: extract channel 0. For the trained model (K=1) this
            # is the only channel. Setting close_idx=0 via CHANGE[9] means both
            # branches resolve to samples[:,0,:] for the current checkpoint, but
            # the explicit branch documents intent and guards against future K>1
            # unconditional models where "extract one channel" would be wrong.
            if is_unconditional:
                close_preds = (samples[:, 0, :]
                               .cpu().numpy()
                               .reshape(n_chunk, num_samples, seq_len))
            else:
                close_preds = (samples[:, close_idx, :]
                               .cpu().numpy()
                               .reshape(n_chunk, num_samples, seq_len))

            for i in range(n_chunk):
                gen_close_local.append(close_preds[i])   # (num_samples, L)

            print(f"  [rank {rank} | {split_name}] chunk {chunk_idx + 1}/{n_chunks} done "
                  f"(windows {chunk_start + 1}–{chunk_end} / {n_local})")

    # ── Build local row lists ─────────────────────────────────────────────────
    local_gen_rows  = []
    local_ohlc_rows = []

    for li, (gw_idx, dataset_idx) in enumerate(zip(local_global_idxs, local_dataset_idxs)):
        fi, start = flat_index[dataset_idx]
        fname = os.path.basename(files[fi])

        for s_idx in range(num_samples):
            row = {
                "window_idx":   gw_idx,
                "sample_idx":   s_idx,
                "file":         fname,
                "window_start": start,
            }
            row.update(zip(step_cols, gen_close_local[li][s_idx]))
            local_gen_rows.append(row)

        for feat_idx, feat_name in enumerate(feat_cols):
            row = {
                "window_idx":   gw_idx,
                "feature":      feat_name,
                "file":         fname,
                "window_start": start,
            }
            row.update(zip(step_cols, gt_windows_local[li, feat_idx, :]))
            local_ohlc_rows.append(row)

    # ── Each rank writes its temp CSVs ────────────────────────────────────────
    df_gen_local  = pd.DataFrame(local_gen_rows)
    df_ohlc_local = pd.DataFrame(local_ohlc_rows)

    tmp_gen  = os.path.join(out_dir, f"_tmp_{split_name}_gen_rank{rank}.csv")
    tmp_ohlc = os.path.join(out_dir, f"_tmp_{split_name}_ohlc_rank{rank}.csv")

    df_gen_local.to_csv(tmp_gen,  index=False)
    df_ohlc_local.to_csv(tmp_ohlc, index=False)
    print(f"  [rank {rank} | {split_name}] wrote {len(df_gen_local)} gen rows and "
          f"{len(df_ohlc_local)} ohlc rows to temp files")

    # ── Barrier: all ranks must finish writing before rank 0 merges ───────────
    dist.barrier()

    if not is_main:
        return None, None

    # ── Rank 0: merge, sort, save final CSVs, delete temps ────────────────────
    gen_shards  = [pd.read_csv(os.path.join(out_dir, f"_tmp_{split_name}_gen_rank{r}.csv"))
                   for r in range(world_size)]
    ohlc_shards = [pd.read_csv(os.path.join(out_dir, f"_tmp_{split_name}_ohlc_rank{r}.csv"))
                   for r in range(world_size)]

    df_gen  = (pd.concat(gen_shards,  ignore_index=True)
               .sort_values(["window_idx", "sample_idx"])
               .reset_index(drop=True))
    df_ohlc = (pd.concat(ohlc_shards, ignore_index=True)
               .sort_values(["window_idx", "feature"])
               .reset_index(drop=True))

    gen_path = os.path.join(out_dir, f"{split_name}_generated_close.csv")

    # CHANGE[5]: GT CSV suffix is 'gt_features' for unconditional, 'gt_ohlc' for
    # conditional. For unconditional models the file contains only the single
    # target feature (Close); naming it 'gt_ohlc' would actively mislead
    # downstream analysis scripts that expect three OHL channels alongside Close.
    gt_suffix  = "gt_features" if is_unconditional else "gt_ohlc"
    ohlc_path  = os.path.join(out_dir, f"{split_name}_{gt_suffix}.csv")

    df_gen.to_csv(gen_path,  index=False)
    df_ohlc.to_csv(ohlc_path, index=False)

    for r in range(world_size):
        os.remove(os.path.join(out_dir, f"_tmp_{split_name}_gen_rank{r}.csv"))
        os.remove(os.path.join(out_dir, f"_tmp_{split_name}_ohlc_rank{r}.csv"))

    print(f"Saved {split_name} generated Close → {gen_path}  "
          f"({len(df_gen)} rows = {n_windows} windows × {num_samples} samples)")
    print(f"Saved {split_name} GT features     → {ohlc_path}  "
          f"({n_windows} windows × {K} features)")

    return df_gen, df_ohlc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── DDP init ──────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = (rank == 0)
    device     = torch.device(f"cuda:{rank}")

    args = parse_args()

    # CHANGE[6]: Resolve --n_samples / --num_samples precedence.
    # --n_samples is canonical; --num_samples is the deprecated alias from
    # generate_samples_gpus.py. --n_samples takes precedence when both are given.
    # If neither is provided, fail with a clear message rather than silently
    # using a None or 0 default that would produce an empty or degenerate run.
    if args.n_samples is not None:
        num_samples = args.n_samples
    elif args.num_samples is not None:
        if is_main:
            print("Warning: --num_samples is deprecated; use --n_samples instead.")
        num_samples = args.num_samples
    else:
        raise ValueError("Specify the number of samples per window with --n_samples.")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_main:
        print(f"DDP: {world_size} GPU(s) active")

    # ── Broadcast timestamp from rank 0 so all ranks use the same out_dir ─────
    ckpt_stem = os.path.splitext(args.checkpoint_name)[0]
    if is_main:
        ts_tensor = torch.tensor(
            [int(datetime.now().strftime("%Y%m%d%H%M%S"))], dtype=torch.long, device=device
        )
    else:
        ts_tensor = torch.zeros(1, dtype=torch.long, device=device)
    dist.broadcast(ts_tensor, src=0)
    run_timestamp = datetime.strptime(str(ts_tensor.item()), "%Y%m%d%H%M%S").strftime("%Y%m%d_%H%M%S")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint_path = os.path.join("checkpoints", args.checkpoint_folder, args.checkpoint_name)
    if is_main:
        print(f"Loading checkpoint: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    # CHANGE[7]: Cross-check --mask_mode against the training mode stored in
    # the checkpoint. The two generation modes require different model
    # architectures (input_dim=1 vs 2; side_dim differs by one mask channel —
    # see CSDIModel.__init__). Loading a conditional checkpoint and running
    # it with mask_mode=unconditional (or vice versa) would either cause a
    # state_dict shape mismatch on load or produce silently incorrect outputs.
    # Checking here, before model instantiation, gives an immediate clear error.
    is_unconditional        = bool(config["model"]["is_unconditional"])
    requested_unconditional = (args.mask_mode == "unconditional")
    if is_unconditional != requested_unconditional:
        raise ValueError(
            f"--mask_mode '{args.mask_mode}' does not match the checkpoint: "
            f"config['model']['is_unconditional']={is_unconditional}. "
            f"Use --mask_mode {'unconditional' if is_unconditional else 'predict_close'}."
        )

    # CHANGE[8]: --data_root overrides config["train"]["data_root"] before
    # reconstruct_split reads it. The checkpoint stores the training-time path;
    # at generation time (different node, reorganised storage) that path may not
    # exist. All ranks apply the override independently on their local config
    # copy, so no additional broadcast is needed.
    if args.data_root is not None:
        if is_main:
            print(f"Overriding data_root: {config['train']['data_root']} → {args.data_root}")
        config["train"]["data_root"] = args.data_root

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

    # CHANGE[9]: close_idx is determined without calling feat_cols.index("close").
    #
    # predict_close: close_idx is read from config["data"]["close_idx"] when
    # present (explicit training-time record), falling back to .index("close").
    # This preserves backwards compatibility with older checkpoints.
    #
    # unconditional: close_idx=0 is set directly. There is no "conditioning vs
    # target" split across channels; every channel is a target. Calling
    # .index("close") would crash if the column were named differently (e.g.
    # "log_adj_close") and is conceptually wrong — the index is not used to
    # partition features, only to keep the sample-extraction index (CHANGE[4])
    # well-defined for K=1 models.
    if is_unconditional:
        close_idx = 0
    else:
        if "close_idx" in config.get("data", {}):
            close_idx = int(config["data"]["close_idx"])
        else:
            close_idx = feat_cols.index("close")

    if is_main:
        print(f"Checkpoint   : {args.checkpoint_name}")
        print(f"mask_mode    : {args.mask_mode}  (is_unconditional={is_unconditional})")
        print(f"target_dim   : {target_dim}  |  seq_len: {seq_len}")
        print(f"columns      : {columns}")
        print(f"feat_cols    : {feat_cols}  (close_idx={close_idx})")
        print(f"SDE type     : {config['process']['sde_type']}")
        print(f"Noise sched. : {config['process']['noise_schedule']} "
              f"with N={config['process']['N']} steps")

    # ── Model (no DDP — each rank holds its own copy for inference) ───────────
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Diffusion processes ───────────────────────────────────────────────────
    if args.sigma_max is not None:
        if is_main:
            print(f"Overriding sigma_max: {config['process']['sigma_max']} → {args.sigma_max}")
        config["process"]["sigma_max"] = args.sigma_max

    processes = Diffusion_Processes(config["process"])
    num_steps = args.num_reverse_steps if args.num_reverse_steps is not None else processes.N

    # CHANGE[10]: rho resolution: CLI arg → config["edm"]["rho"] → default 7.0.
    # Separating rho from the checkpoint config allows tuning the sigma-schedule
    # curvature at generation time without retraining. The same pattern already
    # exists for sigma_max; this makes rho consistent with it.
    if args.edm_rho is not None:
        rho = float(args.edm_rho)
        if is_main:
            config_rho = config.get("edm", {}).get("rho", 7.0)
            print(f"Overriding rho: {config_rho} → {rho}")
    else:
        rho = float(config.get("edm", {}).get("rho", 7.0))

    if is_main:
        print(f"Diffusion_Processes — SDE: {processes.sde_type}, N: {processes.N}, "
              f"model_steps: {processes.model_steps}")
        print(f"Reverse steps: {num_steps}  |  rho: {rho}")

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = os.path.join(
        args.out_dir, ckpt_stem,
        f"csv{args.num_csv}_samples{num_samples}_steps{num_steps}_seed{args.seed}_{run_timestamp}",
    )
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()   # fail fast if any rank is unhealthy before real work starts

    # ── Reconstruct train/val split — all ranks independently ─────────────────
    repo_root = os.path.abspath(os.path.dirname(__file__))
    files, flat_index, train_indices, val_indices, date_to_idx, date_col = \
        reconstruct_split(config, repo_root)

    n_train_use = min(args.num_csv, len(train_indices))
    n_val_use   = min(args.num_csv, len(val_indices))

    if is_main:
        if n_train_use < args.num_csv:
            print(f"Warning: only {n_train_use} training windows available "
                  f"(requested {args.num_csv}).")
        if len(val_indices) == 0:
            print("Warning: no validation windows — skipping val generation.")
        elif n_val_use < args.num_csv:
            print(f"Warning: only {n_val_use} validation windows available "
                  f"(requested {args.num_csv}).")
        print(f"\nWindows to generate:")
        print(f"  Train : {n_train_use}  ×  {num_samples} samples  "
              f"= {n_train_use * num_samples} rows in CSV")
        if n_val_use:
            print(f"  Val   : {n_val_use}  ×  {num_samples} samples  "
                  f"= {n_val_use * num_samples} rows in CSV")

    selected_train = train_indices[:n_train_use]
    selected_val   = val_indices[:n_val_use]

    # ── W&B init (rank 0 only) ────────────────────────────────────────────────
    use_wandb = (args.wandb_project is not None) and is_main
    if use_wandb:
        run_name = args.wandb_run_name or f"gen_{args.mask_mode}_{ckpt_stem}"
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = run_name,
            config  = {
                "checkpoint_folder":  args.checkpoint_folder,
                "checkpoint_name":    args.checkpoint_name,
                "mask_mode":          args.mask_mode,
                "is_unconditional":   is_unconditional,
                "num_csv":            args.num_csv,
                "num_samples":        num_samples,
                "num_reverse_steps":  num_steps,
                "sde_type":           config["process"]["sde_type"],
                "noise_schedule":     config["process"]["noise_schedule"],
                "sde_N":              config["process"]["N"],
                "target_dim":         target_dim,
                "seq_len":            seq_len,
                "close_idx":          close_idx,
                "world_size":         world_size,
                "rho":                rho,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    # ── Generate for TRAIN split ──────────────────────────────────────────────
    if is_main:
        print("\n── Generating for TRAIN split ──")
    dist.barrier()

    df_gen_train, _ = run_split_ddp(
        split_name       = "train",
        window_indices   = selected_train,
        flat_index       = flat_index,
        files            = files,
        feat_cols        = feat_cols,
        close_idx        = close_idx,
        K                = K,
        seq_len          = seq_len,
        processes        = processes,
        model            = model,
        num_samples      = num_samples,
        num_steps        = num_steps,
        chunk_size       = args.chunk_size,
        device           = device,
        out_dir          = out_dir,
        rho              = rho,
        date_to_idx      = date_to_idx,
        date_col         = date_col,
        rank             = rank,
        world_size       = world_size,
        is_main          = is_main,
        is_unconditional = is_unconditional,
    )

    # ── Generate for VAL split ────────────────────────────────────────────────
    df_gen_val = None
    if selected_val:
        if is_main:
            print("\n── Generating for VAL split ──")
        dist.barrier()

        df_gen_val, _ = run_split_ddp(
            split_name       = "val",
            window_indices   = selected_val,
            flat_index       = flat_index,
            files            = files,
            feat_cols        = feat_cols,
            close_idx        = close_idx,
            K                = K,
            seq_len          = seq_len,
            processes        = processes,
            model            = model,
            num_samples      = num_samples,
            num_steps        = num_steps,
            chunk_size       = args.chunk_size,
            device           = device,
            out_dir          = out_dir,
            rho              = rho,
            date_to_idx      = date_to_idx,
            date_col         = date_col,
            rank             = rank,
            world_size       = world_size,
            is_main          = is_main,
            is_unconditional = is_unconditional,
        )

    plt.close("all")

    # ── W&B summary (rank 0 only) ─────────────────────────────────────────────
    if use_wandb and df_gen_train is not None:
        step_cols = [c for c in df_gen_train.columns if c.startswith("step_")]
        gen_np    = df_gen_train[step_cols].to_numpy()
        log_dict  = {
            "gen/n_train_windows": n_train_use,
            "gen/n_val_windows":   n_val_use,
            "gen/num_samples":     num_samples,
            "gen/world_size":      world_size,
            "gen/mask_mode":       args.mask_mode,
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
