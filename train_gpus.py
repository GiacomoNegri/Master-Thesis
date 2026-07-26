# csdi_train_gpus.py  — multi-GPU version via DistributedDataParallel (DDP)
# Launch with torchrun, e.g.:
#   torchrun --standalone --nproc_per_node=4 csdi_train_gpus.py --config ...
#
# Mirrors csdi_train.py exactly; DDP additions are marked with # DDP.
import os
import json
from typing import Any, Dict, Optional
from datetime import datetime
from tqdm import tqdm
import time

import torch
import torch.nn as nn

# Flash Attention CUDA kernels require a minimum number of SMs that MIG GPU
# slices (e.g. 4g.40gb) do not satisfy. Mem-efficient SDP has a hard CUDA
# kernel limit of batch*heads <= 65535, which is violated when seq_len is
# large (B*L*nheads can reach 500k+). Force math (standard softmax) SDP.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import wandb

from src.models.model_core import CSDIModel
from src.training.edm_loss import EDMLoss
from src.utils.dataloader import make_dataloader
from src.utils.utils import (
    set_seed, get_checkpoint_save_interval, load_checkpoint,
    build_final_checkpoint_name, build_run_metadata,
    unpack_batch, get_predict_close_mask, get_randmask,
    get_final_config,
)


# DDP: override make_checkpoint_payload to unwrap model.module before saving
def make_checkpoint_payload(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    history: Dict[str, Any],
    scheduler=None,
) -> Dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model  # DDP
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": raw_model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "run_metadata": build_run_metadata(config),
        "history": history,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


# ----------------------------
# Training loop
# ----------------------------

def train(
    config:            Dict[str, Any],
    train_loader:      DataLoader,
    val_loader:        Optional[DataLoader] = None,
    resume_checkpoint: Optional[str]        = None,
    train_sampler=None,    # DDP: DistributedSampler — needed to call set_epoch each epoch
    local_rank: int = 0,   # DDP
    is_main:    bool = True,  # DDP
) -> None:
    device = torch.device(f"cuda:{local_rank}")  # DDP: each process owns exactly one GPU
    set_seed(int(config["train"]["seed"]))

    # Model
    target_dim = int(config["data"]["target_dim"])
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)

    # EDM loss — sigma_data must equal model.sigma_data (both read from config)
    edm_cfg = config.get("edm", {})
    edm_loss_fn = EDMLoss(
        P_mean=float(edm_cfg.get("P_mean", -1.2)),
        P_std=float(edm_cfg.get("P_std",   1.2)),
        sigma_data=float(config["model"].get("sigma_data", 1.0)),
    )

    # Optimizer
    optim = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    # AMP
    use_amp = bool(config["train"].get("use_amp", True)) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = config["train"]["out_dir"]
    if is_main:  # DDP: only rank 0 creates directories
        os.makedirs(out_dir, exist_ok=True)

    num_epochs      = int(config["train"]["epochs"])
    steps_per_epoch = len(train_loader)
    ckpt_every      = get_checkpoint_save_interval(num_epochs)
    run_timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    start_epoch = 0
    global_step = 0
    history     = {"epoch_losses": [], "val_losses": []}

    # LR scheduler
    use_cosine = bool(config["train"].get("lr_cosine_annealing", False))
    eta_min    = float(config["train"].get("lr_eta_min", 1e-6))
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=num_epochs, eta_min=eta_min, last_epoch=-1
        )
        if is_main:
            print(f"CosineAnnealingLR: lr={config['train']['lr']:.2e} → eta_min={eta_min:.2e} over {num_epochs} epochs")
    else:
        scheduler = None

    # Resume checkpoint — must happen BEFORE DDP wrapping
    if resume_checkpoint is not None:
        if is_main:
            print(f"Resuming from: {resume_checkpoint}")
        start_epoch, global_step, history = load_checkpoint(
            resume_checkpoint, model, optim, scaler, device, scheduler=scheduler
        )
        if is_main:
            print(f"Resumed at epoch={start_epoch}, global_step={global_step}")
        if use_cosine and "scheduler" not in torch.load(resume_checkpoint, map_location="cpu"):
            for _ in range(start_epoch):
                scheduler.step()
            if is_main:
                print(f"  [scheduler] Fast-forwarded cosine to epoch {start_epoch}")

    # DDP: wrap model after checkpoint load, before training
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)  # needed when K=1: ResidualBlock.forward_feature skips feature_layer, leaving its params grad-free

    # DDP: per-rank seed offset so random masks differ across GPUs
    set_seed(int(config["train"]["seed"]) + local_rank)

    # W&B — only rank 0
    wandb_cfg = config.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("project")) and is_main  # DDP
    if use_wandb:
        auto_name = build_final_checkpoint_name(config, global_step=global_step).replace(".pt", "")
        init_kwargs = dict(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg["project"],
            name=wandb_cfg.get("run_name") or auto_name,
            config=build_run_metadata(config),
        )
        prior_run_id = history.get("wandb_run_id")
        if prior_run_id is not None:
            init_kwargs["id"]     = prior_run_id
            init_kwargs["resume"] = "must"
        wandb.init(**init_kwargs)
        history["wandb_run_id"] = wandb.run.id

    # Early stopping
    early_stop_patience = config["train"].get("early_stop_patience", None)
    if early_stop_patience is not None:
        early_stop_patience = int(early_stop_patience) if int(early_stop_patience) > 0 else None
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    close_idx = int(config["data"].get("close_idx", 3))

    # DDP: signal used by all ranks to synchronise early-stop decision
    stop_signal = torch.zeros(1, device=device)

    # ---- epoch loop ----
    for epoch in range(start_epoch, num_epochs):
        model.train()

        # DDP: tell the sampler which epoch so shuffling differs per epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(
            enumerate(train_loader),
            total=steps_per_epoch,
            desc=f"epoch {epoch+1}/{num_epochs}",
            dynamic_ncols=True,
            disable=not is_main,  # DDP: progress bar only on rank 0
        )

        epoch_loss_sum   = 0.0
        epoch_loss_count = 0
        ema_loss  = None
        _grad_buf = []
        _loss_buf = []
        _sigma_buf = []
        _sigma_mean_buf = []
        ema_beta          = float(config["train"].get("ema_beta", 0.98))
        debug             = bool(config["train"].get("debug", False))
        loss_spike_factor = config["train"].get("loss_spike_factor", None)
        if loss_spike_factor is not None:
            loss_spike_factor = float(loss_spike_factor)
        mask_mode = config["train"].get("mask_mode", "random")
        t0 = time.time()

        for batch_idx, batch in pbar:
            observed_data, observed_mask, observed_tp = unpack_batch(batch, device)

            log_this_batch = debug and (batch_idx == 0) and is_main
            if log_this_batch:
                print(f"\n[debug | epoch {epoch+1} | train | batch 0]")
                _d    = observed_data.float()
                _flat = _d.flatten()
                print(f"  data  | mean={_flat.mean():.4f}  std={_flat.std():.4f}  "
                      f"min={_flat.min():.4f}  max={_flat.max():.4f}  "
                      f"p5={_flat.quantile(0.05):.4f}  p50={_flat.quantile(0.50):.4f}  p95={_flat.quantile(0.95):.4f}")
                _win_mean = _d.mean(dim=-1).flatten()
                _win_std  = _d.std(dim=-1).flatten()
                print(f"  win μ | mean={_win_mean.mean():.4f}  std={_win_mean.std():.4f}  "
                      f"p5={_win_mean.quantile(0.05):.4f}  p50={_win_mean.quantile(0.50):.4f}  p95={_win_mean.quantile(0.95):.4f}")
                print(f"  win σ | mean={_win_std.mean():.4f}  std={_win_std.std():.4f}  "
                      f"p5={_win_std.quantile(0.05):.4f}  p50={_win_std.quantile(0.50):.4f}  p95={_win_std.quantile(0.95):.4f}")

            # Build conditioning and target masks
            if config["model"]["is_unconditional"] or mask_mode == "unconditional":
                cond_mask = torch.zeros_like(observed_mask)
            elif mask_mode == "predict_close":
                cond_mask = get_predict_close_mask(observed_mask, close_idx=close_idx)
            else:
                cond_mask = get_randmask(
                    observed_mask,
                    min_ratio=float(config["train"]["cond_min_ratio"]),
                    max_ratio=float(config["train"]["cond_max_ratio"]),
                )

            target_mask = (observed_mask.float() * (1.0 - cond_mask.float())).float()
            assert target_mask.sum() > 0, "No target entries — check conditioning mask logic."

            if epoch == 0 and batch_idx == 0 and is_main:
                print("cond_mask   per feature (mean over B,L):", cond_mask.float().mean(dim=(0, 2)))
                print("target_mask per feature (mean over B,L):", target_mask.float().mean(dim=(0, 2)))

            # EDM forward + loss
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, sigma_t, x_t = edm_loss_fn(
                    model=model,
                    observed_data=observed_data,
                    cond_mask=cond_mask,
                    target_mask=target_mask,
                    observed_tp=observed_tp,
                    debug=debug,
                )

            loss_val = float(loss.detach().item())
            if batch_idx == 0 and is_main:
                print(f"  [batch 0] loss={loss_val:.6f}  "
                      f"sigma | min={sigma_t.float().min():.4f}  max={sigma_t.float().max():.4f}  "
                      f"mean={sigma_t.float().mean():.4f}  median={sigma_t.float().median():.4f}")
            elif log_this_batch and is_main:
                print(f"  sigma | min={sigma_t.float().min():.4f}  max={sigma_t.float().max():.4f}  "
                      f"mean={sigma_t.float().mean():.4f}  median={sigma_t.float().median():.4f}")
            if log_this_batch:
                _xt_flat = x_t.float().flatten()
                print(f"  x_t   | mean={_xt_flat.mean():.4g}  std={_xt_flat.std():.4g}  "
                      f"min={_xt_flat.min():.4g}  max={_xt_flat.max():.4g}  "
                      f"p5={_xt_flat.quantile(0.05):.4g}  p95={_xt_flat.quantile(0.95):.4g}")
            epoch_loss_sum   += loss_val
            epoch_loss_count += 1
            ema_loss = loss_val if ema_loss is None else (ema_beta * ema_loss + (1.0 - ema_beta) * loss_val)

            if (debug and is_main and loss_spike_factor is not None and ema_loss is not None
                    and loss_val > loss_spike_factor * ema_loss):
                print(f"\n[loss spike | epoch {epoch+1} | batch {batch_idx}]  "
                      f"loss={loss_val:.6f}  ema={ema_loss:.6f}  ratio={loss_val/ema_loss:.2f}x")
                _d    = observed_data.float()
                _flat = _d.flatten()
                print(f"  data  | mean={_flat.mean():.4f}  std={_flat.std():.4f}  "
                      f"min={_flat.min():.4f}  max={_flat.max():.4f}  "
                      f"p5={_flat.quantile(0.05):.4f}  p50={_flat.quantile(0.50):.4f}  p95={_flat.quantile(0.95):.4f}")
                _win_mean = _d.mean(dim=-1).flatten()
                _win_std  = _d.std(dim=-1).flatten()
                print(f"  win μ | mean={_win_mean.mean():.4f}  std={_win_mean.std():.4f}  "
                      f"p5={_win_mean.quantile(0.05):.4f}  p50={_win_mean.quantile(0.50):.4f}  p95={_win_mean.quantile(0.95):.4f}")
                print(f"  win σ | mean={_win_std.mean():.4f}  std={_win_std.std():.4f}  "
                      f"p5={_win_std.quantile(0.05):.4f}  p50={_win_std.quantile(0.50):.4f}  p95={_win_std.quantile(0.95):.4f}")
                print(f"  sigma | min={sigma_t.float().min():.4f}  max={sigma_t.float().max():.4f}  "
                      f"mean={sigma_t.float().mean():.4f}  median={sigma_t.float().median():.4f}")
                # Noised data stats: exact x_t from the forward pass
                _xt_flat = x_t.float().flatten()
                print(f"  x_t   | mean={_xt_flat.mean():.4g}  std={_xt_flat.std():.4g}  "
                      f"min={_xt_flat.min():.4g}  max={_xt_flat.max():.4g}  "
                      f"p5={_xt_flat.quantile(0.05):.4g}  p95={_xt_flat.quantile(0.95):.4g}")
                # Per-sigma bin: how many samples fall in each noise level range
                _sig_np = sigma_t.float().cpu().numpy()
                _spike_bins = [(0.0, 0.1, "<0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, float("inf"), ">1.0")]
                _bin_str = "  sigma bins | " + "  ".join(
                    f"{lbl}:{int(((s >= lo) & (s < hi)).sum())}" for lo, hi, lbl in _spike_bins
                    for s in [_sig_np]
                )
                print(_bin_str)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()

            global_step += 1

            elapsed     = time.time() - t0
            it_per_s    = (batch_idx + 1) / max(elapsed, 1e-9)
            grad_norm_v = float(grad_norm)
            _grad_buf.append(grad_norm_v)
            _loss_buf.append(loss_val)
            _sigma_buf.extend(sigma_t.float().cpu().numpy())
            _sigma_mean_buf.append(sigma_t.float().mean().item())
            if log_this_batch:
                _clip_flag = "CLIPPED" if grad_norm_v >= 4.99 else "ok"
                print(f"  grad_norm | {grad_norm_v:.4f}  ({_clip_flag})")

            if is_main:  # DDP: postfix only on rank 0
                pbar.set_postfix({
                    "step":  global_step,
                    "loss":  f"{loss_val:.4f}",
                    "ema":   f"{ema_loss:.4f}",
                    "avg":   f"{(epoch_loss_sum / epoch_loss_count):.4f}",
                    "it/s":  f"{it_per_s:.2f}",
                    "σ":     f"{_sigma_mean_buf[-1]:.3f}",
                    "gnorm": f"{grad_norm_v:.2f}",
                })

            if is_main and global_step % int(config["train"]["log_every_steps"]) == 0:
                print(
                    f"epoch={epoch+1}/{num_epochs}  step={global_step}  "
                    f"loss={loss_val:.6f}  grad_norm={grad_norm_v:.4f}  "
                    f"sigma_mean={sigma_t.float().mean().item():.4f}"
                )
                if use_wandb:
                    wandb.log({
                        "train/step_loss": loss_val,
                        "train/ema_loss":  ema_loss,
                        "train/avg_loss":  epoch_loss_sum / epoch_loss_count,
                        "train/it_per_s":  it_per_s,
                        "train/grad_norm": grad_norm_v,
                    }, step=global_step)

        # ---- end of epoch ----
        if is_main and _grad_buf:
            _gn = np.array(_grad_buf)
            _lb = np.array(_loss_buf)
            _clip_pct = (_gn >= 4.99).mean() * 100
            _r_str = ""
            if len(_gn) > 1 and _gn.std() > 1e-12 and _lb.std() > 1e-12:
                _r = float(np.corrcoef(_lb, _gn)[0, 1])
                _r_str = f"  Pearson r(loss, grad_norm)={_r:.3f}"
            print(
                f"  [grad] mean={_gn.mean():.4f}  std={_gn.std():.4f}  "
                f"p50={np.median(_gn):.4f}  p95={np.percentile(_gn, 95):.4f}  "
                f"max={_gn.max():.4f}  clipped={_clip_pct:.1f}%{_r_str}"
            )
        if is_main and _sigma_buf:
            _sb = np.array(_sigma_buf)
            print(
                f"  [sigma] mean={_sb.mean():.4f}  "
                f"q25={np.quantile(_sb, 0.25):.4f}  q50={np.quantile(_sb, 0.50):.4f}  q75={np.quantile(_sb, 0.75):.4f}  "
                f"p5={np.percentile(_sb,5):.4f}  p95={np.percentile(_sb,95):.4f}  "
                f"min={_sb.min():.4f}  max={_sb.max():.4f}"
            )
        # Pearson r(sigma_mean, loss) — batch-level, aligned arrays
        if is_main and len(_sigma_mean_buf) > 1 and len(_loss_buf) == len(_sigma_mean_buf):
            _sm = np.array(_sigma_mean_buf)
            _lb2 = np.array(_loss_buf)
            if _sm.std() > 1e-12 and _lb2.std() > 1e-12:
                _r_sl = float(np.corrcoef(_sm, _lb2)[0, 1])
                print(f"  [sigma-loss] Pearson r(sigma_mean, loss)={_r_sl:.4f}")
            # Per-sigma-bin loss stats (batch-level)
            _epoch_bins = [(0.0, 0.1, "<0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, float("inf"), ">1.0")]
            _bin_parts = []
            for lo, hi, lbl in _epoch_bins:
                _mask = (_sm >= lo) & (_sm < hi)
                if _mask.sum() > 0:
                    _bl = _lb2[_mask]
                    _bin_parts.append(f"{lbl}: n={_mask.sum()}  mean={_bl.mean():.4g}  med={np.median(_bl):.4g}  max={_bl.max():.4g}")
            if _bin_parts:
                print("  [sigma bins]")
                for _bp in _bin_parts:
                    print(f"    {_bp}")
        if is_main and _loss_buf:
            _lb = np.array(_loss_buf)
            _q25 = np.quantile(_lb, 0.25)
            _q75 = np.quantile(_lb, 0.75)
            _iqr = _q75 - _q25
            print(
                f"  [loss] mean={_lb.mean():.6f}  std={_lb.std():.6f}  median={np.median(_lb):.6f}  "
                f"iqr={_iqr:.6f}  q25={_q25:.6f}  q75={_q75:.6f}  "
                f"p5={np.quantile(_lb, 0.05):.6f}  p95={np.quantile(_lb, 0.95):.6f}  "
                f"min={_lb.min():.6f}  max={_lb.max():.6f}"
            )
        epoch_avg = epoch_loss_sum / max(epoch_loss_count, 1)

        # Validation — all ranks evaluate their shard, then all-reduce the average
        val_avg = None
        if val_loader is not None:
            model.eval()
            val_loss_sum   = torch.zeros(1, device=device)
            val_loss_count = torch.zeros(1, device=device)
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_loader):
                    observed_data, observed_mask, observed_tp = unpack_batch(val_batch, device)
                    log_this_val_batch = debug and (val_batch_idx == 0) and is_main
                    if log_this_val_batch:
                        print(f"\n[debug | epoch {epoch+1} | val | batch 0]")
                        _d    = observed_data.float()
                        _flat = _d.flatten()
                        print(f"  data  | mean={_flat.mean():.4f}  std={_flat.std():.4f}  "
                              f"min={_flat.min():.4f}  max={_flat.max():.4f}  "
                              f"p5={_flat.quantile(0.05):.4f}  p50={_flat.quantile(0.50):.4f}  p95={_flat.quantile(0.95):.4f}")
                        _win_mean = _d.mean(dim=-1).flatten()
                        _win_std  = _d.std(dim=-1).flatten()
                        print(f"  win μ | mean={_win_mean.mean():.4f}  std={_win_mean.std():.4f}  "
                              f"p5={_win_mean.quantile(0.05):.4f}  p50={_win_mean.quantile(0.50):.4f}  p95={_win_mean.quantile(0.95):.4f}")
                        print(f"  win σ | mean={_win_std.mean():.4f}  std={_win_std.std():.4f}  "
                              f"p5={_win_std.quantile(0.05):.4f}  p50={_win_std.quantile(0.50):.4f}  p95={_win_std.quantile(0.95):.4f}")

                    if config["model"]["is_unconditional"] or mask_mode == "unconditional":
                        cond_mask = torch.zeros_like(observed_mask)
                    elif mask_mode == "predict_close":
                        cond_mask = get_predict_close_mask(observed_mask, close_idx=close_idx)
                    else:
                        cond_mask = get_randmask(
                            observed_mask,
                            min_ratio=float(config["train"]["cond_min_ratio"]),
                            max_ratio=float(config["train"]["cond_max_ratio"]),
                        )
                    target_mask = (observed_mask.float() * (1.0 - cond_mask.float())).float()

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_loss, val_sigma_t, _ = edm_loss_fn(
                            model=model,
                            observed_data=observed_data,
                            cond_mask=cond_mask,
                            target_mask=target_mask,
                            observed_tp=observed_tp,
                        )
                    if log_this_val_batch:
                        print(f"  sigma | min={val_sigma_t.float().min():.4f}  max={val_sigma_t.float().max():.4f}  "
                              f"mean={val_sigma_t.float().mean():.4f}  median={val_sigma_t.float().median():.4f}")
                    val_loss_sum   += val_loss.detach()
                    val_loss_count += 1

            # DDP: aggregate val loss across all ranks
            dist.all_reduce(val_loss_sum,   op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_count, op=dist.ReduceOp.SUM)
            val_avg = (val_loss_sum / val_loss_count.clamp(min=1)).item()

            if is_main:
                history["val_losses"].append({"epoch": epoch + 1, "avg_val_loss": val_avg})
            model.train()

        if is_main:
            history["epoch_losses"].append({
                "epoch": epoch + 1, "avg_train_loss": epoch_avg, "avg_val_loss": val_avg
            })

            if val_avg is not None:
                print(f"[epoch {epoch+1}/{num_epochs}]  train={epoch_avg:.6f}  val={val_avg:.6f}")
            else:
                print(f"[epoch {epoch+1}/{num_epochs}]  train={epoch_avg:.6f}")

            if use_wandb:
                epoch_metrics = {
                    "epoch/train_loss":   epoch_avg,
                    "epoch/epoch":        epoch + 1,
                    "epoch/epoch_time_s": time.time() - t0,
                }
                if val_avg is not None:
                    epoch_metrics["epoch/val_loss"] = val_avg
                wandb.log(epoch_metrics, step=global_step)

        if scheduler is not None:
            scheduler.step()
            if is_main:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [scheduler] lr={current_lr:.2e}")
                if use_wandb:
                    wandb.log({"train/lr": current_lr}, step=global_step)

        # Early stopping — rank 0 decides, result broadcast to all ranks
        if early_stop_patience is not None and val_avg is not None:
            if is_main:
                if val_avg < best_val_loss:
                    best_val_loss     = val_avg
                    epochs_no_improve = 0
                    best_path = os.path.join(out_dir, f"best_{run_timestamp}.pt")
                    torch.save(
                        make_checkpoint_payload(model, optim, scaler, config, epoch, global_step, history, scheduler=scheduler),
                        best_path,
                    )
                    print(f"  [early-stop] New best val={best_val_loss:.6f} — saved {best_path}")
                    stop_signal.fill_(0)
                else:
                    epochs_no_improve += 1
                    print(f"  [early-stop] No improvement {epochs_no_improve}/{early_stop_patience}")
                    if epochs_no_improve >= early_stop_patience:
                        print(f"  [early-stop] Patience exhausted — stopping at epoch {epoch+1}")
                        if use_wandb:
                            wandb.log({"early_stop_epoch": epoch + 1}, step=global_step)
                        stop_signal.fill_(1)

            # DDP: broadcast stop decision from rank 0 to all other ranks
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1:
                break

        # Periodic checkpoint — rank 0 only
        if is_main:
            if ((epoch + 1) % ckpt_every == 0) or ((epoch + 1) == num_epochs):
                ckpt_path = os.path.join(out_dir, f"ckpt_epoch_{epoch+1}_{run_timestamp}.pt")
                torch.save(
                    make_checkpoint_payload(model, optim, scaler, config, epoch, global_step, history, scheduler=scheduler),
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint — rank 0 only
    if is_main:
        final_payload = make_checkpoint_payload(model, optim, scaler, config, epoch, global_step, history, scheduler=scheduler)
        final_payload["training_completed"] = (epoch + 1) == num_epochs
        final_payload["early_stopped"]      = (epoch + 1) < num_epochs

        final_name = build_final_checkpoint_name(config=config, global_step=global_step, actual_epoch=epoch + 1)
        final_path = os.path.join(out_dir, final_name)
        print(f"[checkpoint] Saving final to {final_path} ...", flush=True)
        torch.save(final_payload, final_path)
        print(f"[checkpoint] Saved OK.", flush=True)

        if use_wandb:
            print("[wandb] Finishing run ...", flush=True)
            wandb.finish()
            print("[wandb] Done.", flush=True)

        print("Training complete.", flush=True)

    dist.destroy_process_group()  # DDP


# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    # DDP: init process group before anything else
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    is_main    = (local_rank == 0)

    config, args = get_final_config()

    # DDP: scale LR linearly with the number of GPUs (linear scaling rule)
    base_lr = float(config["train"]["lr"])
    config["train"]["lr"] = base_lr * world_size
    if is_main:
        print(f"DDP: {world_size} GPUs — LR scaled {base_lr:.2e} → {config['train']['lr']:.2e}")
        print("Final merged config:")
        print(json.dumps(config, indent=2))

    train_loader = make_dataloader(
        root_dir=config["train"]["data_root"],
        batch_size=config["train"]["batch_size"],
        seq_len=config["train"]["seq_len"],
        stride=config["train"]["stride"],
        num_workers=config["train"]["num_workers"],
        shuffle=config["train"]["shuffle"],
        pin_memory=config["train"]["pin_memory"],
        columns=tuple(config["data"].get("columns", ("date", "log_adj_close"))),
    )

    dataset      = train_loader.dataset
    dataset_size = len(dataset)

    subset_ratio = config["train"].get("train_subset_ratio", None)
    subset_size  = config["train"].get("train_subset_size",  None)

    if subset_ratio is not None and subset_size is not None:
        raise ValueError("Use only one of train_subset_ratio or train_subset_size")
    if subset_size is not None and subset_size <= 0:
        raise ValueError("train_subset_size must be > 0")

    # --- Temporal validation split (identical on all ranks — same seed) ---
    # split_indices places a contiguous hold-out block at an independently
    # sampled position within each file, so held-out calendar dates differ
    # across files.  Any given date is absent from training only in the
    # fraction of files whose block happens to cover it; temporal embeddings
    # receive gradient signal for all dates across the full corpus.
    # The method is pure Python/NumPy and fully deterministic given the seed,
    # so every rank arrives at the same train_pool / val_indices without any
    # inter-rank communication.  Windows that straddle a block boundary are
    # discarded — they are the natural gap preventing data-point leakage.
    val_loader  = None
    val_sampler = None
    val_split_ratio = config["train"].get("val_split_ratio", None)
    if val_split_ratio is not None:
        if not (0 < val_split_ratio < 1):
            raise ValueError("val_split_ratio must be in (0, 1)")
        train_pool, val_indices = dataset.split_indices(
            val_fraction=val_split_ratio,
            seed=int(config["train"]["seed"]),
        )
        val_dataset = Subset(dataset, val_indices)
        # DDP: each rank evaluates a disjoint shard of the val set
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader  = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            sampler=val_sampler,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=train_loader.collate_fn,
        )
        if is_main:
            print(f"Validation set: {len(val_indices)} windows (temporal blocks, no overlap)")
    else:
        train_pool = list(range(dataset_size))

    # --- Training pool subset ---
    # Shuffle training pool with the same seed on all ranks so the order is
    # identical everywhere.  Any partial subset is then a random draw rather
    # than the first N sequential windows from self.index.
    rng            = torch.Generator().manual_seed(int(config["train"]["seed"]))
    shuffled_order = torch.randperm(len(train_pool), generator=rng).tolist()
    train_pool     = [train_pool[i] for i in shuffled_order]

    # Subset fraction/size is relative to the training pool only (not the
    # full dataset), so fractions remain meaningful after the val carve-out.
    if subset_ratio is not None:
        if not (0 < subset_ratio <= 1):
            raise ValueError("train_subset_ratio must be in (0, 1]")
        subset_size = max(1, int(len(train_pool) * subset_ratio))
    if subset_size is not None:
        subset_size = min(subset_size, len(train_pool))

    if subset_size is not None:
        train_dataset = Subset(dataset, train_pool[:subset_size])
        if is_main:
            print(f"Training subset: {subset_size}/{len(train_pool)} windows from training pool")
    else:
        train_dataset = Subset(dataset, train_pool)
        if is_main:
            print(f"Training pool: {len(train_pool)} windows (excl. val and boundary-discarded)")

    # DDP: DistributedSampler splits the training data across ranks
    train_sampler = DistributedSampler(train_dataset, shuffle=config["train"]["shuffle"])
    train_loader  = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        sampler=train_sampler,  # shuffle=False because sampler handles it
        num_workers=config["train"]["num_workers"],
        pin_memory=config["train"]["pin_memory"],
        collate_fn=train_loader.collate_fn,
        drop_last=True,
    )

    if is_main:
        batch = next(iter(train_loader))
        print(
            batch["observed_data"].shape,
            batch["observed_mask"].shape,
            batch["observed_tp"].shape,
        )

    train(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_checkpoint=args.resume_checkpoint,
        train_sampler=train_sampler,
        local_rank=local_rank,
        is_main=is_main,
    )