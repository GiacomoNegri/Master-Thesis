# csdi_train.py  — EDM training script for CSDI-based financial time-series model
import os
import json
import random
import argparse
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

import yaml
import wandb

from src.models.model_core import CSDIModel
from src.utils.dataloader import csdi_collate_fn, make_dataloader
from src.training.edm_loss import EDMLoss

from src.utils.utils import set_seed, get_checkpoint_save_interval, load_checkpoint, build_final_checkpoint_name, build_run_metadata, unpack_batch, get_predict_close_mask, get_randmask, make_checkpoint_payload, get_final_config


# ----------------------------
# Training loop
# ----------------------------

def train(
    config:            Dict[str, Any],
    train_loader:      DataLoader,
    val_loader:        Optional[DataLoader] = None,
    resume_checkpoint: Optional[str]        = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["train"]["seed"]))

    # Model
    target_dim = int(config["data"]["target_dim"])
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)

    # EDM loss — sigma_data must equal model.sigma_data (both read from config["model"]["sigma_data"])
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
    os.makedirs(out_dir, exist_ok=True)

    num_epochs       = int(config["train"]["epochs"])
    steps_per_epoch  = len(train_loader)
    ckpt_every       = get_checkpoint_save_interval(num_epochs)
    run_timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        print(f"CosineAnnealingLR: lr={config['train']['lr']:.2e} → eta_min={eta_min:.2e} over {num_epochs} epochs")
    else:
        scheduler = None

    if resume_checkpoint is not None:
        print(f"Resuming from: {resume_checkpoint}")
        _ckpt_keys = set(torch.load(resume_checkpoint, map_location="cpu").keys())
        start_epoch, global_step, history = load_checkpoint(
            resume_checkpoint, model, optim, scaler, device, scheduler=scheduler
        )
        print(f"Resumed at epoch={start_epoch}, global_step={global_step}")
        if use_cosine and "scheduler" not in _ckpt_keys:
            for _ in range(start_epoch):
                scheduler.step()
            print(f"  [scheduler] Fast-forwarded cosine to epoch {start_epoch}")

    # W&B
    wandb_cfg = config.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("project"))
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
    best_val_loss   = float("inf")
    epochs_no_improve = 0

    close_idx = int(config["data"].get("close_idx", 3))

    # ---- epoch loop ----
    for epoch in range(start_epoch, num_epochs):
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=steps_per_epoch,
            desc=f"epoch {epoch+1}/{num_epochs}",
            dynamic_ncols=True,
        )
        epoch_loss_sum   = 0.0
        epoch_loss_count = 0
        ema_loss = None
        _grad_buf = []
        _loss_buf = []
        _sigma_buf = []
        ema_beta         = float(config["train"].get("ema_beta", 0.98))
        debug            = bool(config["train"].get("debug", False))
        loss_spike_factor = config["train"].get("loss_spike_factor", None)
        if loss_spike_factor is not None:
            loss_spike_factor = float(loss_spike_factor)
        mask_mode = config["train"].get("mask_mode", "random")
        t0 = time.time()

        for batch_idx, batch in pbar:
            observed_data, observed_mask, observed_tp = unpack_batch(batch, device)

            log_this_batch = debug and (batch_idx == 0)
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

            if epoch == 0 and batch_idx == 0:
                print("cond_mask per feature (mean over B,L):", cond_mask.float().mean(dim=(0, 2)))
                print("target_mask per feature (mean over B,L):", target_mask.float().mean(dim=(0, 2)))

            # EDM forward + loss
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, sigma_t = edm_loss_fn(
                    model=model,
                    observed_data=observed_data,
                    cond_mask=cond_mask,
                    target_mask=target_mask,
                    observed_tp=observed_tp,
                    debug=debug,
                )

            loss_val = float(loss.detach().item())
            if batch_idx == 0:
                print(f"  [batch 0] loss={loss_val:.6f}  "
                      f"sigma | min={sigma_t.float().min():.4f}  max={sigma_t.float().max():.4f}  "
                      f"mean={sigma_t.float().mean():.4f}  median={sigma_t.float().median():.4f}")
            elif log_this_batch:
                print(f"  sigma | min={sigma_t.float().min():.4f}  max={sigma_t.float().max():.4f}  "
                      f"mean={sigma_t.float().mean():.4f}  median={sigma_t.float().median():.4f}")
            epoch_loss_sum   += loss_val
            epoch_loss_count += 1
            ema_loss = loss_val if ema_loss is None else (ema_beta * ema_loss + (1.0 - ema_beta) * loss_val)

            if (debug and loss_spike_factor is not None and ema_loss is not None
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
            _sigma_buf.append(sigma_t.float().mean().item())
            if log_this_batch:
                _clip_flag = "CLIPPED" if grad_norm_v >= 4.99 else "ok"
                print(f"  grad_norm | {grad_norm_v:.4f}  ({_clip_flag})")

            pbar.set_postfix({
                "step": global_step,
                "loss": f"{loss_val:.4f}",
                "ema":  f"{ema_loss:.4f}",
                "avg":  f"{(epoch_loss_sum / epoch_loss_count):.4f}",
                "it/s": f"{it_per_s:.2f}",
                "σ":    f"{sigma_t.float().mean().item():.3f}",
                "gnorm":f"{grad_norm_v:.2f}",
            })

            if global_step % int(config["train"]["log_every_steps"]) == 0:
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
        if _grad_buf:
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
        if _sigma_buf:
            _sb = np.array(_sigma_buf)
            print(
                f"  [sigma] mean={_sb.mean():.4f}  p5={np.percentile(_sb,5):.4f}  "
                f"p50={np.median(_sb):.4f}  p95={np.percentile(_sb,95):.4f}  "
                f"min={_sb.min():.4f}  max={_sb.max():.4f}"
            )
        epoch_avg = epoch_loss_sum / max(epoch_loss_count, 1)

        # Validation
        val_avg = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_loader):
                    observed_data, observed_mask, observed_tp = unpack_batch(val_batch, device)
                    log_this_val_batch = debug and (val_batch_idx == 0)
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
                        val_loss, val_sigma_t = edm_loss_fn(
                            model=model,
                            observed_data=observed_data,
                            cond_mask=cond_mask,
                            target_mask=target_mask,
                            observed_tp=observed_tp,
                            debug=debug,
                        )
                    if log_this_val_batch:
                        print(f"  sigma | min={val_sigma_t.float().min():.4f}  max={val_sigma_t.float().max():.4f}  "
                              f"mean={val_sigma_t.float().mean():.4f}  median={val_sigma_t.float().median():.4f}")
                    val_loss_sum   += float(val_loss.item())
                    val_loss_count += 1

            val_avg = val_loss_sum / max(val_loss_count, 1)
            history["val_losses"].append({"epoch": epoch + 1, "avg_val_loss": val_avg})

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
            current_lr = scheduler.get_last_lr()[0]
            print(f"  [scheduler] lr={current_lr:.2e}")
            if use_wandb:
                wandb.log({"train/lr": current_lr}, step=global_step)

        # Early stopping
        if early_stop_patience is not None and val_avg is not None:
            if val_avg < best_val_loss:
                best_val_loss     = val_avg
                epochs_no_improve = 0
                best_path = os.path.join(out_dir, f"best_{run_timestamp}.pt")
                torch.save(
                    make_checkpoint_payload(model, optim, scaler, config, epoch, global_step, history, scheduler=scheduler),
                    best_path,
                )
                print(f"  [early-stop] New best val={best_val_loss:.6f} — saved {best_path}")
            else:
                epochs_no_improve += 1
                print(f"  [early-stop] No improvement {epochs_no_improve}/{early_stop_patience}")
                if epochs_no_improve >= early_stop_patience:
                    print(f"  [early-stop] Patience exhausted — stopping at epoch {epoch+1}")
                    if use_wandb:
                        wandb.log({"early_stop_epoch": epoch + 1}, step=global_step)
                    break

        # Periodic checkpoint
        if ((epoch + 1) % ckpt_every == 0) or ((epoch + 1) == num_epochs):
            ckpt_path = os.path.join(out_dir, f"ckpt_epoch_{epoch+1}_{run_timestamp}.pt")
            torch.save(
                make_checkpoint_payload(model, optim, scaler, config, epoch, global_step, history, scheduler=scheduler),
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint
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


# ----------------------------
# Entry point
# ----------------------------

def _real_main():
    config, args = get_final_config()

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
    if subset_ratio is not None:
        if not (0 < subset_ratio <= 1):
            raise ValueError("train_subset_ratio must be in (0, 1]")
        subset_size = max(1, int(dataset_size * subset_ratio))
    if subset_size is not None:
        if subset_size <= 0:
            raise ValueError("train_subset_size must be > 0")
        subset_size = min(subset_size, dataset_size)

    rng = torch.Generator().manual_seed(int(config["train"]["seed"]))
    all_indices = torch.randperm(dataset_size, generator=rng).tolist()

    # Carve out validation first (disjoint from training)
    val_loader     = None
    val_split_ratio = config["train"].get("val_split_ratio", None)
    if val_split_ratio is not None:
        if not (0 < val_split_ratio < 1):
            raise ValueError("val_split_ratio must be in (0, 1)")
        val_size    = max(1, int(dataset_size * val_split_ratio))
        val_indices = all_indices[:val_size]
        train_pool  = all_indices[val_size:]
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=csdi_collate_fn,
            )

        print(f"Validation set: {val_size}/{dataset_size} samples")
    else:
        train_pool = all_indices

    # Training subset from the remaining pool
    if subset_size is not None:
        subset_size = min(subset_size, len(train_pool))
        batch_size = config["train"]["batch_size"]
        subset_indices = train_pool[:subset_size]

        # If subset is smaller than batch_size, duplicate indices to fill batches
        if subset_size < batch_size:
            repeats = (batch_size + subset_size - 1) // subset_size  # ceil division
            subset_indices = subset_indices * repeats

        train_loader = DataLoader(
            Subset(dataset, subset_indices),
            batch_size=batch_size,
            shuffle=config["train"]["shuffle"],
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=getattr(train_loader, "collate_fn", None),
        )
        print(f"Training subset: {subset_size}/{dataset_size} samples (excl. val)")
        if len(subset_indices) > subset_size:
            print(f"  (expanded to {len(subset_indices)} indices to fill batch_size={batch_size})")
    elif val_split_ratio is not None:
        train_loader = DataLoader(
            Subset(dataset, train_pool),
            batch_size=config["train"]["batch_size"],
            shuffle=config["train"]["shuffle"],
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=getattr(train_loader, "collate_fn", None),
        )
        print(f"Training pool: {len(train_pool)}/{dataset_size} samples (excl. val)")
    else:
        print(f"Training: full dataset ({dataset_size} samples)")

    sample = next(iter(train_loader))
    print("Batch shapes:", sample["observed_data"].shape,
          sample["observed_mask"].shape, sample["observed_tp"].shape)

    train(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_checkpoint=args.resume_checkpoint,
    )
    print("Training is finished")


if __name__ == "__main__":
    import traceback as _tb, sys as _sys, faulthandler as _fh
    _fh.enable()
    try:
        _real_main()
    except SystemExit as _e:
        print(f"\n[FATAL] SystemExit({_e.code})", flush=True)
        _tb.print_exc()
        raise
    except BaseException as _e:
        print(f"\n[FATAL] {type(_e).__name__}: {_e}", flush=True)
        _tb.print_exc()
        _sys.exit(1)