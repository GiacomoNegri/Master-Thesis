# train_csdi.py
import os
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.models.model_core import CSDIModel
from src.utils.WIP_processes import Diffusion_Processes
from src.utils.dataloader import make_dataloader


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Conditioning mask helper
# ----------------------------
def get_randmask(observed_mask: torch.Tensor, min_ratio: float = 0.1, max_ratio: float = 0.9) -> torch.Tensor:
    """
    Create a conditional mask (what is revealed to the model) by randomly hiding
    a fraction of the observed entries.

    observed_mask: (B,K,L) in {0,1}  -> indicates which entries are truly observed in data
    returns cond_mask: (B,K,L) in {0,1}  -> subset of observed_mask used as conditioning
    """
    device = observed_mask.device
    B = observed_mask.shape[0]

    # Choose a random missing ratio per sample
    miss_ratio = torch.empty(B, device=device).uniform_(min_ratio, max_ratio)  # fraction to HIDE
    rand = torch.rand_like(observed_mask.float())  # (B,K,L)
    # keep if rand > miss_ratio; but only where observed_mask==1
    keep = (rand > miss_ratio.view(B, 1, 1)).float()
    cond_mask = (observed_mask.float() * keep).float()
    return cond_mask


# ----------------------------
# Expected batch format
# ----------------------------
def unpack_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assumes your DataLoader returns either:
      - dict with keys: observed_data, observed_mask, observed_tp
      - or a tuple/list: (observed_data, observed_mask, observed_tp)

    Shapes expected:
      observed_data: (B,K,L) float
      observed_mask: (B,K,L) {0,1}
      observed_tp:   (B,L) float (time positions, e.g. 0..L-1 or normalized)
    """
    if isinstance(batch, dict):
        observed_data = batch["observed_data"]
        observed_mask = batch["observed_mask"]
        observed_tp = batch["observed_tp"]
    else:
        observed_data, observed_mask, observed_tp = batch

    return (
        observed_data.to(device),
        observed_mask.to(device),
        observed_tp.to(device),
    )


# ----------------------------
# Loss: MSE on target (non-conditioned) entries
# ----------------------------
def masked_mse(eps_hat: torch.Tensor, eps: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    eps_hat: (B,K,L)
    eps:     (B,K,L)
    target_mask: (B,K,L) in {0,1} where 1 = predict/score this location
    """
    denom = target_mask.sum().clamp(min=1.0)
    return ((eps_hat - eps) ** 2 * target_mask).sum() / denom


# ----------------------------
# Main training loop
# ----------------------------
def train(
    config: Dict[str, Any],
    train_loader,
    val_loader=None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["train"]["seed"]))

    # 1) Instantiate diffusion processes (forward noising) based on SDE definitions
    processes = Diffusion_Processes(config["process"])
    # forward_process uses sde.marginal_prob(x0,t) under the hood :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

    # 2) Instantiate denoiser model (CSDIModel wraps diff_CSDI backbone)
    target_dim = int(config["data"]["target_dim"])
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)  # :contentReference[oaicite:6]{index=6}
    model.train()

    # 3) Optimizer
    optim = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    # Optional: AMP
    use_amp = bool(config["train"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 4) Checkpointing
    out_dir = config["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_every = int(config["train"]["save_every_steps"])

    # Add for tracking
    num_epochs = int(config["train"]["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    
    global_step = 0
    # for epoch in range(int(config["train"]["epochs"])):
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=steps_per_epoch,
            desc=f"epoch {epoch+1}/{num_epochs}",
            dynamic_ncols=True,
        )

        # Just for tracking performance
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        ema_loss = None
        ema_beta = float(config["train"].get("ema_beta", 0.98))  # smoothing
        # Just for tracking performance

        t0 = time.time()

        for batch_idx, batch in pbar:
            observed_data, observed_mask, observed_tp = unpack_batch(batch, device)

            # (A) Build conditioning mask (what the model can see as context)
            if config["model"]["is_unconditional"]:
                cond_mask = torch.zeros_like(observed_mask)  # no conditioning channel used
            else:
                cond_mask = get_randmask(
                    observed_mask,
                    min_ratio=float(config["train"]["cond_min_ratio"]),
                    max_ratio=float(config["train"]["cond_max_ratio"]),
                )

            # The model should only be penalized on entries that are:
            #   - truly observed in dataset (observed_mask==1)
            #   - NOT revealed in the conditioning (cond_mask==0)
            target_mask = (observed_mask.float() * (1.0 - cond_mask.float())).float()

            # (B) Forward diffusion: sample x_t and eps at a random continuous time t
            # forward_process returns x_t (same shape as x0), continuous t in [0,T], and eps :contentReference[oaicite:7]{index=7}
            x_t, t_cont, eps = processes.forward_process(observed_data)

            # (C) Map continuous time -> discrete model step index for DiffusionEmbedding
            # This matches your reverse_process logic using TimeMapper.cont_to_idx :contentReference[oaicite:8]{index=8}
            t_idx = processes.mapper.cont_to_idx(t_cont)  # (B,) long in [0, model_steps-1]

            # (D) Predict eps with the denoiser
            # CSDIModel.forward() builds side_info and makes the 1/2-channel input for diff_CSDI :contentReference[oaicite:9]{index=9}
            with torch.cuda.amp.autocast(enabled=use_amp):
                eps_hat = model(
                    x_t=x_t,
                    t=t_idx,
                    observed_data=observed_data,
                    cond_mask=cond_mask,
                    observed_tp=observed_tp,
                )  # (B,K,L)

                loss = masked_mse(eps_hat, eps, target_mask)
            
            # Just for tracking performance
            loss_val = float(loss.detach().item())
            epoch_loss_sum += loss_val
            epoch_loss_count += 1

            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * loss_val
            # Just for tracking performance

            # (E) Optimize
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            global_step += 1

            # Just for tracking performance
            elapsed = time.time() - t0
            steps_done = batch_idx + 1
            it_per_s = steps_done / max(elapsed, 1e-9)

            pbar.set_postfix({
                "step": global_step,
                "loss": f"{loss_val:.4f}",                 # last batch loss
                "ema": f"{ema_loss:.4f}",                  # smoothed loss
                "avg": f"{(epoch_loss_sum/epoch_loss_count):.4f}",  # epoch running avg
                "it/s": f"{it_per_s:.2f}",
                "t": f"{t_idx.float().mean().item():.1f}",
            })
            # Update bar every step (cheap)
            pbar.set_postfix({
                "step": global_step,
                "loss": f"{loss.item():.4f}",
                "t": f"{t_idx.float().mean().item():.1f}",
            })
            # Just for tracking performance

            # (F) Logging (minimal)
            if global_step % int(config["train"]["log_every_steps"]) == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"t_idx_mean={t_idx.float().mean().item():.2f}"
                )

            # (G) Save checkpoint
            if global_step % save_every == 0:
                ckpt_path = os.path.join(out_dir, f"ckpt_step_{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "config": config,
                    },
                    ckpt_path,
                )
                print(f"Saved: {ckpt_path}")

    # Final save
    torch.save({"model": model.state_dict(), "config": config}, os.path.join(out_dir, "final.pt"))
    print("Training complete. Saved final.pt")


if __name__ == "__main__":
    # Example config dict (you can externalize later)
    config = {
        "data": {
            "target_dim": 5,  # K = number of variables/features
        },
        "model": {
            "timeemb": 128,
            "featureemb": 64,
            "is_unconditional": False,
        },
        "diffusion": {
            # These are used by diff_CSDI + DiffusionEmbedding :contentReference[oaicite:10]{index=10}
            "num_steps": 50,                 # should match process.model_steps
            "diffusion_embedding_dim": 128,
            "channels": 64,
            "layers": 4,
            "nheads": 8,
            "is_linear": False,
        },
        "process": {
            # These are used by Diffusion_Processes to pick the SDE and discretization :contentReference[oaicite:11]{index=11}
            "N": 1000,               # SDE discretization steps (mainly for sampling; training uses marginal_prob)
            "sde_type": "ve",        # "ve", "vp", "subVP", else -> GBMLogSDE in your code :contentReference[oaicite:12]{index=12}
            "model_steps": 50,       # discrete steps used by the neural net (t_idx range)
            "eps": 1e-3,
            "enforce_observed": True,
        },
        "train": {
            "seed": 42,
            "epochs": 1,
            "batch_size":32,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "use_amp": True,
            "cond_min_ratio": 0.1,         # reveal at least 10% of observed entries
            "cond_max_ratio": 0.9,         # reveal at most 90%
            "log_every_steps": 50,
            "save_every_steps": 1000,
            "seq_len":252,
            "stride":5,
            "num_workers":0,
            "shuffle":False,
            "pin_memory": False, #pin_memory=True tells the loader to allocate batch tensors in page-locked (pinned) CPU memory.
            "out_dir": "./checkpoints/csdi",
            "data_root": "./data/sp500_individual_gbm"
        },
    }

    train_loader = make_dataloader(
        root_dir=config["train"]["data_root"],
        batch_size=config["train"]["batch_size"],
        seq_len=config["train"]["seq_len"],
        stride=config["train"]["stride"],
        num_workers=config["train"]["num_workers"],
        shuffle=config["train"]["shuffle"],
        pin_memory=config["train"]["pin_memory"],
    )

    #Sanity check all good, you sure, sure 10000%.
    batch = next(iter(train_loader))
    print(batch["observed_data"].shape, batch["observed_mask"].shape, batch["observed_tp"].shape)

    train(config, train_loader)
    print("Training is finished")