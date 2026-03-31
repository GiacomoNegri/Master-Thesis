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
from torch.utils.data import DataLoader, Subset

from src.models.model_core import CSDIModel
from src.utils.WIP_processes import Diffusion_Processes
from src.utils.dataloader import make_dataloader

import yaml
from copy import deepcopy
from typing import Any, Dict

import wandb

# ----------------------------
# YAML uploader & updater
# ----------------------------
def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update nested dictionaries.
    Values in `updates` overwrite values in `base`.
    """
    out = deepcopy(base)
    for key, value in updates.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out

# ----------------------------
# Parse CLI arguments
# ----------------------------
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )

    # train overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--use_amp", type=str2bool, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    # process overrides
    parser.add_argument("--sde_type", type=str, default=None)
    parser.add_argument("--model_steps", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument(
        "--noise_schedule",
        type=str,
        default=None,
        choices=["linear", "exponential", "cosine"],
        help=(
            "Beta/sigma noise schedule shape. "
            "'linear': linear interpolation (VP/subVP default); "
            "'exponential': exponential interpolation; "
            "'cosine': cosine-smoothed schedule."
        ),
    )
    parser.add_argument("--sigma_min", type=float, default=None,
                        help="Minimum sigma for VE/GBM SDEs (overrides config)")
    parser.add_argument("--sigma_max", type=float, default=None,
                        help="Maximum sigma for VE/GBM SDEs (overrides config)")
    parser.add_argument("--beta_min", type=float, default=None,
                        help="Minimum beta for VP/subVP SDEs (overrides config)")
    parser.add_argument("--beta_max", type=float, default=None,
                        help="Maximum beta for VP/subVP SDEs (overrides config)")

    # model overrides
    parser.add_argument("--is_unconditional", type=str2bool, default=None)
    parser.add_argument("--timeemb", type=int, default=None)
    parser.add_argument("--featureemb", type=int, default=None)

    # data overrides
    parser.add_argument("--target_dim", type=int, default=None)

    # resume training
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from"
        )
    
    # training only on a portion
    parser.add_argument("--train_subset_ratio", type=float, default=None,
                    help="Fraction of dataset to use for training, e.g. 0.1 for 10%")
    parser.add_argument("--train_subset_size", type=int, default=None,
                        help="Number of samples to use for training")
    parser.add_argument("--val_split_ratio", type=float, default=None,
                        help="Fraction of dataset to hold out for validation, e.g. 0.1 for 10%")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Stop training if val loss does not improve for this many epochs. Requires val_split_ratio. 0 or omit to disable.")
    parser.add_argument("--likelihood_weighting", type=str2bool, default=None,
                        help="If true, weight the loss by 1/sigma^2(t) (likelihood weighting). If false, use plain MSE.")
    parser.add_argument(
        "--mask_mode",
        type=str,
        default=None,
        choices=["random", "unconditional", "predict_close"],
        help=(
            "Conditioning strategy during training. "
            "'random': randomly hide 10-90%% of entries (default CSDI); "
            "'unconditional': no context at all (requires is_unconditional=true); "
            "'predict_close': always condition on Open/High/Low/Volume, predict Close."
        ),
    )

    # wandb
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name. If omitted, W&B logging is disabled.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team or username), e.g. 'thesis-giacomo-negri'")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (optional, auto-generated if not set)")

    return parser.parse_args()

# ----------------------------
# Build overridden dictionary
# ----------------------------

def build_cli_override_dict(args) -> Dict[str, Any]:
    override = {
        "data": {},
        "model": {},
        "process": {},
        "train": {},
        "wandb": {},
    }

    # data
    if args.target_dim is not None:
        override["data"]["target_dim"] = args.target_dim

    # model
    if args.timeemb is not None:
        override["model"]["timeemb"] = args.timeemb
    if args.featureemb is not None:
        override["model"]["featureemb"] = args.featureemb
    if args.is_unconditional is not None:
        override["model"]["is_unconditional"] = args.is_unconditional

    # process
    if args.sde_type is not None:
        override["process"]["sde_type"] = args.sde_type
    if args.model_steps is not None:
        override["process"]["model_steps"] = args.model_steps
    if args.N is not None:
        override["process"]["N"] = args.N
    if args.noise_schedule is not None:
        override["process"]["noise_schedule"] = args.noise_schedule
    if args.sigma_min is not None:
        override["process"]["sigma_min"] = args.sigma_min
    if args.sigma_max is not None:
        override["process"]["sigma_max"] = args.sigma_max
    if args.beta_min is not None:
        override["process"]["beta_min"] = args.beta_min
    if args.beta_max is not None:
        override["process"]["beta_max"] = args.beta_max

    # train
    if args.epochs is not None:
        override["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        override["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        override["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        override["train"]["weight_decay"] = args.weight_decay
    if args.use_amp is not None:
        override["train"]["use_amp"] = args.use_amp
    if args.save_every_steps is not None:
        override["train"]["save_every_steps"] = args.save_every_steps
    if args.log_every_steps is not None:
        override["train"]["log_every_steps"] = args.log_every_steps
    if args.data_root is not None:
        override["train"]["data_root"] = args.data_root
    if args.out_dir is not None:
        override["train"]["out_dir"] = args.out_dir
    if args.train_subset_ratio is not None:
        override["train"]["train_subset_ratio"] = args.train_subset_ratio
    if args.train_subset_size is not None:
        override["train"]["train_subset_size"] = args.train_subset_size
    if args.val_split_ratio is not None:
        override["train"]["val_split_ratio"] = args.val_split_ratio
    if args.early_stop_patience is not None:
        override["train"]["early_stop_patience"] = args.early_stop_patience
    if args.mask_mode is not None:
        override["train"]["mask_mode"] = args.mask_mode
    if args.likelihood_weighting is not None:
        override["train"]["likelihood_weighting"] = args.likelihood_weighting

    # wandb
    if args.wandb_project is not None:
        override["wandb"]["project"] = args.wandb_project
    if args.wandb_entity is not None:
        override["wandb"]["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        override["wandb"]["run_name"] = args.wandb_run_name

    # remove empty sections
    override = {k: v for k, v in override.items() if v}
    return override

# ----------------------------
# Final Configuration builder
# ----------------------------
def get_final_config() -> Dict[str, Any]:
    args = parse_args()

    yaml_config = load_yaml_config(args.config)
    cli_override = build_cli_override_dict(args)

    final_config = deep_update(yaml_config, cli_override)
    return final_config, args

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


def get_predict_close_mask(observed_mask: torch.Tensor, close_idx: int = 3) -> torch.Tensor:
    """
    Fixed conditioning mask for 'predict Close' mode.
    All features except Close are fully revealed as context;
    Close (feature index close_idx) is always the prediction target.

    Feature order matches the dataloader: Open=0, High=1, Low=2, Close=3, Volume=4
    """
    cond_mask = observed_mask.clone().float()
    cond_mask[:, close_idx, :] = 0.0
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
def masked_mse(eps_hat, eps, target_mask, sigma_t=None):
    """
    sigma_t: optional (B,) tensor of per-sample noise std from the forward process.
             When provided, applies likelihood weighting 1/sigma^2(t) to each position,
             concentrating the gradient signal at low-t (where the true data distribution
             is most visible). When None, plain unweighted MSE is used.
    """
    sq_err = (eps_hat - eps) ** 2  # (B, K, L)

    if sigma_t is not None:
        # 1/sigma^2(t) per batch element, broadcast to (B, K, L)
        weight = (1.0 / sigma_t.float().pow(2).clamp(min=1e-8))[:, None, None]
        denom = (target_mask.float() * weight).sum().clamp(min=1e-10)
        loss = (sq_err * weight * target_mask).sum() / denom
    else:
        denom = target_mask.sum().clamp(min=1.0)
        loss = (sq_err * target_mask).sum() / denom

    # DEBUGGING: temporary debugging output to understand error distribution
    masked = sq_err[target_mask.bool()]  # unweighted, for interpretable diagnostics
    print(f"  err     | mean={masked.mean():.4f}  std={masked.std():.4f}  "
          f"p50={masked.median():.4f}  p95={masked.quantile(0.95):.4f}  "
          f"p99={masked.quantile(0.99):.4f}  max={masked.max():.4f}")

    if sigma_t is not None:
        w_masked = (sq_err * weight)[target_mask.bool()]
        print(f"  err(LW) | mean={w_masked.mean():.4f}  std={w_masked.std():.4f}  "
              f"p50={w_masked.median():.4f}  p95={w_masked.quantile(0.95):.4f}  "
              f"p99={w_masked.quantile(0.99):.4f}  max={w_masked.max():.4f}")

    return loss



# ----------------------------
# Main training loop
# ----------------------------

# Helper function so that we save at ~10% of epochs, or every 5 epochs if total < 10
def get_checkpoint_save_interval(num_epochs: int) -> int:
    """
    Save every 5 epochs if total epochs < 10, otherwise every 10% of total epochs.
    """
    if num_epochs < 10:
        return 5
    return max(1, num_epochs // 10)

# Helper funciton to report the checkpoints parameters
def build_run_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the most relevant config values for quick inspection in saved .pt files.
    The full config is still saved separately.
    """
    return {
        "train": {
            "seed": config["train"]["seed"],
            "epochs": config["train"]["epochs"],
            "batch_size": config["train"]["batch_size"],
            "lr": config["train"]["lr"],
            "weight_decay": config["train"]["weight_decay"],
            "use_amp": config["train"]["use_amp"],
            "cond_min_ratio": config["train"]["cond_min_ratio"],
            "cond_max_ratio": config["train"]["cond_max_ratio"],
            "mask_mode": config["train"].get("mask_mode", "random"),
            "seq_len": config["train"]["seq_len"],
            "stride": config["train"]["stride"],
        },
        "process": {
            "sde_type": config["process"]["sde_type"],
            "N": config["process"]["N"],
            "model_steps": config["process"]["model_steps"],
            "eps": config["process"].get("eps", 1e-3),
            "enforce_observed": config["process"].get("enforce_observed", True),
        },
        "diffusion": {
            "num_steps": config["diffusion"]["num_steps"],
            "diffusion_embedding_dim": config["diffusion"]["diffusion_embedding_dim"],
            "channels": config["diffusion"]["channels"],
            "layers": config["diffusion"]["layers"],
            "nheads": config["diffusion"]["nheads"],
            "is_linear": config["diffusion"]["is_linear"],
        },
        "model": {
            "timeemb": config["model"]["timeemb"],
            "featureemb": config["model"]["featureemb"],
            "is_unconditional": config["model"]["is_unconditional"],
        },
        "data": {
            "target_dim": config["data"]["target_dim"],
        },
    }

# For resuming training

def make_checkpoint_payload(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Dict[str, Any],
    epoch: int,
    global_step: int,
    history: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,                       # full config
        "run_metadata": build_run_metadata(config),  # quick-access summary
        "history": history,
    }


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Loads checkpoint and returns:
      start_epoch, global_step, history
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])

    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt["epoch"]) + 1
    global_step = int(ckpt.get("global_step", 0))
    history = ckpt.get("history", {"epoch_losses": [], "val_losses": []})

    return start_epoch, global_step, history

from datetime import datetime

# Final checkpoint name
def build_final_checkpoint_name(
    config: Dict[str, Any],
    global_step: int,
    timestamp: Optional[str] = None,
    actual_epoch: Optional[int] = None,
) -> str:
    """
    Build a descriptive final checkpoint filename.

    Requested fields:
      1. epochs
      2. global_step
      3. sde_type
      4. lr
      5. N
      6. is_linear -> 'linear' or 'notlinear'
      7. layers
      8. nheads
      + timestamp to avoid overwriting
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_root = str(config['train']['data_root'])
    if data_root == "./data/sp500_individual_gbm":
        data_root = "REAL"
    elif data_root == "./data/fake_individual_gbm":
        data_root = "FAKE"
    elif data_root == "./data/fake_individual_gbm_close":
        data_root = "FAKE_REPL"
    elif data_root == "./data/replication_returns":
        data_root = "REPL_RET_"
    elif data_root == "./data/replication_returns_norm":
        data_root = "REPL_RET_NORM_"
    else:
        data_root = "REPL_"
    mask_mode = str(config['train']['mask_mode'])
    if mask_mode == "random":
        mask_mode = "RAND"
    elif mask_mode == "unconditional" or bool(config['model']['is_unconditional']):
        mask_mode = "UNCO"
    else:
        mask_mode = "CLOS"
    
    epochs = actual_epoch if actual_epoch is not None else int(config["train"]["epochs"])
    sde_type = str(config["process"]["sde_type"])
    lr = float(config["train"]["lr"])
    N = int(config["process"]["N"])
    is_linear = "linear" if bool(config["diffusion"]["is_linear"]) else "notlinear"
    layers = int(config["diffusion"]["layers"])
    nheads = int(config["diffusion"]["nheads"])
    noise_schedule = str(config["process"].get("noise_schedule", "linear"))

    # safer string for learning rate, e.g. 1e-4 -> 1e-04
    lr_str = f"{lr:.0e}"

    lw_tag = "LW_" if bool(config["train"].get("likelihood_weighting", False)) else "NO_"

    filename = (
        f"{lw_tag}"
        f"{data_root}_"
        f"{mask_mode}_"
        f"final_"
        f"ep-{epochs}_"
        f"step-{global_step}_"
        f"sde-{sde_type}_"
        f"noise-{noise_schedule}_"
        f"lr-{lr_str}_"
        f"N-{N}_"
        f"{is_linear}_"
        f"layers-{layers}_"
        f"nheads-{nheads}_"
        f"{timestamp}.pt"
    )
    return filename

def train(
    config: Dict[str, Any],
    train_loader,
    val_loader=None,
    resume_checkpoint: Optional[str] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["train"]["seed"]))

    # 1) Instantiate diffusion processes
    processes = Diffusion_Processes(config["process"])

    # 2) Instantiate model
    target_dim = int(config["data"]["target_dim"])
    model = CSDIModel(target_dim=target_dim, config=config, device=device).to(device)

    # 3) Optimizer
    optim = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    # 4) AMP
    use_amp = bool(config["train"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5) Output dir
    out_dir = config["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    num_epochs = int(config["train"]["epochs"])
    # Total number of batches per epoch (for progress bar)
    steps_per_epoch = len(train_loader)
    ckpt_every_epochs = get_checkpoint_save_interval(num_epochs)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resume state
    start_epoch = 0
    global_step = 0
    history = {"epoch_losses": [], "val_losses": []}

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        start_epoch, global_step, history = load_checkpoint(
            resume_checkpoint, model, optim, scaler, device
        )
        print(f"Resumed at epoch={start_epoch}, global_step={global_step}")

    # W&B initialisation (optional — only when --wandb_project is provided)
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
        # resume the same W&B run if we loaded a checkpoint that already has one
        prior_run_id = history.get("wandb_run_id")
        if prior_run_id is not None:
            init_kwargs["id"] = prior_run_id
            init_kwargs["resume"] = "must"
        wandb.init(**init_kwargs)
        history["wandb_run_id"] = wandb.run.id

    # Early stopping state
    early_stop_patience = config["train"].get("early_stop_patience", None)
    if early_stop_patience is not None:
        early_stop_patience = int(early_stop_patience) if int(early_stop_patience) > 0 else None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()

        pbar = tqdm(
            enumerate(train_loader), #iterable to wrap
            total=steps_per_epoch, #how many steps to expect
            desc=f"epoch {epoch+1}/{num_epochs}", #label
            dynamic_ncols=True, #adapt the bar width
        )

        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        ema_loss = None
        ema_beta = float(config["train"].get("ema_beta", 0.98))
        use_lw = bool(config["train"].get("likelihood_weighting", False))
        t0 = time.time()

        for batch_idx, batch in pbar:
            observed_data, observed_mask, observed_tp = unpack_batch(batch, device)

            # conditioning mask
            mask_mode = config["train"].get("mask_mode", "random")
            if config["model"]["is_unconditional"] or mask_mode == "unconditional":
                cond_mask = torch.zeros_like(observed_mask)
            elif mask_mode == "predict_close":
                cond_mask = get_predict_close_mask(observed_mask)
            else:  # "random"
                cond_mask = get_randmask(
                    observed_mask,
                    min_ratio=float(config["train"]["cond_min_ratio"]),
                    max_ratio=float(config["train"]["cond_max_ratio"]),
                )

            target_mask = (observed_mask.float() * (1.0 - cond_mask.float())).float()

            # forward diffusion
            x_t, t_cont, eps, sigma_t = processes.forward_process(observed_data)

            # forward + loss
            with torch.amp.autocast("cuda", enabled=use_amp):
                eps_hat = model(
                    x_t=x_t,
                    t=t_cont,
                    observed_data=observed_data,
                    cond_mask=cond_mask,
                    observed_tp=observed_tp,
                )
                if use_lw:
                    loss = masked_mse(eps_hat, eps, target_mask, sigma_t=sigma_t)
                else:
                    loss = masked_mse(eps_hat, eps, target_mask, None)

            loss_val = float(loss.detach().item())
            epoch_loss_sum += loss_val
            epoch_loss_count += 1

            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * loss_val

            # optimize
            optim.zero_grad(set_to_none=True)

            # DEBUGGING: zero-initalizied output is a local trap
            # w = model.diffmodel.output_projection2.weight
            # print(f"output_proj2 weight norm: {w.norm().item():.6f}")   # should be > 0 after step 1
            # print(f"output_proj2 weight grad: {w.grad.norm().item() if w.grad is not None else 'None'}")
            # END DEBUGGING

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            # DEBUGGING: AMP scaler silently skipping every update
            # scale = scaler.get_scale()
            # print(f"AMP scale: {scale}")   # if this keeps halving, NaN gradients are occurring
            # END DEBUGGING

            global_step += 1

            elapsed = time.time() - t0
            steps_done = batch_idx + 1
            it_per_s = steps_done / max(elapsed, 1e-9)

            pbar.set_postfix({
                "step": global_step,
                "loss": f"{loss_val:.4f}",
                "ema": f"{ema_loss:.4f}",
                "avg": f"{(epoch_loss_sum / epoch_loss_count):.4f}",
                "it/s": f"{it_per_s:.2f}",
                "t": f"{t_cont.float().mean().item():.1f}",
            })

            if global_step % int(config["train"]["log_every_steps"]) == 0:
                print(
                    f"epoch={epoch+1}/{num_epochs} "
                    f"step={global_step} "
                    f"loss={loss_val:.6f} "
                    f"t_cont_mean={t_cont.float().mean().item():.2f}"
                )
                if use_wandb:
                    wandb.log({
                        "train/step_loss": loss_val,
                        "train/ema_loss": ema_loss,
                        "train/avg_loss": epoch_loss_sum / epoch_loss_count,
                        "train/it_per_s": it_per_s,
                    }, step=global_step)

        # end of epoch
        epoch_avg = epoch_loss_sum / max(epoch_loss_count, 1)

        # validation
        val_avg = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    observed_data, observed_mask, observed_tp = unpack_batch(val_batch, device)

                    if config["model"]["is_unconditional"] or mask_mode == "unconditional":
                        cond_mask = torch.zeros_like(observed_mask)
                    elif mask_mode == "predict_close":
                        cond_mask = get_predict_close_mask(observed_mask)
                    else:  # "random"
                        cond_mask = get_randmask(
                            observed_mask,
                            min_ratio=float(config["train"]["cond_min_ratio"]),
                            max_ratio=float(config["train"]["cond_max_ratio"]),
                        )

                    target_mask = (observed_mask.float() * (1.0 - cond_mask.float())).float()

                    x_t, t_cont, eps, sigma_t = processes.forward_process(observed_data)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        eps_hat = model(
                            x_t=x_t,
                            t=t_cont,
                            observed_data=observed_data,
                            cond_mask=cond_mask,
                            observed_tp=observed_tp,
                        )
                        if use_lw:
                            val_loss = masked_mse(eps_hat, eps, target_mask, sigma_t=sigma_t)
                        else:
                            val_loss = masked_mse(eps_hat, eps, target_mask, None)

                    val_loss_sum += float(val_loss.item())
                    val_loss_count += 1

            val_avg = val_loss_sum / max(val_loss_count, 1)
            history["val_losses"].append({"epoch": epoch + 1, "avg_val_loss": val_avg})

        history["epoch_losses"].append({
            "epoch": epoch + 1,
            "avg_train_loss": epoch_avg,
            "avg_val_loss": val_avg,
        })

        if val_avg is not None:
            print(f"[epoch {epoch+1}/{num_epochs}] avg_train_loss={epoch_avg:.6f}  avg_val_loss={val_avg:.6f}")
        else:
            print(f"[epoch {epoch+1}/{num_epochs}] avg_train_loss={epoch_avg:.6f}")

        if use_wandb:
            epoch_metrics = {
                "epoch/train_loss": epoch_avg,
                "epoch/epoch": epoch + 1,
                "epoch/epoch_time_s": time.time() - t0,
            }
            if val_avg is not None:
                epoch_metrics["epoch/val_loss"] = val_avg
            wandb.log(epoch_metrics, step=global_step)

        # early stopping (only when val_loader is provided)
        if early_stop_patience is not None and val_avg is not None:
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                epochs_no_improve = 0
                best_ckpt_path = os.path.join(out_dir, f"best_{run_timestamp}.pt")
                torch.save(
                    make_checkpoint_payload(model, optim, scaler, config, epoch, global_step, history),
                    best_ckpt_path,
                )
                print(f"  [early-stop] New best val_loss={best_val_loss:.6f} — saved {best_ckpt_path}")
            else:
                epochs_no_improve += 1
                print(f"  [early-stop] No improvement for {epochs_no_improve}/{early_stop_patience} epochs")
                if epochs_no_improve >= early_stop_patience:
                    print(f"  [early-stop] Patience exhausted — stopping training at epoch {epoch+1}")
                    if use_wandb:
                        wandb.log({"early_stop_epoch": epoch + 1}, step=global_step)
                    break

        # save checkpoint by epoch cadence
        should_save = ((epoch + 1) % ckpt_every_epochs == 0) or ((epoch + 1) == num_epochs)
        if should_save:
            ckpt_path = os.path.join(out_dir, f"ckpt_epoch_{epoch+1}_{run_timestamp}.pt")
            payload = make_checkpoint_payload(
                model=model,
                optim=optim,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                history=history,
            )
            torch.save(payload, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # final save
    final_payload = {
    "model": model.state_dict(),
    "optim": optim.state_dict(),
    "scaler": scaler.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "config": config,
    "run_metadata": build_run_metadata(config),
    "history": history,
    "training_completed": (epoch + 1) == num_epochs,
    "early_stopped": (epoch + 1) < num_epochs,
    }

    final_filename = build_final_checkpoint_name(config=config, global_step=global_step, actual_epoch=epoch + 1)
    final_path = os.path.join(out_dir, final_filename)

    torch.save(final_payload, final_path)
    print(f"Training complete. Saved: {final_path}")

    if use_wandb:
        wandb.finish()

# MAIN

if __name__ == "__main__":
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
    )

    dataset = train_loader.dataset
    dataset_size = len(dataset)

    # Subset conditions
    subset_ratio = config["train"].get("train_subset_ratio", None)
    subset_size = config["train"].get("train_subset_size", None)

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

    # --- Single permutation guarantees train/val are disjoint ---
    rng = torch.Generator().manual_seed(int(config["train"]["seed"]))
    all_indices = torch.randperm(dataset_size, generator=rng).tolist()

    # Carve out validation first
    val_loader = None
    val_split_ratio = config["train"].get("val_split_ratio", None)
    if val_split_ratio is not None:
        if not (0 < val_split_ratio < 1):
            raise ValueError("val_split_ratio must be in (0, 1)")
        val_size = max(1, int(dataset_size * val_split_ratio))
        val_indices = all_indices[:val_size]
        train_pool = all_indices[val_size:]
        val_dataset = Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
        )
        print(f"Validation set: {val_size}/{dataset_size} samples")
    else:
        train_pool = all_indices

    # Then carve out the training subset from the remaining indices
    if subset_size is not None:
        subset_size = min(subset_size, len(train_pool))
        train_indices = train_pool[:subset_size]
        subset_dataset = Subset(dataset, train_indices)
        train_loader = DataLoader(
            subset_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=config["train"]["shuffle"],
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
            collate_fn=getattr(train_loader, "collate_fn", None),
        )
        print(f"Using subset of dataset: {subset_size}/{dataset_size} samples (excl. val)")
    else:
        if val_split_ratio is not None:
            # Rebuild train_loader excluding val indices
            train_dataset = Subset(dataset, train_pool)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["train"]["batch_size"],
                shuffle=config["train"]["shuffle"],
                num_workers=config["train"]["num_workers"],
                pin_memory=config["train"]["pin_memory"],
                collate_fn=getattr(train_loader, "collate_fn", None),
            )
        print(f"Using full training pool: {len(train_pool)}/{dataset_size} samples (excl. val)")

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
    )

    print("Training is finished")

# python csdi_train_modified.py --config configs/replication.yaml --epochs 1 --train_subset_ratio 0.005 --val_split_ratio 0.005 --data_root "./data/replication" --noise_schedule exponential --sigma_min 0.01 --sigma_max 1.0 --beta_min 0.01 --beta_max 20.0