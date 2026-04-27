import os
import json
import random
import argparse
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

import yaml
import wandb

# ----------------------------
# YAML helpers
# ----------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `updates` into `base`; values in `updates` win."""
    out = deepcopy(base)
    for key, value in updates.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


# ----------------------------
# CLI argument parsing
# ----------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="EDM training for CSDI financial time-series model")

    # config file
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # train overrides
    parser.add_argument("--epochs",           type=int,      default=None)
    parser.add_argument("--batch_size",       type=int,      default=None)
    parser.add_argument("--lr",               type=float,    default=None)
    parser.add_argument("--weight_decay",     type=float,    default=None)
    parser.add_argument("--use_amp",          type=str2bool, default=None)
    parser.add_argument("--log_every_steps",  type=int,      default=None)
    parser.add_argument("--data_root",        type=str,      default=None)
    parser.add_argument("--out_dir",          type=str,      default=None)
    parser.add_argument("--seq_len",          type=int,      default=None)
    parser.add_argument("--stride",           type=int,      default=None)
    parser.add_argument("--mask_mode",        type=str,      default=None,
                        choices=["random", "unconditional", "predict_close"])
    parser.add_argument("--debug",            type=str2bool, default=None)
    parser.add_argument("--lr_cosine_annealing", type=str2bool, default=None)
    parser.add_argument("--lr_eta_min",       type=float,    default=None)
    parser.add_argument("--train_subset_ratio", type=float,  default=None)
    parser.add_argument("--train_subset_size",  type=int,    default=None)
    parser.add_argument("--val_split_ratio",    type=float,  default=None)
    parser.add_argument("--early_stop_patience", type=int,   default=None)
    parser.add_argument("--loss_spike_factor",   type=float, default=None,
                        help="When loss > loss_spike_factor * EMA loss, print batch stats (requires --debug true)")

    # EDM loss overrides (map to config["edm"] and config["model"])
    parser.add_argument("--P_mean",     type=float, default=None,
                        help="Log-normal mean for EDM sigma sampling (default: -1.2)")
    parser.add_argument("--P_std",      type=float, default=None,
                        help="Log-normal std for EDM sigma sampling (default: 1.2)")
    parser.add_argument("--sigma_data", type=float, default=None,
                        help="Empirical data std for EDM preconditioning; must match model.sigma_data")

    # model overrides
    parser.add_argument("--is_unconditional",      type=str2bool, default=None)
    parser.add_argument("--timeemb",               type=int,      default=None)
    parser.add_argument("--featureemb",            type=int,      default=None)

    # diffusion backbone overrides
    parser.add_argument("--channels",              type=int,   default=None)
    parser.add_argument("--layers",                type=int,   default=None)
    parser.add_argument("--nheads",                type=int,   default=None)
    parser.add_argument("--diffusion_embedding_dim", type=int, default=None)

    # data
    parser.add_argument("--target_dim", type=int, default=None)

    # process overrides (saved to checkpoint config, used by generate_samples.py)
    parser.add_argument("--sde_type",       type=str,   default=None)
    parser.add_argument("--noise_schedule", type=str,   default=None,
                        choices=["linear", "exponential", "cosine"])
    parser.add_argument("--sigma_min",      type=float, default=None)
    parser.add_argument("--sigma_max",      type=float, default=None)
    parser.add_argument("--beta_min",       type=float, default=None)
    parser.add_argument("--beta_max",       type=float, default=None)
    parser.add_argument("--model_steps",    type=int,   default=None)
    parser.add_argument("--N",              type=int,   default=None)

    # resume
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    # wandb
    parser.add_argument("--wandb_project",  type=str, default=None)
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


def build_cli_override_dict(args) -> Dict[str, Any]:
    override: Dict[str, Any] = {
        "data": {}, "model": {}, "diffusion": {}, "process": {}, "train": {}, "edm": {}, "wandb": {}
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
    if args.sigma_data is not None:
        override["model"]["sigma_data"] = args.sigma_data

    # diffusion backbone
    if args.channels is not None:
        override["diffusion"]["channels"] = args.channels
    if args.layers is not None:
        override["diffusion"]["layers"] = args.layers
    if args.nheads is not None:
        override["diffusion"]["nheads"] = args.nheads
    if args.diffusion_embedding_dim is not None:
        override["diffusion"]["diffusion_embedding_dim"] = args.diffusion_embedding_dim

    # process (written to saved config for sampler use; not consumed during training)
    if args.sde_type is not None:
        override["process"]["sde_type"] = args.sde_type
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
    if args.model_steps is not None:
        override["process"]["model_steps"] = args.model_steps
    if args.N is not None:
        override["process"]["N"] = args.N

    # EDM loss
    if args.P_mean is not None:
        override["edm"]["P_mean"] = args.P_mean
    if args.P_std is not None:
        override["edm"]["P_std"] = args.P_std

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
    if args.log_every_steps is not None:
        override["train"]["log_every_steps"] = args.log_every_steps
    if args.data_root is not None:
        override["train"]["data_root"] = args.data_root
    if args.out_dir is not None:
        override["train"]["out_dir"] = args.out_dir
    if args.seq_len is not None:
        override["train"]["seq_len"] = args.seq_len
    if args.stride is not None:
        override["train"]["stride"] = args.stride
    if args.mask_mode is not None:
        override["train"]["mask_mode"] = args.mask_mode
    if args.debug is not None:
        override["train"]["debug"] = args.debug
    if args.lr_cosine_annealing is not None:
        override["train"]["lr_cosine_annealing"] = args.lr_cosine_annealing
    if args.lr_eta_min is not None:
        override["train"]["lr_eta_min"] = args.lr_eta_min
    if args.train_subset_ratio is not None:
        override["train"]["train_subset_ratio"] = args.train_subset_ratio
    if args.train_subset_size is not None:
        override["train"]["train_subset_size"] = args.train_subset_size
    if args.val_split_ratio is not None:
        override["train"]["val_split_ratio"] = args.val_split_ratio
    if args.early_stop_patience is not None:
        override["train"]["early_stop_patience"] = args.early_stop_patience
    if args.loss_spike_factor is not None:
        override["train"]["loss_spike_factor"] = args.loss_spike_factor

    # wandb
    if args.wandb_project is not None:
        override["wandb"]["project"] = args.wandb_project
    if args.wandb_entity is not None:
        override["wandb"]["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        override["wandb"]["run_name"] = args.wandb_run_name

    return {k: v for k, v in override.items() if v}


def get_final_config() -> Tuple[Dict[str, Any], Any]:
    args = parse_args()
    yaml_config = load_yaml_config(args.config)
    cli_override = build_cli_override_dict(args)
    return deep_update(yaml_config, cli_override), args


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Conditioning mask helpers
# ----------------------------

def get_randmask(observed_mask: torch.Tensor, min_ratio: float = 0.1, max_ratio: float = 0.9) -> torch.Tensor:
    """Randomly hide min_ratio–max_ratio of observed entries as targets."""
    B = observed_mask.shape[0]
    miss_ratio = torch.empty(B, device=observed_mask.device).uniform_(min_ratio, max_ratio)
    rand = torch.rand_like(observed_mask.float())
    keep = (rand > miss_ratio.view(B, 1, 1)).float()
    return (observed_mask.float() * keep).float()


def get_predict_close_mask(observed_mask: torch.Tensor, close_idx: int = 3) -> torch.Tensor:
    """Condition on Open/High/Low; always predict Close (feature index close_idx)."""
    cond_mask = observed_mask.clone().float()
    cond_mask[:, close_idx, :] = 0.0
    return cond_mask


# ----------------------------
# Batch unpacking
# ----------------------------

def unpack_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Accept dict or tuple batches from the dataloader.
    Returns observed_data (B,K,L), observed_mask (B,K,L), observed_tp (B,L) on device.
    """
    if isinstance(batch, dict):
        observed_data = batch["observed_data"]
        observed_mask = batch["observed_mask"]
        observed_tp   = batch["observed_tp"]
    else:
        observed_data, observed_mask, observed_tp = batch
    return observed_data.to(device), observed_mask.to(device), observed_tp.to(device)


# ----------------------------
# Checkpoint helpers
# ----------------------------

def get_checkpoint_save_interval(num_epochs: int) -> int:
    """Save every 5 epochs when total < 10, otherwise every 10% of total epochs."""
    if num_epochs < 10:
        return 5
    return max(1, num_epochs // 10)


def build_run_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Quick-access summary of the most important config values stored in each checkpoint."""
    edm_cfg = config.get("edm", {})
    return {
        "train": {
            "seed":           config["train"]["seed"],
            "epochs":         config["train"]["epochs"],
            "batch_size":     config["train"]["batch_size"],
            "lr":             config["train"]["lr"],
            "weight_decay":   config["train"]["weight_decay"],
            "use_amp":        config["train"]["use_amp"],
            "cond_min_ratio": config["train"]["cond_min_ratio"],
            "cond_max_ratio": config["train"]["cond_max_ratio"],
            "mask_mode":      config["train"].get("mask_mode", "random"),
            "seq_len":        config["train"]["seq_len"],
            "stride":         config["train"]["stride"],
        },
        "edm": {
            "P_mean":     edm_cfg.get("P_mean",    -1.2),
            "P_std":      edm_cfg.get("P_std",      1.2),
            "sigma_data": config["model"].get("sigma_data", 1.0),
        },
        "diffusion": {
            "num_steps":             config["diffusion"]["num_steps"],
            "diffusion_embedding_dim": config["diffusion"]["diffusion_embedding_dim"],
            "channels":              config["diffusion"]["channels"],
            "layers":                config["diffusion"]["layers"],
            "nheads":                config["diffusion"]["nheads"],
            "is_linear":             config["diffusion"]["is_linear"],
        },
        "model": {
            "timeemb":        config["model"]["timeemb"],
            "featureemb":     config["model"]["featureemb"],
            "is_unconditional": config["model"]["is_unconditional"],
            "sigma_data":     config["model"].get("sigma_data", 1.0),
        },
        "data": {
            "target_dim": config["data"]["target_dim"],
        },
    }


def make_checkpoint_payload(
    model:        nn.Module,
    optim:        torch.optim.Optimizer,
    scaler:       torch.amp.GradScaler,
    config:       Dict[str, Any],
    epoch:        int,
    global_step:  int,
    history:      Dict[str, Any],
    scheduler=None,
) -> Dict[str, Any]:
    payload = {
        "epoch":        epoch,
        "global_step":  global_step,
        "model":        model.state_dict(),
        "optim":        optim.state_dict(),
        "scaler":       scaler.state_dict(),
        "config":       config,
        "run_metadata": build_run_metadata(config),
        "history":      history,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def load_checkpoint(
    ckpt_path: str,
    model:     nn.Module,
    optim:     torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    device:    torch.device,
    scheduler=None,
) -> Tuple[int, int, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt["epoch"]) + 1
    global_step = int(ckpt.get("global_step", 0))
    history     = ckpt.get("history", {"epoch_losses": [], "val_losses": []})
    return start_epoch, global_step, history


def build_final_checkpoint_name(
    config:       Dict[str, Any],
    global_step:  int,
    timestamp:    Optional[str] = None,
    actual_epoch: Optional[int] = None,
) -> str:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Abbreviated data-root tag
    _root_map = {
        "./data/sp500_individual_gbm":     "REAL",
        "./data/fake_individual_gbm":      "FAKE",
        "./data/fake_individual_gbm_close":"FAKE_REPL",
        "./data/replication_returns":      "REPL_RET",
        "./data/replication_returns_norm": "REPL_RET_NORM",
        "./data/toy_gaussian":             "TOY_GAUSS",
        "./data/toy_ar1":                  "TOY_AR1",
        "./data/toy_studentt":             "TOY_STUDENTT",
        "./data/toy_gbm":                  "TOY_GBM",
        "./data/toy_gbm_norm":             "TOY_GBM_NORM",
        "./data/fake_fts":                 "FAKE_FTS",
        "./data/fake_fts_processed":       "FAKE_FTS_PROC",
        "./data/replication_processed":    "REPL_PROC",
    }
    data_root = _root_map.get(str(config["train"]["data_root"]), "REPL_")

    mask_mode = str(config["train"]["mask_mode"])
    if mask_mode == "random":
        mask_tag = "RAND"
    elif mask_mode == "unconditional" or bool(config["model"]["is_unconditional"]):
        mask_tag = "UNCO"
    else:
        mask_tag = "CLOS"

    epochs   = actual_epoch if actual_epoch is not None else int(config["train"]["epochs"])
    lr       = float(config["train"]["lr"])
    channels = int(config["diffusion"]["channels"])
    layers   = int(config["diffusion"]["layers"])
    nheads   = int(config["diffusion"]["nheads"])
    emb_dim  = int(config["diffusion"]["diffusion_embedding_dim"])
    sd       = float(config["model"].get("sigma_data", 1.0))
    P_mean   = float(config.get("edm", {}).get("P_mean", -1.2))

    return (
        f"EDM_"
        f"{data_root}_"
        f"{mask_tag}_"
        f"ep-{epochs}_"
        f"step-{global_step}_"
        f"lr-{lr:.0e}_"
        f"ch-{channels}_"
        f"layers-{layers}_"
        f"nheads-{nheads}_"
        f"diffemb-{emb_dim}_"
        f"sd-{sd:.1f}_"
        f"Pm-{P_mean:.1f}_"
        f"{timestamp}.pt"
    )
