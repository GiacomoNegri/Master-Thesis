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
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed (overrides config train.seed)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--use_amp", type=str2bool, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)

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

    # diffusion overrides
    parser.add_argument("--channels", type=int, default=None,
                        help="Number of residual channels in the diffusion model (overrides config)")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of residual layers in the diffusion model (overrides config)")
    parser.add_argument("--nheads", type=int, default=None,
                        help="Number of attention heads in the diffusion model (overrides config)")
    parser.add_argument("--diffusion_embedding_dim", type=int, default=None,
                        help="Dimension of the diffusion embedding (overrides config)")

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
                        help="If true, apply Song's likelihood weighting λ(t)=g(t)²/σ²(t) to the MSE loss. If false, use plain MSE.")
    parser.add_argument("--normalization", type=str2bool, default=None,
                        help="If true, z-score normalize each window per feature before training (mean=0, std=1 per (K,) channel across the L time steps).")
    parser.add_argument("--debug", type=str2bool, default=None,
                        help="If true, print per-batch sigma / err / eps diagnostics. If false, only loss is printed.")
    parser.add_argument("--loss_spike_factor", type=float, default=None,
                        help="Print a line whenever loss > factor × ema_loss (e.g. 5.0). null/omit to disable.")
    parser.add_argument("--print_plots", type=str2bool, default=None,
                        help="If true, collect diagnostic statistics and save 11 PNG plots "
                             "to <out_dir>/diagnostics/epoch_NNN/ every --plots_every epochs. "
                             "Useful for preliminary model analysis. Default false (controlled by config).")
    parser.add_argument("--plots_every", type=int, default=None,
                        help="Generate diagnostic plots every N epochs (only active when --print_plots true). "
                             "Default 1 — i.e. every epoch.")
    parser.add_argument("--importance_sampling", type=str2bool, default=None,
                        help="If true, sample t ~ p(t) ∝ g(t)²/σ²(t) instead of uniform. "
                             "Concentrates training on hard small-t timesteps; disables likelihood weighting.")
    parser.add_argument("--lr_cosine_annealing", type=str2bool, default=None,
                        help="[Legacy] If true, equivalent to --lr_scheduler cosine-annealing. "
                             "Ignored when --lr_scheduler is set explicitly.")
    parser.add_argument("--lr_eta_min", type=float, default=None,
                        help="Minimum LR at the end of the cosine annealing cycle (default 1e-6).")
    parser.add_argument("--lr_scheduler",
                        type=str,
                        default=None,
                        choices=["cosine-annealing", "reduce-on-plateau", "none"],
                        help="LR scheduler. Takes precedence over --lr_cosine_annealing. "
                             "'cosine-annealing': decay from lr to lr_eta_min over the full run; "
                             "'reduce-on-plateau': multiply lr by lr_gamma when val loss stalls "
                             "(requires val_split_ratio and lr_patience); "
                             "'none': constant LR.")
    parser.add_argument("--lr_patience", type=int, default=None,
                        help="Patience epochs for reduce-on-plateau: reduce LR after this many "
                             "epochs with no val-loss improvement. Only used when "
                             "--lr_scheduler=reduce-on-plateau.")
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
        "diffusion": {},
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


    # diffusion
    if args.channels is not None:
        override["diffusion"]["channels"] = args.channels
    if args.layers is not None:
        override["diffusion"]["layers"] = args.layers
    if args.nheads is not None:
        override["diffusion"]["nheads"] = args.nheads
    if args.diffusion_embedding_dim is not None:
        override["diffusion"]["diffusion_embedding_dim"] = args.diffusion_embedding_dim

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
    if args.seed is not None:
        override["train"]["seed"] = args.seed
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
    if args.normalization is not None:
        override["train"]["normalization"] = args.normalization
    if args.debug is not None:
        override["train"]["debug"] = args.debug
    if args.loss_spike_factor is not None:
        override["train"]["loss_spike_factor"] = args.loss_spike_factor
    if args.print_plots is not None:
        override["train"]["print_plots"] = args.print_plots
    if args.plots_every is not None:
        override["train"]["plots_every"] = args.plots_every
    if args.importance_sampling is not None:
        override["train"]["importance_sampling"] = args.importance_sampling
    if args.lr_cosine_annealing is not None:
        override["train"]["lr_cosine_annealing"] = args.lr_cosine_annealing
    if args.lr_eta_min is not None:
        override["train"]["lr_eta_min"] = args.lr_eta_min
    if args.lr_scheduler is not None:
        override["train"]["lr_scheduler"] = args.lr_scheduler
    if args.lr_patience is not None:
        override["train"]["lr_patience"] = args.lr_patience
    if args.seq_len is not None:
        override["train"]["seq_len"] = args.seq_len
    if args.stride is not None:
        override["train"]["stride"] = args.stride

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
    a fraction of the observed entries.F

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


def zscore_normalize_windows(observed_data: torch.Tensor) -> torch.Tensor:
    """
    Z-score normalize each window independently per feature channel.

    observed_data: (B, K, L)
    Returns:       (B, K, L)  with mean≈0, std≈1 along the L dimension for each (b, k).
    """
    mu  = observed_data.mean(dim=2, keepdim=True)          # (B, K, 1)
    std = observed_data.std(dim=2, keepdim=True).clamp(min=1e-8)  # (B, K, 1)
    return (observed_data - mu) / std


# ----------------------------
# Loss: MSE on target (non-conditioned) entries
# ----------------------------
def masked_mse(eps_hat, eps, target_mask, sigma_t=None, g_t=None, debug=False):
    """
    sigma_t: optional (B,) tensor of per-sample marginal std σ(t) from the forward process.
    g_t:     optional (B,) tensor of per-sample diffusion coefficient g(t).
             When both are provided, applies Song's likelihood weighting
             λ(t) = g(t)² / σ(t)², which is the theoretically correct per-timestep weight
             for matching the log-likelihood objective. When only sigma_t is provided,
             falls back to 1/σ²(t). When neither, plain unweighted MSE is used.
    """
    sq_err = (eps_hat - eps) ** 2  # (B, K, L)

    if sigma_t is not None:
        # Song's likelihood weighting: λ(t) = g(t)^2 / σ(t)^2
        if g_t is not None:
            weight = (g_t.float().pow(2) / sigma_t.float().pow(2).clamp(min=1e-8))[:, None, None]
        # else:
        #     weight = (1.0 / sigma_t.float().pow(2).clamp(min=1e-8))[:, None, None]
            # TRYING TO ADD WEIGHT CLAMPING in order to address LW effects
            # weight = weight.clamp(max=25.0)  # min-SNR-5: cap at SNR=5 → max_weight ≈ 1/σ_threshold
        denom = (target_mask.float() * weight).sum().clamp(min=1e-10)
        loss = (sq_err * weight * target_mask).sum() / denom
    else:
        denom = target_mask.sum().clamp(min=1.0)
        loss = (sq_err * target_mask).sum() / denom

    if debug:
        masked = sq_err[target_mask.bool()]  # unweighted, for interpretable diagnostics
        print(f"err | mean={masked.mean():.4f}  std={masked.std():.4f}  "
              f"p50={masked.median():.4f}  p95={masked.quantile(0.95):.4f}  "
              f"p99={masked.quantile(0.99):.4f}  max={masked.max():.4f}")

        if sigma_t is not None:
            w_masked = (sq_err * weight)[target_mask.bool()]
            print(f"err(LW) | mean={w_masked.mean():.4f}  std={w_masked.std():.4f}  "
                  f"p50={w_masked.median():.4f}  p95={w_masked.quantile(0.95):.4f}  "
                  f"p99={w_masked.quantile(0.99):.4f}  max={w_masked.max():.4f}")

    return loss


# ----------------------------
# Diagnostic collection & plotting
# (HPC-safe — uses Agg backend, saves PNGs, no display needed)
# Activated only when print_plots=true; zero overhead otherwise.
# ----------------------------

class DiagnosticsCollector:
    """
    Accumulates per-batch statistics over one epoch for diagnostic plots.
    Every tensor is moved to CPU numpy immediately after each batch so GPU
    memory is not held across the epoch.
    """

    def __init__(self, delta_t: float = 0.0):
        # Δt used to compute reverse-step error metrics (pass (T - eps) / N)
        self.delta_t = delta_t
        # per-sample arrays (B,) per batch — concatenated at epoch end
        self.t_batch:           list = []  # sampled continuous t per sample
        self.err_batch:         list = []  # mean |ε̂ − ε| per sample (over target mask)
        self.rel_err_batch:     list = []  # mean |ε̂−ε|/|ε| per sample
        self.sigma_batch:       list = []  # σ(t) per sample
        self.g_batch:           list = []  # g(t) per sample (empty when not available)
        self.score_batch:       list = []  # mean −ε̂/σ(t) per sample (inferred score)
        self.snr_batch:         list = []  # 1/σ²(t) per sample — higher = easier timestep
        self.err_rev_batch:     list = []  # per-element reverse-step drift error (B,)
        self.rel_stoch_batch:   list = []  # dimensionless ratio: drift err / diffusion noise (B,)
        self.x0_mse_batch:      list = []  # per-sample mean ||hat_x0 - x0||^2 over target mask (B,)
        # raw signed values from target-masked entries (subsampled per batch)
        self.eps_hat_vals:   list = []  # ε̂  values — for histogram and scatter
        self.eps_vals:       list = []  # ε   values — for scatter (paired with eps_hat_vals)
        # per-step scalars (one value per optimisation step / batch)
        self.grad_norms:     list = []  # gradient norm after clipping
        self.batch_mean_t:   list = []  # mean t of the batch (for scatter vs grad norm)
        self.batch_mean_err: list = []  # mean error of the batch (for correlation with grad)

    def add(
        self,
        t_cont:       torch.Tensor,                   # (B,)
        eps_hat:      torch.Tensor,                   # (B, K, L) — detached float32
        eps:          torch.Tensor,                   # (B, K, L)
        sigma_t:      torch.Tensor,                   # (B,)
        target_mask:  torch.Tensor,                   # (B, K, L) in {0, 1}
        grad_norm:    float,
        g_t:          Optional[torch.Tensor] = None,  # (B,) or None
        mean_coeff_t: Optional[torch.Tensor] = None,  # (B,) or None; 1.0 for VE/GBM SDEs
    ) -> None:
        with torch.no_grad():
            # the detachment is needed for functioning under the use of AMP
            eps_hat_f = eps_hat.float().detach()
            eps_f     = eps.float().detach()
            sigma_f   = sigma_t.float().detach()
            mask_f    = target_mask.float().detach()
            t_f       = t_cont.float().detach()

            # number of target entries per sample (guard against div-by-zero)
            n = mask_f.sum(dim=(1, 2)).clamp(min=1.0)   # (B,)

            # absolute error — zeroed outside target mask
            diff         = (eps_hat_f - eps_f) * mask_f  # (B, K, L)
            abs_diff     = diff.abs()
            mean_abs_err = abs_diff.sum(dim=(1, 2)) / n  # (B,)

            # relative error:  |Δ| / (|ε| + 1e-8)
            rel_diff     = abs_diff / (eps_f.abs() + 1e-8) * mask_f
            mean_rel_err = rel_diff.sum(dim=(1, 2)) / n  # (B,)

            # inferred score = −ε̂ / σ(t)
            score_map  = (-eps_hat_f / sigma_f[:, None, None].clamp(min=1e-8)) * mask_f
            mean_score = score_map.sum(dim=(1, 2)) / n   # (B,)

            # the following have dimensions BN
            self.t_batch.append(t_f.cpu().numpy())
            self.err_batch.append(mean_abs_err.cpu().numpy())
            self.rel_err_batch.append(mean_rel_err.cpu().numpy())
            self.sigma_batch.append(sigma_f.cpu().numpy())
            self.score_batch.append(mean_score.cpu().numpy())

            # SNR = 1 / σ²(t): high SNR = small noise = easy for data structure, hard for score
            snr = 1.0 / (sigma_f.pow(2) + 1e-8)   # (B,)
            self.snr_batch.append(snr.cpu().numpy())

            # Reverse-step drift error and relative stochastic error.
            # Both are computable only when g(t) is available (collect_diag=True ensures it).
            # err_rev  = g²/σ · mean_abs_err · Δt
            #            = per-element RMS of the drift correction error per reverse step.
            # R        = err_rev / (g · √Δt)
            #            = dimensionless ratio of drift error to Brownian diffusion per step.
            #            R ≪ 1 → score errors dominated by noise (sampling is safe).
            #            R ≥ 1 → score errors comparable to diffusion (sampling may diverge).
            if g_t is not None and self.delta_t > 0:
                g_f   = g_t.float().detach()             # (B,)
                dt    = self.delta_t
                # err_rev (B,): g²/σ · mean_abs_err · Δt
                err_rev = (g_f.pow(2) / sigma_f.clamp(min=1e-8)) * mean_abs_err * dt
                # R (B,): err_rev / (g · √Δt)
                rel_stoch = err_rev / (g_f.clamp(min=1e-8) * (dt ** 0.5))
                self.err_rev_batch.append(err_rev.cpu().numpy())
                self.rel_stoch_batch.append(rel_stoch.cpu().numpy())

            # x0 reconstruction MSE: ||hat_x0 - x0||^2 per sample (over target mask).
            # Algebraic identity: hat_x0 - x0 = (σ / mean_coeff) * (ε - ε̂),
            # so MSE_x0 = (σ / mean_coeff)^2 * mean sq-err of ε.
            # For VE/GBM mean_coeff = 1 (identity); for VP/SubVP pass the actual coeff.
            # When mean_coeff_t is None we fall back to mean_coeff = 1.
            if mean_coeff_t is not None:
                mc_f = mean_coeff_t.float().detach().clamp(min=1e-8)  # (B,)
            else:
                mc_f = torch.ones_like(sigma_f)
            scale_sq = (sigma_f / mc_f).pow(2)  # (B,)
            x0_sq_err = (eps_hat_f - eps_f).pow(2) * mask_f  # (B, K, L)
            mean_x0_mse = (x0_sq_err * scale_sq[:, None, None]).sum(dim=(1, 2)) / n  # (B,)
            self.x0_mse_batch.append(mean_x0_mse.cpu().numpy())

            # raw signed values inside the target mask — subsampled to 300 per batch
            # so memory stays bounded regardless of B, K, L.  Paired so scatter is valid.
            flat_hat = eps_hat_f[mask_f.bool()].cpu().numpy()
            flat_eps = eps_f[mask_f.bool()].cpu().numpy()
            _max_per_batch = 300
            if len(flat_hat) > _max_per_batch:
                idx = np.random.choice(len(flat_hat), _max_per_batch, replace=False)
                flat_hat = flat_hat[idx]
                flat_eps = flat_eps[idx]
            self.eps_hat_vals.append(flat_hat)
            self.eps_vals.append(flat_eps)

            # the following three has length N
            self.grad_norms.append(float(grad_norm))
            self.batch_mean_t.append(float(t_f.mean()))
            self.batch_mean_err.append(float(mean_abs_err.mean()))

            if g_t is not None:
                self.g_batch.append(g_t.float().detach().cpu().numpy())

    def arrays(self) -> dict:
        """Return flat numpy arrays for every accumulated quantity."""
        cat = lambda lst: np.concatenate(lst) if lst else np.array([])
        return {
            "t":        cat(self.t_batch),
            "err":      cat(self.err_batch),
            "rel_err":  cat(self.rel_err_batch),
            "sigma":    cat(self.sigma_batch),
            "g":        cat(self.g_batch),
            "score":    cat(self.score_batch),
            "snr":          cat(self.snr_batch),
            "err_rev":      cat(self.err_rev_batch),
            "rel_stoch":    cat(self.rel_stoch_batch),
            "x0_mse":       cat(self.x0_mse_batch),
            "eps_hat_vals": cat(self.eps_hat_vals),
            "eps_vals":     cat(self.eps_vals),
            "grad":     np.array(self.grad_norms),
            "t_mean":   np.array(self.batch_mean_t),
            "err_mean": np.array(self.batch_mean_err),
        }


def error_by_t_bins(
    t:      "np.ndarray",
    err:    "np.ndarray",
    n_bins: int   = 10,
    t_max:  float = 1.0,
) -> tuple:
    """
    Compute mean absolute error per uniform t-bin over [0, t_max].

    Parameters
    ----------
    t       : (N,) array of continuous time values
    err     : (N,) array of per-sample mean absolute errors
    n_bins  : number of equal-width bins
    t_max   : upper edge of the last bin (set to t.max() when calling)

    Returns
    -------
    edges     : (n_bins+1,) array of bin boundaries
    bin_means : (n_bins,)   array of per-bin mean error (NaN for empty bins)
    """
    edges = np.linspace(0.0, t_max, n_bins + 1)
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (t >= lo) & (t < hi)
        means.append(float(err[in_bin].mean()) if in_bin.any() else float("nan"))
    return edges, np.array(means)


def plot_and_save_diagnostics(
    collector:     "DiagnosticsCollector",
    epoch:         int,
    diag_base_dir: str,
    use_wandb:     bool = False,
    global_step:   int  = 0,
    sde_type:      str  = "ve",
) -> None:
    """
    Produce all 14 diagnostic plots for one epoch and save them as PNGs.

    Each figure is closed immediately after saving — no display is needed
    (uses the Agg backend, safe on headless HPC nodes).

    Output: <diag_base_dir>/epoch_<NNN>/01_hist_t.png … 14_scatter_err_vs_snr.png
            where diag_base_dir = ./images/diagnostics/<run_name>
    If use_wandb=True the images are also uploaded to the active W&B run.

    Plots generated
    ---------------
    01  Histogram of sampled continuous t
    02  Histogram of per-sample mean abs error
    03  [COMMENTED OUT] relative error vs t — inflated by near-zero ε; 03b is cleaner
    03b Scatter  absolute error vs noise-level axis (log σ or log-SNR)
    04  Scatter  batch mean error vs grad norm  + Pearson r printed to log
    05  Histogram of gradient norms (post-clipping, per step)
    06  Scatter  gradient norm vs batch mean t
    07  Histogram of σ(t)
    08  [COMMENTED OUT] error vs σ(t) — monotone in t, duplicates transformed 03b
    09  Histogram of g(t)
    10  Scatter  g(t) vs σ(t)  (per sample)
    11  Histogram of inferred score  −ε̂ / σ(t)  (per-sample mean)
    12  Bar chart  mean abs error per uniform t-bin  (binned error summary)
    13  [COMMENTED OUT] histogram of SNR — monotone transform of t, duplicates plot 01
    14  [COMMENTED OUT] error vs SNR — same as transformed 03b (log-SNR axis), redundant
    15  Scatter  ε̂ vs ε  (signed, target-masked, subsampled) — alignment check
    16  Histogram of ε̂  (signed) — distribution check vs N(0,1)
    17  Scatter  reverse-step drift error  err_rev = g²/σ · mean_abs_err · Δt  vs noise-level axis
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive — mandatory on headless nodes
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnostics] matplotlib not found — skipping plots (pip install matplotlib)")
        return

    d = collector.arrays()
    t, err, rel_err        = d["t"],     d["err"],    d["rel_err"]
    sigma, g, score        = d["sigma"], d["g"],      d["score"]
    snr                    = d["snr"]
    err_rev                = d["err_rev"]
    rel_stoch              = d["rel_stoch"]
    x0_mse                 = d["x0_mse"]
    eps_hat_vals           = d["eps_hat_vals"]
    eps_vals               = d["eps_vals"]
    grad, t_mean, err_mean = d["grad"],  d["t_mean"], d["err_mean"]

    # pre-compute binned error — used for plot 12 and optional W&B table
    t_max_val = float(t.max()) if len(t) > 0 else 1.0
    bin_edges, bin_means = error_by_t_bins(t, err, n_bins=10, t_max=t_max_val)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Interpretable x-axis for per-sample scatter plots (03b, 17).
    # log σ(t) / log-SNR are more informative than raw t because they directly
    # reflect the noise level rather than an abstract time index.
    if sde_type in ("ve", "gbm"):
        x_interp = np.log(sigma + 1e-8)    # log σ(t): ranges from log σ_min to log σ_max
        x_label  = "log σ(t)"
    else:                                   # vp, subvp
        x_interp = np.log(snr + 1e-8)      # log(1/σ²) ≡ log-SNR
        x_label  = "log-SNR  [log(1/σ²(t))]"

    diag_dir = os.path.join(diag_base_dir, f"epoch_{epoch:03d}")
    os.makedirs(diag_dir, exist_ok=True)

    saved: dict = {}   # name → file path, used for W&B upload

    # ---- tiny layout helpers ----------------------------------------
    def _save(fig, name: str) -> None:
        path = os.path.join(diag_dir, f"{name}.png")
        os.makedirs(diag_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        saved[name] = path

    def _hist(ax, data, xlabel: str, color: str = "steelblue", bins: int = 50) -> None:
        ax.hist(data, bins=bins, color=color, edgecolor="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")

    def _scatter(ax, x, y, xlabel: str, ylabel: str,
                 color: str = "steelblue", s: int = 3, alpha: float = 0.3) -> None:
        ax.scatter(x, y, s=s, alpha=alpha, color=color, rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ep = f"Epoch {epoch}"

    # ---- 1. Histogram of sampled continuous t -----------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    _hist(ax, t, "t  (continuous)", color="steelblue")
    ax.set_title(f"{ep} — Sampled t distribution")
    _save(fig, f"{epoch}_01_hist_t")

    # ---- 2. Histogram of abs errors (per-sample mean) ---------------
    fig, ax = plt.subplots(figsize=(6, 4))
    _hist(ax, err, "mean |ε̂ − ε|  per sample", color="darkorange")
    ax.set_title(f"{ep} — Abs-error distribution")
    _save(fig, f"{epoch}_02_hist_err")

    # ---- 3. Scatter: relative error vs t ----------------------------
    # REDUNDANT: relative error is dominated by near-zero ε values, making it
    # misleading.  Plot 03b (absolute error) conveys the same signal more cleanly.
    # fig, ax = plt.subplots(figsize=(6, 4))
    # _scatter(ax, t, rel_err, "t  (continuous)", "relative error  |ε̂−ε| / |ε|", color="purple")
    # ax.set_title(f"{ep} — Relative error vs t")
    # _save(fig, f"{epoch}_03_scatter_rel_err_vs_t")

    # ---- 3b. Scatter: absolute error vs noise-level axis ------------
    if len(x_interp) == len(err):
        fig, ax = plt.subplots(figsize=(6, 4))
        _scatter(ax, x_interp, err, x_label, "mean |ε̂ − ε|  per sample", color="darkorange")
        ax.set_title(f"{ep} — Absolute error vs {x_label}")
        _save(fig, f"{epoch}_03b_scatter_abs_err_vs_noise_level")

    # ---- 4. Batch error vs grad norm: Pearson r + scatter -----------
    # if len(grad) > 1 and len(err_mean) == len(grad):
    #     corr = float("nan")
    #     if np.std(err_mean) > 1e-12 and np.std(grad) > 1e-12:
    #         corr = float(np.corrcoef(err_mean, grad)[0, 1])
    #     print(f"  [diagnostics] Pearson(batch_mean_err, grad_norm) = {corr:.4f}")
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _scatter(ax, err_mean, grad, "batch mean |ε̂−ε|", "gradient norm (post-clip)",
    #              color="crimson", s=12, alpha=0.6)
    #     ax.set_title(f"{ep} — Batch error vs gradient  (r = {corr:.3f})")
    #     _save(fig, f"{epoch}_04_scatter_err_vs_grad")
    # else:
    #     print("  [diagnostics] too few batches for error–gradient correlation")

    # ---- 5. Histogram of gradient norms (per step) ------------------
    # fig, ax = plt.subplots(figsize=(6, 4))
    # _hist(ax, grad, "gradient norm (after clipping)", color="forestgreen")
    # ax.set_title(f"{ep} — Gradient norm distribution")
    # _save(fig,f"{epoch}_05_hist_grad")

    # ---- 6. Scatter: gradient norm vs batch mean t ------------------
    # if len(grad) == len(t_mean):
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _scatter(ax, t_mean, grad, "batch mean t", "gradient norm",
    #              color="teal", s=12, alpha=0.6)
    #     ax.set_title(f"{ep} — Gradient norm vs t")
    #     _save(fig, f"{epoch}_06_scatter_grad_vs_t")

    # # ---- 7. Histogram of σ(t) ---------------------------------------
    # if len(sigma) > 0:
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _hist(ax, sigma, "σ(t)", color="navy")
    #     ax.set_title(f"{ep} — σ(t) distribution")
    #     _save(fig, f"{epoch}_07_hist_sigma")

    # ---- 8. Scatter: error vs σ(t) ----------------------------------
    # REDUNDANT: σ(t) is a monotone function of t, so this duplicates plot 03b
    # (which now uses log σ / log-SNR directly).  Kept for reference only.
    # if len(sigma) == len(err):
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _scatter(ax, sigma, err, "σ(t)", "mean |ε̂ − ε|  per sample", color="chocolate")
    #     ax.set_title(f"{ep} — Error vs σ(t)")
    #     _save(fig, f"{epoch}_08_scatter_err_vs_sigma")

    # ---- 9. Histogram of g(t) ---------------------------------------
    # if len(g) > 0:
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _hist(ax, g, "g(t)", color="darkred")
    #     ax.set_title(f"{ep} — g(t) distribution")
    #     _save(fig, f"{epoch}_09_hist_g")

    # # ---- 10. Scatter: g(t) vs σ(t) ----------------------------------
    # if len(g) > 0 and len(sigma) == len(g):
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _scatter(ax, sigma, g, "σ(t)", "g(t)", color="indigo")
    #     ax.set_title(f"{ep} — g(t) vs σ(t)")
    #     _save(fig, f"{epoch}_10_scatter_g_vs_sigma")

    # ---- 11. Histogram of inferred score = −ε̂ / σ(t) ---------------
    if len(score) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        _hist(ax, score, "score = −ε̂ / σ(t)  (per-sample mean)", color="slategray", bins=60)
        ax.set_title(f"{ep} — Inferred score distribution")
        _save(fig, f"{epoch}_11_hist_score")

    # ---- 12. Bar chart: mean abs error per uniform t-bin -------------
    # Gives a numerical summary of where in [0, T] the model is failing,
    # complementing the scatter in 03b.
    if len(t) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_width = (bin_edges[1] - bin_edges[0]) * 0.85
        bars = ax.bar(bin_centers, bin_means, width=bar_width,
                      color="steelblue", edgecolor="none")
        # mark empty bins so they don't silently disappear
        for bar, val in zip(bars, bin_means):
            if np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, 0,
                        "∅", ha="center", va="bottom", fontsize=8, color="gray")
        ax.set_xlabel("t  (bin centre)")
        ax.set_ylabel("mean |ε̂ − ε|  per bin")
        ax.set_title(f"{ep} — Binned error by t")
        _save(fig, f"{epoch}_12_bar_binned_err_vs_t")

    # ---- 13. Histogram of process SNR = 1 / σ²(t) -------------------
    # REDUNDANT: SNR = 1/σ²(t) is a monotone function of t, so this histogram
    # carries the same information as plot 01 (t distribution), just rescaled.
    # if len(snr) > 0:
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _hist(ax, snr, "process SNR = 1/σ²(t)", color="goldenrod")
    #     ax.set_title(f"{ep} — Process SNR distribution")
    #     _save(fig, f"{epoch}_13_hist_snr")

    # ---- 14. Scatter: error vs process SNR ---------------------------
    # REDUNDANT: log-SNR is used as the x-axis of plot 03b for VP/subVP SDEs,
    # and SNR = e^{-2 log σ} is a monotone function of log σ used for VE/GBM.
    # Either way, 03b captures the same error-vs-noise-level relationship.
    # if len(snr) > 0 and len(snr) == len(err):
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     _scatter(ax, snr, err, "process SNR = 1/σ²(t)", "mean |ε̂ − ε|  per sample",
    #              color="goldenrod")
    #     ax.set_title(f"{ep} — Error vs process SNR")
    #     _save(fig, f"{epoch}_14_scatter_err_vs_snr")

    # ---- 15. Scatter: ε̂ vs ε  (signed, subsampled) ------------------
    # Identity line y=x means perfect prediction; cloud width = residual noise.
    # Systematic bias (cloud above/below y=x) reveals directional mis-prediction.
    if len(eps_hat_vals) > 0 and len(eps_hat_vals) == len(eps_vals):
        v_min = float(np.nanpercentile(eps_vals, 1))
        v_max = float(np.nanpercentile(eps_vals, 99))
        fig, ax = plt.subplots(figsize=(6, 6))
        _scatter(ax, eps_vals, eps_hat_vals, "ε  (true noise)", "ε̂  (predicted noise)",
                 color="steelblue", s=2, alpha=0.2)
        ax.plot([v_min, v_max], [v_min, v_max], color="crimson", lw=1, linestyle="--",
                label="y = x")
        ax.set_xlim(v_min, v_max)
        ax.set_ylim(v_min, v_max)
        ax.legend(fontsize=8)
        ax.set_title(f"{ep} — ε̂ vs ε  (signed, subsampled)")
        ax.set_aspect("equal", adjustable="box")
        _save(fig, f"{epoch}_15_scatter_eps_hat_vs_eps")

    # ---- 16. Histogram of ε̂  (signed) -------------------------------
    # Should look like N(0,1) for a well-trained model.
    # Collapse toward 0, heavy tails, or asymmetry all signal training issues.
    if len(eps_hat_vals) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        _hist(ax, eps_hat_vals, "ε̂  (predicted noise, signed)", color="darkorange", bins=80)
        # overlay reference N(0,1) curve
        x_ref = np.linspace(eps_hat_vals.min(), eps_hat_vals.max(), 300)
        n_total = len(eps_hat_vals)
        bin_width = (eps_hat_vals.max() - eps_hat_vals.min()) / 80
        ref_density = (n_total * bin_width *
                       np.exp(-0.5 * x_ref ** 2) / np.sqrt(2 * np.pi))
        ax.plot(x_ref, ref_density, color="crimson", lw=1.5, linestyle="--",
                label="N(0,1) ref")
        ax.legend(fontsize=8)
        ax.set_title(f"{ep} — ε̂ distribution  (vs N(0,1) reference)")
        _save(fig, f"{epoch}_16_hist_eps_hat")

    # ---- 17. Scatter: reverse-step drift error vs t ------------------
    # err_rev(t) = g²(t)/σ(t) · mean_abs_err · Δt
    # = per-element RMS displacement error in one Euler-Maruyama reverse step.
    # Steep rise at large t → coarse-scale reconstruction is unreliable.
    # Steep rise at small t → fine-structure prediction is unreliable.
    if len(err_rev) > 0 and len(err_rev) == len(t) and len(x_interp) == len(err_rev):
        fig, ax = plt.subplots(figsize=(6, 4))
        _scatter(ax, x_interp, err_rev,
                 x_label, "err_rev = g²/σ · mean|ε̂−ε| · Δt  per sample",
                 color="mediumorchid")
        ax.set_title(f"{ep} — Reverse-step drift error vs {x_label}")
        _save(fig, f"{epoch}_17_scatter_err_rev_vs_noise_level")
        # print a concise numerical summary so the log captures R statistics
        # even without opening the plot
        if len(rel_stoch) > 0:
            print(
                f"  [rev-err] err_rev | mean={err_rev.mean():.3e}  "
                f"p50={np.median(err_rev):.3e}  p95={np.percentile(err_rev, 95):.3e}  "
                f"max={err_rev.max():.3e}\n"
                f"  [rev-err] R=err_rev/(g√Δt) | mean={rel_stoch.mean():.3e}  "
                f"p50={np.median(rel_stoch):.3e}  p95={np.percentile(rel_stoch, 95):.3e}  "
                f"max={rel_stoch.max():.3e}  "
                f"— fraction R≥1: {(rel_stoch >= 1.0).mean()*100:.1f}%"
            )

    # ---- 18. Bar chart: x0 reconstruction MSE per σ(t) bin ----------
    # Uses ||hat_x0 - x0||^2 = (σ/mean_coeff)^2 · ||ε - ε̂||^2 (algebraic identity).
    # Bins by actual σ(t) value rather than by t, so the x-axis directly shows
    # the noise level at which the model struggles to reconstruct x0.
    if len(x0_mse) > 0 and len(sigma) == len(x0_mse):
        _s_min = float(sigma.min())
        _s_max = float(sigma.max())
        _n_bins = 10
        _edges = np.linspace(_s_min, _s_max, _n_bins + 1)
        _centers = 0.5 * (_edges[:-1] + _edges[1:])
        _bin_mse = []
        for _lo, _hi in zip(_edges[:-1], _edges[1:]):
            _in = (sigma >= _lo) & (sigma < _hi)
            _bin_mse.append(float(x0_mse[_in].mean()) if _in.any() else float("nan"))
        _bin_mse = np.array(_bin_mse)

        fig, ax = plt.subplots(figsize=(7, 4))
        _bw = (_edges[1] - _edges[0]) * 0.85
        _bars = ax.bar(_centers, _bin_mse, width=_bw, color="teal", edgecolor="none")
        for _bar, _val in zip(_bars, _bin_mse):
            if np.isnan(_val):
                ax.text(_bar.get_x() + _bar.get_width() / 2, 0,
                        "∅", ha="center", va="bottom", fontsize=8, color="gray")
        ax.set_xlabel("σ(t)  (bin centre)")
        ax.set_ylabel("mean ||x̂₀ − x₀||²  per bin")
        ax.set_title(f"{ep} — x₀ reconstruction MSE by σ(t)")
        _save(fig, f"{epoch}_18_bar_x0_mse_vs_sigma")

        # text summary — printed even without opening the PNG
        _valid = [(c, m) for c, m in zip(_centers, _bin_mse) if not np.isnan(m)]
        if _valid:
            _parts = "  ".join(f"σ≈{c:.3f}:{m:.3e}" for c, m in _valid)
            print(f"  [x0-mse] per-σ bin: {_parts}")

    print(f"  [diagnostics] {len(saved)} plots saved → {diag_dir}")

    # ---- optional W&B upload ----------------------------------------
    if use_wandb:
        try:
            # Strip the leading "{epoch}_" from each key so that every epoch's
            # image lands under the *same* W&B key (e.g. "diagnostics/01_hist_t").
            # W&B then groups all logged values for that key into one media panel
            # with a step slider, letting you scroll across epochs without clutter.
            from PIL import Image as _PILImage   # add at top of file with other imports

            wb_imgs = {
                f"diagnostics/{k.split('_', 1)[1]}": wandb.Image(_PILImage.open(v))
                for k, v in saved.items()
            }

            wandb.log(wb_imgs, step=global_step)
            print(f"  [diagnostics] {len(wb_imgs)} images uploaded to W&B")
        except Exception as exc:
            print(f"  [diagnostics] W&B image upload failed: {exc}")

        # binned error table — logged separately so it's queryable in W&B
        try:
            valid_rows = [
                [float(c), float(m)]
                for c, m in zip(bin_centers, bin_means)
                if not np.isnan(m)
            ]
            if valid_rows:
                tbl = wandb.Table(
                    columns=["t_bin_center", "mean_abs_error"],
                    data=valid_rows,
                )
                wandb.log({"diagnostics/binned_err_vs_t": tbl}, step=global_step)
                print(f"  [diagnostics] binned-error table ({len(valid_rows)} bins) uploaded to W&B")
        except Exception as exc:
            print(f"  [diagnostics] W&B binned-error table upload failed: {exc}")


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
            # "num_steps": config["diffusion"]["num_steps"],
            "diffusion_embedding_dim": config["diffusion"]["diffusion_embedding_dim"],
            "channels": config["diffusion"]["channels"],
            "layers": config["diffusion"]["layers"],
            "nheads": config["diffusion"]["nheads"],
            "diffusion_embedding_dim": config["diffusion"]["diffusion_embedding_dim"],
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
    scheduler=None,
) -> Dict[str, Any]:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,                       # full config
        "run_metadata": build_run_metadata(config),  # quick-access summary
        "history": history,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    scheduler=None,
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

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

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
      9. diffusion_embedding_dim
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
        data_root = "REPL_RET"
    elif data_root == "./data/replication_returns_norm":
        data_root = "REPL_RET_NORM"
    elif data_root == "./data/toy_gaussian":
        data_root = "TOY_GAUSS"
    elif data_root == "./data/toy_ar1":
        data_root = "TOY_AR1"
    elif data_root == "./data/toy_studentt":
        data_root = "TOY_STUDENTT"
    elif data_root == "./data/toy_gbm":
        data_root = "TOY_GBM"
    elif data_root == "./data/toy_gbm_norm":
        data_root = "TOY_GBM_NORM"
    elif data_root == "./data/replication_prices_window":
        data_root = "REPL_PRIC_WIN"
    elif data_root == "./data/replication_returns_window":
        data_root = "REPL_RET_WIN"
    elif data_root == "./data/replication_returns_other":
        data_root = "REPL_RET_OTHER"
    elif data_root == "./data/log_replication":
        data_root = "REPL_LOG"
    elif data_root == "./data/filtered_windows/low_zero_windows":
        data_root = "REPL_LOW_ZERO"
    elif data_root == "./data/filtered_windows/high_zero_windows":
        data_root = "REPL_HIGH_ZERO"
    elif data_root == "./data/filtered_windows/moderate_variance_windows":
        data_root = "REPL_LOW_VAR"
    elif data_root == "./data/filtered_windows/high_variance_windows":
        data_root = "REPL_HIGH_VAR"
    elif data_root == "./data/filtered_windows/post_2001":
        data_root = "REPL_POST_2001"
    elif data_root == "./data/filtered_windows/pre_2001":
        data_root = "REPL_PRE_2001"
    elif data_root == "./data/filtered_windows/low_kurtosis":
        data_root = "REPL_LOW_KURT"
    elif data_root == "./data/filtered_windows/high_kurtosis":
        data_root = "REPL_HIGH_KURT"
    elif data_root == "./data/filtered_windows/post_2001_high_variance":
        data_root = "REPL_POST_2001_HIGH_VAR"
    elif data_root == "./data/filtered_windows/post_2001_low_variance":
        data_root = "REPL_POST_2001_LOW_VAR"
    elif data_root == "./data/filtered_windows/post_2001_high_kurtosis":
        data_root = "REPL_POST_2001_HIGH_KURT"
    elif data_root == "./data/filtered_windows/post_2001_low_kurtosis":
        data_root = "REPL_POST_2001_LOW_KURT"
    else:
        data_root = "REPL"
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
    # N = int(config["process"]["N"])
    # is_linear = "linear" if bool(config["diffusion"]["is_linear"]) else "notlinear"
    channels = int(config["diffusion"]["channels"])
    layers = int(config["diffusion"]["layers"])
    nheads = int(config["diffusion"]["nheads"])
    diffusion_embedding_dim = int(config["diffusion"]["diffusion_embedding_dim"])
    noise_schedule = str(config["process"].get("noise_schedule", "linear"))

    seq_len = int(config["train"].get("seq_len", 128))
    stride = int(config["train"].get("stride", 128))

    _sched_key = config["train"].get("lr_scheduler", None)
    if _sched_key is None:
        # legacy fallback
        _sched_key = "cosine-annealing" if bool(config["train"].get("lr_cosine_annealing", False)) else "none"
    if _sched_key == "cosine-annealing":
        cosine_annealing = "COSAN"
    elif _sched_key == "reduce-on-plateau":
        cosine_annealing = "ROP"
    else:
        cosine_annealing = "NOAN"

    # safer string for learning rate, e.g. 1e-4 -> 1e-04
    lr_str = f"{lr:.0e}"

    lw_tag = "LW" if bool(config["train"].get("likelihood_weighting", False)) else "NO"

    is_tag = "IS" if bool(config["train"].get("importance_sampling", False)) else "NO"

    filename = (
        f"{lw_tag}_"
        f"{is_tag}_"
        f"{data_root}_"
        f"{mask_mode}_"
        f"ep-{epochs}_"
        # f"step-{global_step}_"
        f"sde-{sde_type}_"
        f"noise-{noise_schedule}_"
        f"lr-{lr_str}_"
        f"{cosine_annealing}_"
        # f"N-{N}_"
        # f"{is_linear}_"
        f"channels-{channels}_"
        f"layers-{layers}_"
        f"nheads-{nheads}_"
        f"diffemb-{diffusion_embedding_dim}_"
        f"seq-{seq_len}_"
        f"stride-{stride}_"
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

    # Diagnostic images go to ./images/diagnostics/<run_name>/epoch_NNN/
    # Short folder name: <lw>_<is>_<data>_<mask>_ep-<N>_sde-<type>_<timestamp>
    _diag_lw   = "LW" if bool(config["train"].get("likelihood_weighting", False)) else "NO"
    _diag_is   = "IS" if bool(config["train"].get("importance_sampling", False)) else "NO"
    _diag_data = str(config["train"]["data_root"]).split("/")[-1]
    _diag_mask = str(config["train"]["mask_mode"])[:4].upper()
    _diag_ep   = int(config["train"]["epochs"])
    _diag_sde  = str(config["process"]["sde_type"])
    _run_name_for_diag = (
        f"{_diag_lw}_{_diag_is}_{_diag_data}_{_diag_mask}_"
        f"ep-{_diag_ep}_sde-{_diag_sde}_{run_timestamp}"
    )
    diag_base_dir = os.path.join("./images/diagnostics", _run_name_for_diag)

    # Resume state
    start_epoch = 0
    global_step = 0
    history = {"epoch_losses": [], "val_losses": []}

    # 5) LR scheduler
    # Resolve scheduler: new lr_scheduler key takes precedence over legacy lr_cosine_annealing.
    _lr_sched = config["train"].get("lr_scheduler", None)
    if _lr_sched is None:
        _lr_sched = "cosine-annealing" if bool(config["train"].get("lr_cosine_annealing", False)) else "none"

    eta_min = float(config["train"].get("lr_eta_min", 1e-6))

    if _lr_sched == "cosine-annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=num_epochs,
            eta_min=eta_min,
            last_epoch=-1,   # will be overridden by scheduler.load_state_dict if resuming
        )
        print(f"CosineAnnealingLR: lr={config['train']['lr']:.2e} → eta_min={eta_min:.2e} over {num_epochs} epochs")
    elif _lr_sched == "reduce-on-plateau":
        if val_loader is None:
            raise ValueError(
                "lr_scheduler='reduce-on-plateau' requires a validation set. "
                "Set train.val_split_ratio in config or pass --val_split_ratio."
            )
        _lr_patience = config["train"].get("lr_patience", None)
        if _lr_patience is None or int(_lr_patience) <= 0:
            raise ValueError(
                "lr_scheduler='reduce-on-plateau' requires lr_patience > 0. "
                "Set train.lr_patience in config or pass --lr_patience."
            )
        lr_gamma = float(config["train"].get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=lr_gamma,
            patience=int(_lr_patience),
            min_lr=eta_min,
        )
        print(f"ReduceLROnPlateau: patience={_lr_patience} epochs, factor={lr_gamma}, min_lr={eta_min:.2e}")
    else:  # "none"
        scheduler = None

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        start_epoch, global_step, history = load_checkpoint(
            resume_checkpoint, model, optim, scaler, device, scheduler=scheduler
        )
        print(f"Resumed at epoch={start_epoch}, global_step={global_step}")
        if _lr_sched == "cosine-annealing" and "scheduler" not in torch.load(resume_checkpoint, map_location="cpu"):
            # Checkpoint pre-dates scheduler support: fast-forward the cosine position manually
            for _ in range(start_epoch):
                scheduler.step()
            print(f"  [scheduler] Fast-forwarded cosine schedule to epoch {start_epoch}")

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
        init_timeout = wandb_cfg.get("init_timeout", 300)
        wandb.init(**init_kwargs, settings=wandb.Settings(init_timeout=init_timeout))
        history["wandb_run_id"] = wandb.run.id

    # Early stopping state
    early_stop_patience = config["train"].get("early_stop_patience", None)
    if early_stop_patience is not None:
        early_stop_patience = int(early_stop_patience) if int(early_stop_patience) > 0 else None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Δt for reverse-step error metrics: (T − eps) / N  (Euler-Maruyama step size)
    _delta_t = (processes.sde.T - processes.eps_time) / float(config["process"]["N"])

    # Importance sampling setup (computed once before training)
    use_is = bool(config["train"].get("importance_sampling", False))
    t_probs, t_grid = None, None
    if use_is:
        from src.utils.sde_utils import calculate_importance_sampling_probabilities
        t_probs, t_grid = calculate_importance_sampling_probabilities(
            processes.sde,
            N_timesteps=int(config["process"]["N"]),
            device=device,
            eps_time=processes.eps_time,
        )
        N_grid      = len(t_probs)
        # Effective sample size: ESS = 1/Σp² — if ESS≈N IS is nearly uniform,
        # if ESS≪N the distribution is highly concentrated on a few timesteps.
        ess         = (1.0 / t_probs.pow(2).sum()).item()
        ess_ratio   = ess / N_grid
        # Timestep that receives the highest sampling weight
        peak_idx    = int(t_probs.argmax().item())
        peak_t      = float(t_grid[peak_idx].item())
        # Split of probability mass below / above the midpoint of the t range
        t_mid       = float((t_grid[0] + t_grid[-1]).item() / 2)
        mass_low    = float(t_probs[t_grid <= t_mid].sum().item())
        # How many times more likely is the most-sampled vs least-sampled timestep
        concentration = float((t_probs.max() / t_probs.clamp(min=1e-12).min()).item())
        print(
            f"Importance sampling enabled:\n"
            f"  t_grid   | range=[{t_grid[0].item():.4f}, {t_grid[-1].item():.4f}]  N={N_grid}\n"
            f"  t_probs  | min={t_probs.min():.2e}  max={t_probs.max():.2e}"
            f"  mean={t_probs.mean():.2e}  std={t_probs.std():.2e}\n"
            f"  peak     | t={peak_t:.4f}  (grid idx={peak_idx})\n"
            f"  ESS      | {ess:.1f} / {N_grid}  (ratio={ess_ratio:.3f})"
            f"  —  concentration max/min={concentration:.1f}×\n"
            f"  mass     | t ≤ {t_mid:.3f}: {100*mass_low:.1f}%"
            f"  |  t > {t_mid:.3f}: {100*(1-mass_low):.1f}%"
        )

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
        _grad_buf = []   # grad norm per batch — epoch-end stats (replaces plots 05/06)
        _loss_buf = []   # step loss per batch — for Pearson r with grad norm (replaces plot 04)
        ema_beta    = float(config["train"].get("ema_beta", 0.98))
        use_lw      = bool(config["train"].get("likelihood_weighting", False))
        debug       = bool(config["train"].get("debug", False))
        _spike_factor = config["train"].get("loss_spike_factor", None)
        if _spike_factor is not None:
            _spike_factor = float(_spike_factor)
        print_plots = bool(config["train"].get("print_plots", False))
        plots_every = int(config["train"].get("plots_every", 1))
        # collect diagnostics this epoch only when print_plots is on AND the cadence hits
        collect_diag = print_plots and ((epoch + 1) % max(plots_every, 1) == 0)
        if collect_diag:
            diag = DiagnosticsCollector(delta_t=_delta_t)
        t0 = time.time()

        for batch_idx, batch in pbar:
            observed_data, observed_mask, observed_tp = unpack_batch(batch, device)
            if config["train"].get("normalization", False):
                observed_data = zscore_normalize_windows(observed_data)
            log_this_batch = debug and (batch_idx == 0)

            if log_this_batch:
                print(f"\n[debug | epoch {epoch+1} | train | batch 0]")
                _d    = observed_data.float()
                _flat = _d.flatten()
                print(f"  data  | mean={_flat.mean():.4f}  std={_flat.std():.4f}  "
                      f"min={_flat.min():.4f}  max={_flat.max():.4f}  "
                      f"p5={_flat.quantile(0.05):.4f}  p50={_flat.quantile(0.50):.4f}  p95={_flat.quantile(0.95):.4f}")
                _win_mean = _d.mean(dim=-1).flatten()   # (B*K,) — per-window mean over L timesteps
                _win_std  = _d.std(dim=-1).flatten()    # (B*K,) — per-window std  over L timesteps
                print(f"  win μ | mean={_win_mean.mean():.4f}  std={_win_mean.std():.4f}  "
                      f"p5={_win_mean.quantile(0.05):.4f}  p50={_win_mean.quantile(0.50):.4f}  p95={_win_mean.quantile(0.95):.4f}")
                print(f"  win σ | mean={_win_std.mean():.4f}  std={_win_std.std():.4f}  "
                      f"p5={_win_std.quantile(0.05):.4f}  p50={_win_std.quantile(0.50):.4f}  p95={_win_std.quantile(0.95):.4f}")

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
            # assert target_mask.sum() == observed_data.numel(), f"target_mask empty! sum={target_mask.sum().item()}"

            # forward diffusion
            x_t, t_cont, eps, sigma_t = processes.forward_process(
                observed_data, t_probs=t_probs, t_grid=t_grid
            )

            # g(t) for Song's likelihood weighting λ(t) = g(t)^2 / σ(t)^2
            # Also computed when collect_diag=True so plots 09/10 are always available
            if use_lw or collect_diag:
                with torch.no_grad():
                    _, g_t = processes.sde.sde(x_t, t_cont)
            else:
                g_t = None

            if log_this_batch:
                print(f"  sigma | min={sigma_t.min():.4f}  max={sigma_t.max():.4f}  mean={sigma_t.mean():.4f}  median={sigma_t.median():.4f}")

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
                    loss = masked_mse(eps_hat, eps, target_mask, sigma_t=sigma_t, g_t=g_t, debug=log_this_batch)
                else:
                    loss = masked_mse(eps_hat, eps, target_mask, None, debug=log_this_batch)

                if log_this_batch:
                    print(f"  eps_hat | norm={eps_hat.norm().item():.4f}  mean={eps_hat.mean().item():.4f}  std={eps_hat.std().item():.4f}  min={eps_hat.min().item():.4f}  max={eps_hat.max().item():.4f}  median={eps_hat.median().item():.4f}")
                    print(f"  eps     | norm={eps.norm().item():.4f}  mean={eps.mean().item():.4f}  std={eps.std().item():.4f}  min={eps.min().item():.4f}  max={eps.max().item():.4f}  median={eps.median().item():.4f}")
                    _a = eps.detach().float().view(-1);  _a = _a - _a.mean()
                    _b = eps_hat.detach().float().view(-1);  _b = _b - _b.mean()
                    print(f"  corr(ε, ε̂) = {float((_a * _b).sum() / (_a.norm() * _b.norm() + 1e-8)):.4f}")

            # capture eps_hat before backward releases the computation graph
            _diag_eps_hat = eps_hat.detach().float() if collect_diag else None

            # Collapse detection — every batch, regardless of debug mode.
            # Fires when std(ε̂) < 10 % of std(ε): model is hedging toward zero
            # rather than predicting the noise distribution.
            with torch.no_grad():
                _std_hat = eps_hat.detach().float().std().item()
                _std_eps = eps.float().std().item()
                if _std_hat < 0.1 * (_std_eps + 1e-8):
                    print(f"  [COLLAPSE step={global_step}]  std(ε̂)={_std_hat:.4f}  "
                          f"std(ε)={_std_eps:.4f}  ratio={_std_hat / (_std_eps + 1e-8):.3f}")

            # mean_coeff(t): needed for x0 reconstruction MSE diagnostic.
            # VP/SubVP SDEs expose .mean_coeff(t); VE/GBM have mean_coeff = 1.
            if collect_diag:
                with torch.no_grad():
                    _mean_coeff_t = (
                        processes.sde.mean_coeff(t_cont)
                        if hasattr(processes.sde, "mean_coeff")
                        else torch.ones_like(sigma_t)
                    )
            else:
                _mean_coeff_t = None

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
            scaler.unscale_(optim)
            # IMPORTANT: we are doing gradient clipping, because of extreme gradient values
            max_norm = 5.0
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optim)
            scaler.update()

            # DEBUGGING: AMP scaler silently skipping every update
            # scale = scaler.get_scale()
            # print(f"AMP scale: {scale}")   # if this keeps halving, NaN gradients are occurring
            # END DEBUGGING

            # collect diagnostic statistics for this batch
            if collect_diag:
                diag.add(
                    t_cont=t_cont,
                    eps_hat=_diag_eps_hat,
                    eps=eps,
                    sigma_t=sigma_t,
                    target_mask=target_mask,
                    grad_norm=float(grad_norm),
                    g_t=g_t,
                    mean_coeff_t=_mean_coeff_t,
                )

            global_step += 1

            elapsed = time.time() - t0
            steps_done = batch_idx + 1
            it_per_s = steps_done / max(elapsed, 1e-9)

            grad_norm_val = float(grad_norm)
            _grad_buf.append(grad_norm_val)
            _loss_buf.append(loss_val)

            #Activated after 50 batches to allow EMA to stabilize, and only if a spike factor is set in config
            if _spike_factor is not None and ema_loss is not None and batch_idx >= 50:
                if loss_val > _spike_factor * ema_loss:
                    print(f"  [spike step={global_step}]  loss={loss_val:.4f}  ema={ema_loss:.4f}  "
                          f"ratio={loss_val / ema_loss:.1f}x  t={t_cont.float().mean().item():.3f}  "
                          f"sigma={sigma_t.float().mean().item():.4f}  gnorm={grad_norm_val:.3f}")

            if log_this_batch:
                _clip_flag = "CLIPPED" if grad_norm_val >= max_norm - 0.01 else "ok"
                print(f"  grad_norm | {grad_norm_val:.4f}  ({_clip_flag})")

            pbar.set_postfix({
                "step": global_step,
                "loss": f"{loss_val:.4f}",
                "ema": f"{ema_loss:.4f}",
                "avg": f"{(epoch_loss_sum / epoch_loss_count):.4f}",
                "it/s": f"{it_per_s:.2f}",
                "t": f"{t_cont.float().mean().item():.1f}",
                "gnorm": f"{grad_norm_val:.2f}",
            })

            if global_step % int(config["train"]["log_every_steps"]) == 0:
                print(
                    f"epoch={epoch+1}/{num_epochs} "
                    f"step={global_step} "
                    f"loss={loss_val:.6f} "
                    f"grad_norm={grad_norm_val:.4f} "
                    f"t_cont_mean={t_cont.float().mean().item():.2f}"
                )
                if use_wandb:
                    wandb.log({
                        "train/step_loss": loss_val,
                        "train/ema_loss": ema_loss,
                        "train/avg_loss": epoch_loss_sum / epoch_loss_count,
                        "train/it_per_s": it_per_s,
                        "train/grad_norm": grad_norm_val,
                    }, step=global_step)

        # end of epoch
        epoch_avg = epoch_loss_sum / max(epoch_loss_count, 1)

        # --- Epoch-level gradient stats (text replacement for plots 04/05/06) ---
        if _grad_buf:
            _gn = np.array(_grad_buf)
            _lb = np.array(_loss_buf)
            _clip_pct = (_gn >= 4.99).mean() * 100
            _r_str = ""
            if len(_gn) > 1 and _gn.std() > 1e-12 and _lb.std() > 1e-12:
                _r = float(np.corrcoef(_lb, _gn)[0, 1])
                _warn = "  ← decoupled, check LW weights" if abs(_r) < 0.1 else ""
                _r_str = f"  Pearson r(loss, grad_norm)={_r:.3f}{_warn}"
            print(
                f"  [grad] mean={_gn.mean():.4f}  std={_gn.std():.4f}  "
                f"p50={np.median(_gn):.4f}  p95={np.percentile(_gn, 95):.4f}  "
                f"max={_gn.max():.4f}  clipped={_clip_pct:.1f}%{_r_str}"
            )

        # validation
        val_avg = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_loader):
                    observed_data, observed_mask, observed_tp = unpack_batch(val_batch, device)
                    if config["train"].get("normalization", False):
                        observed_data = zscore_normalize_windows(observed_data)
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
                        cond_mask = get_predict_close_mask(observed_mask)
                    else:  # "random"
                        cond_mask = get_randmask(
                            observed_mask,
                            min_ratio=float(config["train"]["cond_min_ratio"]),
                            max_ratio=float(config["train"]["cond_max_ratio"]),
                        )

                    target_mask = (observed_mask.float() * (1.0 - cond_mask.float())).float()

                    x_t, t_cont, eps, sigma_t = processes.forward_process(
                        observed_data, t_probs=t_probs, t_grid=t_grid
                    )

                    # g(t) for Song's likelihood weighting λ(t) = g(t)^2 / σ(t)^2
                    if use_lw:
                        _, g_t = processes.sde.sde(x_t, t_cont)
                    else:
                        g_t = None

                    if log_this_val_batch:
                        print(f"  sigma | min={sigma_t.min():.4f}  max={sigma_t.max():.4f}  mean={sigma_t.mean():.4f}  median={sigma_t.median():.4f}")

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        eps_hat = model(
                            x_t=x_t,
                            t=t_cont,
                            observed_data=observed_data,
                            cond_mask=cond_mask,
                            observed_tp=observed_tp,
                        )
                        if use_lw:
                            val_loss = masked_mse(eps_hat, eps, target_mask, sigma_t=sigma_t, g_t=g_t, debug=log_this_val_batch)
                        else:
                            val_loss = masked_mse(eps_hat, eps, target_mask, None, debug=log_this_val_batch)

                        if log_this_val_batch:
                            print(f"  eps_hat | norm={eps_hat.norm().item():.4f}  mean={eps_hat.mean().item():.4f}  std={eps_hat.std().item():.4f}  min={eps_hat.min().item():.4f}  max={eps_hat.max().item():.4f}  median={eps_hat.median().item():.4f}")
                            print(f"  eps     | norm={eps.norm().item():.4f}  mean={eps.mean().item():.4f}  std={eps.std().item():.4f}  min={eps.min().item():.4f}  max={eps.max().item():.4f}  median={eps.median().item():.4f}")
                            _va = eps.float().view(-1);        _va = _va - _va.mean()
                            _vb = eps_hat.float().view(-1);    _vb = _vb - _vb.mean()
                            print(f"  corr(ε, ε̂) = {float((_va * _vb).sum() / (_va.norm() * _vb.norm() + 1e-8)):.4f}")

                    # Collapse detection — every val batch, regardless of debug mode.
                    _std_hat_v = eps_hat.float().std().item()
                    _std_eps_v = eps.float().std().item()
                    if _std_hat_v < 0.1 * (_std_eps_v + 1e-8):
                        print(f"  [COLLAPSE val step={val_batch_idx}]  std(ε̂)={_std_hat_v:.4f}  "
                              f"std(ε)={_std_eps_v:.4f}  ratio={_std_hat_v / (_std_eps_v + 1e-8):.3f}")

                    val_loss_sum += float(val_loss.item())
                    val_loss_count += 1

            val_avg = val_loss_sum / max(val_loss_count, 1)
            history["val_losses"].append({"epoch": epoch + 1, "avg_val_loss": val_avg})
            model.train()

        # diagnostic plots (end of epoch, only when cadence hits)
        if collect_diag:
            plot_and_save_diagnostics(
                collector=diag,
                epoch=epoch + 1,
                diag_base_dir=diag_base_dir,
                use_wandb=use_wandb,   # images saved locally only; export from HPC manually
                global_step=global_step,
                sde_type=config["process"]["sde_type"],
            )

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

        # LR scheduler step (once per epoch, after validation)
        if scheduler is not None:
            if _lr_sched == "reduce-on-plateau":
                # val_avg is guaranteed non-None when reduce-on-plateau is active
                # (we validated val_loader is not None at scheduler creation time)
                if val_avg is not None:
                    scheduler.step(val_avg)
                    current_lr = optim.param_groups[0]["lr"]
                    print(f"  [scheduler] ReduceLROnPlateau step — lr={current_lr:.2e}")
                    if use_wandb:
                        wandb.log({"train/lr": current_lr}, step=global_step)
            else:  # cosine-annealing
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [scheduler] lr={current_lr:.2e}")
                if use_wandb:
                    wandb.log({"train/lr": current_lr}, step=global_step)

        # early stopping (only when val_loader is provided)
        if early_stop_patience is not None and val_avg is not None:
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                epochs_no_improve = 0
                best_ckpt_path = os.path.join(out_dir, f"best_{run_timestamp}.pt")
                torch.save(
                    make_checkpoint_payload(model, optim, scaler, config, epoch, global_step, history, scheduler=scheduler),
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
                scheduler=scheduler,
            )
            torch.save(payload, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # final save
    final_payload = make_checkpoint_payload(
        model=model,
        optim=optim,
        scaler=scaler,
        config=config,
        epoch=epoch,
        global_step=global_step,
        history=history,
        scheduler=scheduler,
    )
    final_payload["training_completed"] = (epoch + 1) == num_epochs
    final_payload["early_stopped"] = (epoch + 1) < num_epochs

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
# python csdi_train_modified.py --config configs/replication.yaml --epochs 100 --train_subset_size 16 --val_split_ratio 0.01 --data_root "./data/filtered_windows" --noise_schedule exponential --sigma_min 0.01 --sigma_max 1.0 --beta_min 0.01 --beta_max 20.0
# python csdi_train_modified.py --config configs/replication.yaml --epochs 100 --sde_type vp --noise_schedule cosine --sigma_min 0.01 --sigma_max 1.0 --beta_min 0.01 --beta_max 7.0 --data_root ./data/filtered_windows/pre_2001 --mask_mode unconditional --likelihood_weighting false --importance_sampling false --seq_len 2048 --stride 400 --weight_decay 0.0 --channels 128 --layers 4 --nheads 8 --diffusion_embedding_dim 256 --train_subset_size 16 --val_split_ratio 0.1 --batch_size 16 --lr_scheduler reduce-on-plateau --lr_patience 2 --lr 1e-3  --lr_eta_min 1e-6 --seed 42 --use_amp true --early_stop_patience 1000 --debug false --print_plots false