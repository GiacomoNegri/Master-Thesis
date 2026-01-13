# csdi_scratch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def default_timepoints(x: torch.Tensor) -> torch.Tensor:
    # x: (B, L, K)
    B, L, _ = x.shape
    tp = torch.arange(L, device=x.device).float().unsqueeze(0).expand(B, L)
    return tp


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for diffusion step t (integer).
    Returns (B, d_model).
    """
    def __init__(self, d_model: int, max_steps: int = 2000):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer steps
        half = self.d_model // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(half, device=t.device).float() / half
        )  # (half,)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, d_model)


@dataclass
class DiffusionConfig:
    num_steps: int = 50            # start small for feasibility
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1


# -----------------------------
# Denoiser (Transformer over time)
# -----------------------------
class TimeSeriesDenoiser(nn.Module):
    """
    Predicts noise epsilon for x_t.
    Operates on sequences over time:
      x_t:      (B, L, K)
      cond_x0:  (B, L, K) (conditioning values; not necessarily complete)
      cond_mask:(B, L, K) {0,1} where cond_x0 is enforced/known
      t_emb:    (B, d_model)
    Output:
      eps_pred: (B, L, K)
    """
    def __init__(self, K: int, cfg: DiffusionConfig):
        super().__init__()
        self.K = K
        self.cfg = cfg

        # Input channels: x_t(K) + cond_x0(K) + cond_mask(K) + observed_mask(K) (optional if provided)
        # To keep it simple: x_t(K) + cond_x0(K) + cond_mask(K) => 3K channels
        in_dim = 3 * K

        self.in_proj = nn.Linear(in_dim, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.out_proj = nn.Linear(cfg.d_model, K)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_x0: torch.Tensor,
        cond_mask: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # x_t, cond_x0, cond_mask: (B, L, K)
        # t_emb: (B, d_model)
        B, L, K = x_t.shape
        assert K == self.K

        inp = torch.cat([x_t, cond_x0, cond_mask], dim=-1)  # (B, L, 3K)
        h = self.in_proj(inp)  # (B, L, d_model)

        # Add diffusion-step embedding as a bias to every time token
        t_bias = self.t_mlp(t_emb).unsqueeze(1)  # (B, 1, d_model)
        h = h + t_bias

        h = self.encoder(h)  # (B, L, d_model)
        eps = self.out_proj(h)  # (B, L, K)
        return eps


# -----------------------------
# CSDI-style diffusion wrapper
# -----------------------------
class CSDIFromScratch(nn.Module):
    """
    From-scratch conditional diffusion for time-series imputation.

    Batch format (torch tensors):
      observed_data: (B, L, K) float
      observed_mask: (B, L, K) float {0,1}  # indicates which values exist in your dataset window
      cond_mask:     (B, L, K) float {0,1}  # subset of observed_mask; values to condition on
        - positions with observed_mask=1 but cond_mask=0 are targets for imputation loss
      timepoints:    (B, L) float (optional; not used by transformer here, but kept for compatibility)

    Notes:
      - If your data has no missing values initially, set observed_mask=1 everywhere,
        and create cond_mask by artificially masking for feasibility tests.
    """
    def __init__(self, K: int, cfg: DiffusionConfig):
        super().__init__()
        self.K = K
        self.cfg = cfg

        self.t_embed = SinusoidalTimeEmbedding(cfg.d_model, max_steps=max(2000, cfg.num_steps + 10))
        self.denoiser = TimeSeriesDenoiser(K=K, cfg=cfg)

        # Diffusion schedule buffers
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps).float()  # (T,)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)                 # (T,)
        self.register_buffer("alphas", alphas)               # (T,)
        self.register_buffer("alpha_bars", alpha_bars)       # (T,)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def _get_t_coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sqrt(alpha_bar_t) and sqrt(1-alpha_bar_t) shaped for broadcasting: (B, 1, 1)
        """
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        s = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return a, s

    def _extract_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = batch["observed_data"].float()        # (B, L, K)
        obs_mask = batch["observed_mask"].float()  # (B, L, K)
        cond_mask = batch["cond_mask"].float()     # (B, L, K)

        # enforce logical constraint
        cond_mask = cond_mask * obs_mask

        tp = batch.get("timepoints", None)
        if tp is None:
            tp = default_timepoints(x0)  # (B, L)
        else:
            tp = tp.float()

        return x0, obs_mask, cond_mask, tp

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training loss: E[ || eps_pred - eps ||^2 ] over target positions.
        """
        x0, obs_mask, cond_mask, _ = self._extract_batch(batch)
        device = x0.device
        B, L, K = x0.shape
        assert K == self.K

        # Define target positions: observed but not conditioned
        target_mask = (obs_mask - cond_mask).clamp(min=0.0, max=1.0)  # (B, L, K)

        # Sample diffusion step per batch element
        t = torch.randint(low=0, high=self.cfg.num_steps, size=(B,), device=device)  # (B,)
        a, s = self._get_t_coeffs(t)  # (B,1,1)

        eps = torch.randn_like(x0)  # (B,L,K)
        x_t = a * x0 + s * eps      # forward diffusion

        # Enforce conditioning directly in x_t (important)
        x_t = x_t * (1.0 - cond_mask) + x0 * cond_mask

        t_emb = self.t_embed(t)  # (B, d_model)
        eps_pred = self.denoiser(x_t=x_t, cond_x0=x0, cond_mask=cond_mask, t_emb=t_emb)

        # loss only on target positions
        denom = target_mask.sum().clamp(min=1.0)
        loss = ((eps_pred - eps) ** 2 * target_mask).sum() / denom
        return loss

    @torch.no_grad()
    def impute(self, batch: Dict[str, torch.Tensor], n_samples: int = 8) -> torch.Tensor:
        """
        Returns samples: (B, n_samples, L, K)
        """
        x0, obs_mask, cond_mask, _ = self._extract_batch(batch)
        device = x0.device
        B, L, K = x0.shape
        T = self.cfg.num_steps

        samples = []
        for _ in range(n_samples):
            # Start from pure noise
            x = torch.randn((B, L, K), device=device)

            # Reverse diffusion
            for step in reversed(range(T)):
                t = torch.full((B,), step, device=device, dtype=torch.long)
                t_emb = self.t_embed(t)

                eps_pred = self.denoiser(x_t=x, cond_x0=x0, cond_mask=cond_mask, t_emb=t_emb)

                beta_t = self.betas[step]
                alpha_t = self.alphas[step]
                alpha_bar_t = self.alpha_bars[step]

                # DDPM mean
                # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_pred) + sigma_t*z
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
                mean = coef1 * (x - coef2 * eps_pred)

                if step > 0:
                    z = torch.randn_like(x)
                    # simple variance choice (beta_t). More elaborate choices exist; this is enough for feasibility.
                    x = mean + torch.sqrt(beta_t) * z
                else:
                    x = mean

                # Re-apply conditioning
                x = x * (1.0 - cond_mask) + x0 * cond_mask

            samples.append(x.unsqueeze(1))  # (B,1,L,K)

        return torch.cat(samples, dim=1)  # (B, n_samples, L, K)
