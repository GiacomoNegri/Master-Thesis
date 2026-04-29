import torch
import torch.nn as nn

import time
import matplotlib.pyplot as plt

from src.utils.WIP_SDE import SDE, BetaScheduleSDE, SigmaSchedule, SubVPSDE, VESDE, VPSDE, GBMLogSDE


# ---------------------------------------------------------------------------
# Reverse-step consistency diagnostic
# ---------------------------------------------------------------------------

@torch.no_grad()
def _ode_consistency_discrepancy(
    rsde,
    x: torch.Tensor,
    t_i: torch.Tensor,
    f_full: torch.Tensor,
    t_next: torch.Tensor,
) -> dict:
    """
    Local reverse-step consistency check for deterministic ODE sampling.

    Compares two ways of advancing from state *x* at time *t_i*:
      (1) One full Euler step:
              x_full = x - f_full
      (2) Two consecutive half-steps:
              x_mid  = x - f_full * 0.5                   (no extra model call)
              f2, _  = rsde.discretize(x_mid, t_mid)      (one extra model call)
              x_2h   = x_mid - f2 * 0.5

    For a first-order ODE integrator the discrepancy |x_full - x_2h| is O(dt^2)
    and acts as a proxy for local truncation error.

    Args:
        rsde:    Reverse SDE (must expose .discretize(x, t, labels=None)).
        x:       Current state, shape (B, K, L).
        t_i:     Current time vector, shape (B,).
        f_full:  Pre-computed full-step drift f = drift(x, t) * dt_sde, same shape as x.
        t_next:  Next grid time vector, shape (B,). Midpoint is placed at 0.5*(t_i + t_next).

    Returns:
        dict with float keys "l2", "mae", "max_ae".
    """
    # Full step (reuse already-computed f)
    x_full = x - f_full

    # First half step — scale f_full, no model call
    x_mid = x - f_full * 0.5

    # Second half step — one extra model call at the midpoint time
    t_mid = 0.5 * (t_i + t_next)
    f2, _ = rsde.discretize(x_mid, t_mid, labels=None)
    x_2h = x_mid - f2 * 0.5

    diff = (x_full - x_2h).float()
    return {
        "l2":     diff.norm().item(), #L2 norm collapse the entire discrepancy tensor (B,K,L) into a single scalar that summarise the magnitude of the local error.
        "mae":    diff.abs().mean().item(),
        "max_ae": diff.abs().max().item(),
    }


def _expand_batch_vector_to(x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Expand a (B,) vector to match the shape of x (B, C, H, W, ...).
    Args:
        x: tensor of shape (B, ...)
        vec: tensor of shape (B,)
    Returns:
        tensor of shape (B, 1, 1, 1, ...) broadcastable with x
    """
    while vec.dim() < x.dim():
        vec = vec.unsqueeze(-1)
    return vec


###########################################
# TimeMapper to combine the discrete expecatation from the model
# with the continuous of the model
###########################################
# class TimeMapper:
#     """
#     Defines the bijection between:
#       - continuous SDE time t_cont in [0, T]
#       - discrete model timestep t_idx in {0, ..., S-1}
#     """
#     def __init__(self, T: float, S: int):
#         self.T = float(T)
#         self.S = int(S)

#     def cont_to_idx(self, t_cont: torch.Tensor) -> torch.Tensor:
#         # t_cont: (B,) float in [0, T]
#         t01 = (t_cont / self.T).clamp(0.0, 1.0)
#         idx = torch.round(t01 * (self.S - 1)).long()
#         return idx.clamp(0, self.S - 1)

#     def idx_to_cont(self, t_idx: torch.Tensor) -> torch.Tensor:
#         # t_idx: (B,) long in [0, S-1]
#         t01 = t_idx.float() / (self.S - 1)
#         return t01 * self.T


class Diffusion_Processes:
    def __init__(self, cfg: dict):
        self.N = int(cfg["N"])
        self.sde_type = cfg["sde_type"].lower()

        # Modifications
        self.model_steps = int(cfg['model_steps'])
        self.eps_time = float(cfg.get("eps", 1e-3))

        self.enforce_observed = bool(cfg.get("enforce_observed", True))
        self.min_eps = cfg.get("min_eps", None)
        self.max_eps = cfg.get("max_eps", None)
        self.round_times = bool(cfg.get("round_times", False))
        self.time_num_bins = int(cfg.get("time_num_bins", 10))
        self.time_multiplier = float(cfg.get("time_multiplier", 1.0))
        self.t_min = cfg.get("t_min", None)
        self.t_max = cfg.get("t_max", None)
        self.freeze_eps = bool(cfg.get("freeze_eps", False))
        self.round_eps = bool(cfg.get("round_eps", False))
        self.eps_num_bins = int(cfg.get("eps_num_bins", 10))
        self.eps_multiplier = float(cfg.get("eps_multiplier", 1.0))
        # self.conditional = cfg.get("conditional", False)
        # self.num_attributes = cfg.get("num_attributes", 0)
        # self.guidance_scale = cfg.get("guidance_scale", 1.5)

        noise_schedule = cfg.get("noise_schedule", None)
        sigma_min = float(cfg.get("sigma_min", 0.01))
        sigma_max = float(cfg.get("sigma_max", 1.0))
        beta_min  = float(cfg.get("beta_min",  0.1))
        beta_max  = float(cfg.get("beta_max",  20.0))

        if self.sde_type == "ve":
            self.sde: SDE = VESDE(
                N=self.N,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                schedule=noise_schedule if noise_schedule is not None else "exponential",
            )
        elif self.sde_type == "vp":
            self.sde: SDE = VPSDE(
                N=self.N,
                beta_min=beta_min,
                beta_max=beta_max,
                schedule=noise_schedule if noise_schedule is not None else "linear",
            )
        elif self.sde_type == "subvp":
            self.sde: SDE = SubVPSDE(
                N=self.N,
                beta_min=beta_min,
                beta_max=beta_max,
                schedule=noise_schedule if noise_schedule is not None else "linear",
            )
        else:
            self.sde: SDE = GBMLogSDE(
                N=self.N,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                schedule=noise_schedule if noise_schedule is not None else "exponential",
            )

        # self.mapper = TimeMapper(T=self.sde.T, S=cfg["model_steps"])
        

    @torch.no_grad()
    def forward_process(
        self,
        x0: torch.Tensor,
        t: torch.Tensor = None,
        t_probs: torch.Tensor = None,
        t_grid: torch.Tensor = None,
    ):
        """
        Forward diffusion: add noise to clean data z0 according to the chosen SDE.
        This uses the closed-form marginal p_t(z | z0):
            z_t = mean(z0, t) + std(t) * eps,  eps ~ N(0, I)
        Args:
            x0:      Clean data, shape (B, K, L).
            t:       Optional pre-sampled time vector (B,). If provided, skips sampling.
            t_probs: Optional (N_grid,) IS probability vector. When given together with
                     t_grid, t is drawn from t_grid according to t_probs instead of
                     uniformly. Compute once with sde_utils.calculate_importance_sampling_probabilities.
            t_grid:  Optional (N_grid,) tensor of t values paired with t_probs.
        Returns:
            x_t:    Noised data at sampled time t, same shape as x0.
            t:      Time vector, shape (B,).
            eps:    The Gaussian noise used, same shape as x0.
            std:    Per-sample noise std σ(t), shape (B,).
        """
        device = x0.device
        B = x0.size(0)

        if t is None:
            if t_probs is not None and t_grid is not None:
                # Importance sampling: optionally restrict to [t_min, t_max] window by
                # filtering t_grid and renormalising the IS weights — no hard-wall clipping.
                t_lo = float(self.t_min) if self.t_min is not None else self.eps_time
                t_hi = float(self.t_max) if self.t_max is not None else self.sde.T
                mask = (t_grid >= t_lo) & (t_grid <= t_hi)
                filtered_grid  = t_grid[mask]
                filtered_probs = t_probs[mask]
                filtered_probs = filtered_probs / filtered_probs.sum()  # renormalise
                indices = torch.multinomial(
                    filtered_probs.unsqueeze(0).expand(B, -1),
                    num_samples=1,
                    replacement=True,
                ).squeeze(-1)  # (B,)
                t = filtered_grid[indices]
            else:
                # Default: uniform sampling over [eps_time, T]
                t = self.eps_time + torch.rand(B, device=device) * (self.sde.T - self.eps_time)

        if self.round_times:
            t_lo = float(self.t_min) if self.t_min is not None else self.eps_time
            t_hi = float(self.t_max) if self.t_max is not None else self.sde.T
            bins = torch.linspace(t_lo, t_hi, self.time_num_bins, device=device) * self.time_multiplier
            bin_idx = torch.randint(0, self.time_num_bins, (1,), device=device).item()
            t = bins[bin_idx].expand(B)

        print(f"Forward process: sampling t with shape {t.shape} and range [{t.min().item():.4f}, {t.max().item():.4f}], t_mean={t.mean().item():.4f}")
        # Get closed-form mean and std of p_t(z | z0)
        mean, std = self.sde.marginal_prob(x0, t)  # mean: (B, ...), std: (B,)

        # Sample noise
        freeze_eps_value = 0.5
        eps = torch.full_like(x0, freeze_eps_value) if self.freeze_eps else torch.randn_like(x0)
        if self.min_eps is not None or self.max_eps is not None:
            eps = eps.clamp(
                min=self.min_eps if self.min_eps is not None else -float("inf"),
                max=self.max_eps if self.max_eps is not None else  float("inf"),
            )

        if self.round_eps:
            if self.min_eps is None or self.max_eps is None:
                raise ValueError("round_eps=True requires both min_eps and max_eps to be set.")
            bins = torch.linspace(self.min_eps, self.max_eps, self.eps_num_bins, device=x0.device)
            bin_idx = torch.randint(0, self.eps_num_bins, (1,), device=x0.device).item()
            eps = torch.full_like(x0, bins[bin_idx].item())

        if self.eps_multiplier != 1.0:
            eps = eps * self.eps_multiplier

        print(f"Noise sampled with shape {eps.min().item():.4f}, {eps.max().item():.4f}, mean={eps.mean().item():.4f}, std={eps.std().item():.4f}")
        # Broadcast std to match z0

        # Construct z_t
        std_b = _expand_batch_vector_to(x0,std)
        x_t = mean + std_b * eps

        return x_t, t, eps, std
    
    @torch.no_grad()
    def reverse_process(
        self,
        model: nn.Module, # trained CSDIModel
        shape,
        observed_data: torch.Tensor, #(B,K,L)
        cond_mask: torch.Tensor, #(B,K,L)
        observed_tp: torch.Tensor,
        num_steps: int = None,
        probability_flow: bool = False,
        device: torch.device = None,
        debug: bool = False,
        disc_out: list = None,
        use_heun: bool = False,
        cmp_out: list = None,
        # labels: torch.Tensor = None,
    ):
        """
        Reverse diffusion: sample from the data distribution using the learned model.
        This integrates the reverse-time SDE/ODE defined by self.sde.reverse().
        Assumptions:
            - model(x, t) returns the score ∇_x log p_t(x) (Song-style score model).
              If your model predicts noise ε instead, you must wrap it and convert
              to a score before passing it here.
        Args:
            model: neural net implementing score(x, t).
            shape: shape of the samples to generate, e.g. (B, C, H, W).
            num_steps: number of reverse-time discretization steps (default: self.N).
            probability_flow: if True, use probability flow ODE (deterministic);
                              if False, use reverse SDE (stochastic).
        Returns:
            x: generated samples, tensor of shape `shape`.
        """
        if num_steps is None:
            num_steps = self.N

        if use_heun:
            assert probability_flow, "use_heun requires probability_flow=True"
        
        if device is None:
            device = next(model.parameters()).device

        # # --- FIX: Check if model is a function or a class ---
        # if device is None:
        #     if hasattr(model, "parameters"):
        #         # It's a real PyTorch model
        #         device = next(model.parameters()).device
        #     else:
        #         # It's a function (wrapper), so we assume CUDA or CPU
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #next(model.parameters()).device
        B = shape[0]
        T = self.sde.T
        print(f"This is our SDE: {self.sde}")
        print(f"This is the value of T: {T}")


        def score_fn(x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
            """
            Computes the score using the pre-trained model.
            Handles the mapping from continuous SDE time t to model-specific inputs.
            """
            # t_idx = self.mapper.cont_to_idx(t)

            eps_hat = model.predict_eps(
                x_t = x,
                t = t,
                observed_data = observed_data,
                cond_mask = cond_mask,
                observed_tp = observed_tp
            )
            # if self.conditional:
            #     null_y = torch.full((B,), self.num_classes, device=device)
            #     x_combined = torch.cat([x,x], dim=0)
            #     t_combined = torch.cat([t,t], dim=0)
            #     labels_combined = torch.cat([labels, null_y], dim=0)
            # else:
            #     x_combined = x
            #     t_combined = t
            #     labels_combined = None

            # 1. Get the marginal std (sigma) from the SDE
            #    std shape: (B,)
            _, std = self.sde.marginal_prob(x, t)
            std_b = _expand_batch_vector_to(x, std)
            score = -eps_hat / (std_b + 1e-6)
            # model_input_t = t
            return score

            # 3. Forward Pass
            # .sample is REQUIRED because diffusers models return an output object
            # model_out = model(x_combined, t_combined, labels_combined)

            # eps_cond, eps_uncond = model_out.chunk(2, dim=0)

            # # CDF Extrapolation
            # eps_cfg = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

            # # 4. Convert Output to Score
            # # Reshape std for broadcasting: (B, 1, 1, 1)
            # std = std.view(*std.shape, *([1] * (x.dim() - 1)))
            
            # if self.sde_type == "ve":
            #     # VE: Model predicts score * sigma (approx).
            #     # score = output / sigma
            #     score = eps_cfg / (std + 1e-6)
            # else:
            #     # VP: Model predicts noise (epsilon).
            #     # score = -epsilon / sigma
            #     score = -eps_cfg / (std + 1e-6)

            # return score

        # Build reverse-time SDE/ODE
        rsde: SDE = self.sde.reverse(score_fn, probability_flow=probability_flow)

        # Whether to run the ODE consistency check this run
        _run_consistency = debug and probability_flow and (disc_out is not None)

        # Whether to run the paired ODE-vs-Heun comparison
        # Requires use_heun=True so both f and f2 are already computed per step.
        # Maintains a shadow ODE-Euler trajectory from the same initial x.
        # Costs one extra model call per step (rsde.discretize for x_ode at i > 0).
        _run_cmp = debug and use_heun and (cmp_out is not None)

        # Initialize from the prior at time T
        x = self.sde.prior_sampling(shape).to(device)
        print(f"Check prior {self.sde_type}: Mean = {x.mean()}, Std = {x.std()}")
        ts = torch.linspace(T,self.eps_time, num_steps, device = device)

        # Shadow ODE-Euler trajectory — same starting noise, different solver
        if _run_cmp:
            x_ode = x.clone()

        k = max(num_steps//100, 1)
        start_time = time.time()

        # Tracks (t_value, global_std, dx_norm) at each logging checkpoint
        std_trajectory = []

        # Running accumulators for cumulative trajectory diagnostics.
        # Updated at every consistency-check step:
        #   _cum_l2_sum : Σ L2_i   — total accumulated local error
        #   _cum_sq_sum : Σ L2_i²  — used to derive running RMS
        #   _n_disc     : number of checks seen so far
        _cum_l2_sum: float = 0.0
        _cum_sq_sum: float = 0.0
        _n_disc:     int   = 0

        # Time discretization from T -> 0
        for i in range(num_steps-1):
            t_i = ts[i].expand(B)
            f, G = rsde.discretize(x, t_i, labels=None)  # f: (B, ...), G: (B,)

            # ---- ODE consistency diagnostic (debug + probability_flow only) ----
            if _run_consistency and (i % k == 0):
                disc = _ode_consistency_discrepancy(rsde, x, t_i, f, ts[i + 1].expand(B))
                disc["step"] = i
                disc["t"]    = t_i[0].item()

                # Cumulative trajectory stability metrics
                _cum_l2_sum += disc["l2"]
                _cum_sq_sum += disc["l2"] ** 2
                _n_disc     += 1
                disc["cum_l2_sum"] = _cum_l2_sum                        # Σ L2_i
                disc["cum_rms"]    = (_cum_sq_sum / _n_disc) ** 0.5     # RMS of L2s seen so far

                disc_out.append(disc)
                print(
                    f"  [consistency | step {i+1:4d}/{num_steps} | t={disc['t']:.4f}]"
                    f"  L2={disc['l2']:.4e}  MAE={disc['mae']:.4e}  max_AE={disc['max_ae']:.4e}"
                    f"  cum_L2={disc['cum_l2_sum']:.4e}  cum_RMS={disc['cum_rms']:.4e}"
                )

            # Shadow ODE: get drift at shadow state (free reuse at i=0 since x_ode == x)
            if _run_cmp:
                f_ode = f if i == 0 else rsde.discretize(x_ode, t_i, labels=None)[0]

            G_b = _expand_batch_vector_to(x, G)

            if probability_flow:
                noise = 0.0
            else:
                noise = torch.randn_like(x)

            if use_heun and i < num_steps - 2:
                x_pred = x - f
                t_next = ts[i + 1].expand(B)
                f2, _  = rsde.discretize(x_pred, t_next, labels=None)
                dx     = -0.5 * (f + f2)
                if _run_cmp:
                    # Pure solver correction: what Heun adds over ODE-Euler at the same state.
                    # Equals 0.5*(f - f2); measures local curvature of the reverse score field.
                    heun_corr = 0.5 * (f - f2)
            else:
                dx = -f + G_b * noise
                if _run_cmp:
                    heun_corr = torch.zeros_like(f)  # last step falls back to Euler

            x  = x + dx

            # ---- ODE-vs-Heun comparison: advance shadow and collect stats ----
            if _run_cmp:
                dx_ode_shad = -f_ode
                x_ode       = x_ode + dx_ode_shad

                upd_diff = (dx - dx_ode_shad).float()
                hcorr_f  = heun_corr.float()

                entry: dict = {
                    "step":     i,
                    "t":        t_i[0].item(),
                    # update comparison (Heun state vs ODE shadow state — solver + state effect)
                    "upd_l1":   upd_diff.abs().mean().item(),
                    "upd_l2":   upd_diff.norm().item(),
                    "upd_max":  upd_diff.abs().max().item(),
                    # pure Heun correction (solver only, evaluated at Heun state)
                    "hcorr_l1": hcorr_f.abs().mean().item(),
                    "hcorr_l2": hcorr_f.norm().item(),
                    "hcorr_max":hcorr_f.abs().max().item(),
                }

                if i % k == 0:
                    td     = (x - x_ode).float()
                    td_abs = td.abs()
                    td_flt = td_abs.flatten()
                    tk     = min(5, td_flt.numel())
                    top_vals, top_flat = torch.topk(td_flt, tk)
                    B_s, K_s, L_s = x.shape
                    top_l  = (top_flat % L_s).cpu().tolist()
                    top_kk = ((top_flat // L_s) % K_s).cpu().tolist()
                    top_b  = (top_flat // (L_s * K_s)).cpu().tolist()
                    # temporal roughness: max and mean of |x_{t+1} - x_t| along sequence
                    if L_s > 1:
                        fd_tensor_h = (x.float()[..., 1:] - x.float()[..., :-1]).abs()
                        fd_tensor_o = (x_ode.float()[..., 1:] - x_ode.float()[..., :-1]).abs()
                        fd_h_max  = fd_tensor_h.max().item()
                        fd_h_mean = fd_tensor_h.mean().item()
                        fd_o_max  = fd_tensor_o.max().item()
                        fd_o_mean = fd_tensor_o.mean().item()
                    else:
                        fd_h_max = fd_h_mean = fd_o_max = fd_o_mean = 0.0
                    entry.update({
                        "traj_l1":    td_abs.mean().item(),
                        "traj_l2":    td.norm().item(),
                        "traj_max":   td_abs.max().item(),
                        "top_vals":   top_vals.cpu().tolist(),
                        "top_idx":    list(zip(top_b, top_kk, top_l)),  # (b, feat, time)
                        "min_diff":   float(x.min()) - float(x_ode.min()),
                        "max_diff":   float(x.max()) - float(x_ode.max()),
                        "fd_heun_max":  fd_h_max,
                        "fd_heun_mean": fd_h_mean,
                        "fd_ode_max":   fd_o_max,
                        "fd_ode_mean":  fd_o_mean,
                    })
                    print(
                        f"  [ODE-vs-Heun | step {i+1:4d}/{num_steps} | t={entry['t']:.4f}]"
                        f"  traj_L2={entry['traj_l2']:.4e}  traj_L1={entry['traj_l1']:.4e}  traj_max={entry['traj_max']:.4e}"
                        f"  hcorr_L2={entry['hcorr_l2']:.4e}  hcorr_L1={entry['hcorr_l1']:.4e}"
                        f"  upd_L2={entry['upd_l2']:.4e}  upd_L1={entry['upd_l1']:.4e}"
                        f"  fd_heun_mean={fd_h_mean:.4e}  fd_ode_mean={fd_o_mean:.4e}  fd_heun_max={fd_h_max:.4e}  fd_ode_max={fd_o_max:.4e}"
                        f"  fd_max_diff={fd_h_max - fd_o_max:+.4e}  fd_mean_diff={fd_h_mean - fd_o_mean:+.4e}"
                    )

                cmp_out.append(entry)

            # ---- log stats every 10% of steps (and the first 15) ----
            if (i % k == 0) or (i > num_steps - 10):
                x_cpu  = x.detach().cpu()
                dx_cpu = dx.detach().cpu() if isinstance(dx, torch.Tensor) else None

                global_std = x_cpu.std().item()
                # per-feature std: shape (K,) — std across batch and time dims
                per_feat_std = x_cpu.std(dim=[0, 2]).tolist()   # (K,)
                dx_norm = (dx_cpu.norm() / dx_cpu.numel()).item() if dx_cpu is not None else 0.0 #Frobenious L2 norm, we divide by numel to get an average per-element update magnitude, which is more interpretable and comparable across different shapes.
                # skewness and excess kurtosis over all elements (batch × features × time)
                xf = x_cpu.flatten()
                mu = xf.mean()
                diff = xf - mu
                var  = (diff ** 2).mean()
                skew_val = ((diff ** 3).mean() / (var ** 1.5 + 1e-8)).item()
                kurt_val = ((diff ** 4).mean() / (var ** 2 + 1e-8) - 3.0).item()  # excess kurtosis

                std_trajectory.append((t_i[0].item(), global_std, dx_norm))

                elapsed_time, start_time = time.time() - start_time, time.time()
                feat_str = "  ".join(f"k{ki}={v:.3f}" for ki, v in enumerate(per_feat_std))
                print(
                    f"[step {i+1:4d}/{num_steps} | t={t_i[0].item():.4f}]  "
                    f"mean={x_cpu.mean():.4f}  std={global_std:.4f}  "
                    f"min={x_cpu.min():.4f}  max={x_cpu.max():.4f}  "
                    f"dx_norm={dx_norm:.2e}\n"
                    f"  skew={skew_val:.4f}  kurt(excess)={kurt_val:.4f}\n"
                    f"  per-feat std: {feat_str}\n"
                    f"  elapsed {elapsed_time:.1f}s\n"
                )

                # ---- inline snapshot plot every 10% checkpoint (not for early steps) ----
                # if i % k == 0:
                #     x_np    = x_cpu.numpy()          # (B, K, L)
                #     n_show  = min(x_cpu.shape[0], 5)
                #     t_label = f"t={t_i[0].item():.3f}  step {i+1}/{num_steps}"

                #     fig, axes = plt.subplots(1, 2, figsize=(12, 3))

                #     # Left: marginal histogram of all values — should narrow and shift as t→0
                #     axes[0].hist(x_np.ravel(), bins=80, density=True, color="steelblue", alpha=0.8)
                #     axes[0].set_title(f"Marginal distribution  ({t_label})")
                #     axes[0].set_xlabel("value")
                #     axes[0].set_ylabel("density")
                #     axes[0].grid(True, linewidth=0.4)

                #     # Right: first-feature trajectories for n_show samples
                #     # shows whether samples are diverse or collapsing to the same path
                #     for b in range(n_show):
                #         alpha = 0.9 if b == 0 else 0.4
                #         lw    = 1.4 if b == 0 else 0.7
                #         axes[1].plot(x_np[b, 0], alpha=alpha, linewidth=lw,
                #                      label=f"s{b}" if b == 0 else None)
                #     axes[1].set_title(f"Sample trajectories feat-0  ({t_label})")
                #     axes[1].set_xlabel("time step")
                #     axes[1].set_ylabel("value")
                #     axes[1].grid(True, linewidth=0.4)

                #     plt.tight_layout()
                #     plt.show()

        if self.enforce_observed:
            x = cond_mask * observed_data + (1.0 - cond_mask) * x

        # ── Post-loop diagnostics ──────────────────────────────────────────────
        x_cpu = x.detach().cpu()          # (B, K, L)
        B_p, K_p, L_p = x_cpu.shape

        # 1. std trajectory: should decrease from sigma_max toward data std
        if len(std_trajectory) > 1:
            t_vals   = [r[0] for r in std_trajectory] #time
            std_vals = [r[1] for r in std_trajectory] #std
            dx_vals  = [r[2] for r in std_trajectory] #norm

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
            ax1.plot(t_vals, std_vals, marker="o", markersize=3)
            ax1.set_ylabel("global std(x)")
            ax1.set_title("Denoising trajectory — std should fall from σ_max toward data std")
            ax1.grid(True, linewidth=0.4)
            ax2.semilogy(t_vals, [max(d, 1e-10) for d in dx_vals], marker="o", markersize=3, color="tab:orange")
            ax2.set_ylabel("||dx|| / numel  (log)")
            ax2.set_xlabel("diffusion time t  (T → ε)")
            ax2.set_title("Update magnitude — should shrink as t → 0")
            ax2.grid(True, linewidth=0.4)
            ax1.invert_xaxis()   # show T on the left, ε on the right (denoising direction)
            plt.tight_layout()
            plt.show()

        # 2. Generated time-series for the first sample (all K features)
        fig, axes = plt.subplots(K_p, 1, figsize=(12, 2 * K_p), sharex=True)
        if K_p == 1:
            axes = [axes]
        n_show = min(B_p, 5)
        for k_idx, ax in enumerate(axes):
            for b in range(n_show):
                alpha = 0.9 if b == 0 else 0.35
                lw    = 1.5 if b == 0 else 0.8
                ax.plot(x_cpu[b, k_idx].numpy(), alpha=alpha, linewidth=lw,
                        label=f"sample {b}" if k_idx == 0 else None)
            ax.set_ylabel(f"feat {k_idx}", fontsize=9)
            ax.grid(True, linewidth=0.4)
        axes[0].legend(fontsize=7, loc="upper right")
        axes[-1].set_xlabel("Time step")
        fig.suptitle("Generated samples — normalised space  (each colour = one sample)", fontsize=11)
        plt.tight_layout()
        plt.show()

        # ── ODE-vs-Heun comparison plots ──────────────────────────────────────────
        if _run_cmp and len(cmp_out) > 0:
            _all_ts    = [e["t"]        for e in cmp_out]
            _upd_l1    = [e["upd_l1"]   for e in cmp_out]
            _hcorr_l1  = [e["hcorr_l1"] for e in cmp_out]

            _ckpts = [e for e in cmp_out if "traj_l1" in e]
            if _ckpts:
                _ck_ts        = [e["t"]            for e in _ckpts]
                _traj_l1      = [e["traj_l1"]      for e in _ckpts]
                _traj_max     = [e["traj_max"]      for e in _ckpts]
                _fd_heun_max  = [e["fd_heun_max"]   for e in _ckpts]
                _fd_heun_mean = [e["fd_heun_mean"]  for e in _ckpts]
                _fd_ode_max   = [e["fd_ode_max"]    for e in _ckpts]
                _fd_ode_mean  = [e["fd_ode_mean"]   for e in _ckpts]

                fig_cmp, axes_cmp = plt.subplots(4, 1, figsize=(10, 12))

                # Panel 1: per-step MAE of update diff vs pure Heun correction
                axes_cmp[0].semilogy(
                    _all_ts, [max(v, 1e-12) for v in _upd_l1],
                    ".", ms=2, color="tomato", label="update diff MAE  (Heun@x_heun − ODE@x_ode)")
                axes_cmp[0].semilogy(
                    _all_ts, [max(v, 1e-12) for v in _hcorr_l1],
                    ".", ms=2, color="steelblue", label="Heun correction MAE  ½‖f−f₂‖  (solver only)")
                axes_cmp[0].set_ylabel("MAE  (log)")
                axes_cmp[0].set_title("Per-step update: ODE vs Heun")
                axes_cmp[0].invert_xaxis()
                axes_cmp[0].legend(fontsize=8)
                axes_cmp[0].grid(True, linewidth=0.4)

                # Panel 2: trajectory divergence between Heun and ODE shadow
                axes_cmp[1].semilogy(
                    _ck_ts, [max(v, 1e-12) for v in _traj_l1],
                    "o-", ms=3, color="steelblue", label="traj MAE  mean|x_Heun − x_ODE|")
                axes_cmp[1].semilogy(
                    _ck_ts, [max(v, 1e-12) for v in _traj_max],
                    "s--", ms=3, color="darkorange", label="traj max-abs")
                axes_cmp[1].set_ylabel("mean|x_Heun − x_ODE|  (log)")
                axes_cmp[1].set_title("Trajectory divergence: Heun vs ODE shadow")
                axes_cmp[1].invert_xaxis()
                axes_cmp[1].legend(fontsize=8)
                axes_cmp[1].grid(True, linewidth=0.4)

                # Panel 3: temporal roughness — max adjacent jump
                axes_cmp[2].plot(_ck_ts, _fd_heun_max, "o-", ms=3, color="steelblue", label="Heun  max|Δx_t|")
                axes_cmp[2].plot(_ck_ts, _fd_ode_max,  "o-", ms=3, color="tomato",    label="ODE   max|Δx_t|")
                axes_cmp[2].set_ylabel("max |x_{t+1} − x_t|")
                axes_cmp[2].set_title("Temporal roughness — worst-case adjacent jump")
                axes_cmp[2].invert_xaxis()
                axes_cmp[2].legend(fontsize=8)
                axes_cmp[2].grid(True, linewidth=0.4)

                # Panel 4: temporal roughness — mean adjacent jump
                axes_cmp[3].plot(_ck_ts, _fd_heun_mean, "o-", ms=3, color="steelblue", label="Heun  mean|Δx_t|")
                axes_cmp[3].plot(_ck_ts, _fd_ode_mean,  "o-", ms=3, color="tomato",    label="ODE   mean|Δx_t|")
                axes_cmp[3].set_ylabel("mean |x_{t+1} − x_t|")
                axes_cmp[3].set_title("Temporal roughness — average adjacent jump")
                axes_cmp[3].invert_xaxis()
                axes_cmp[3].legend(fontsize=8)
                axes_cmp[3].grid(True, linewidth=0.4)

                for _ax in axes_cmp:
                    _ax.set_xlabel("diffusion time t  (T → ε)")
                plt.tight_layout()
                plt.show()

        return x