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
    dt_actual: float,
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
        rsde:       Reverse SDE (must expose .discretize(x, t, labels=None)).
        x:          Current state, shape (B, K, L).
        t_i:        Current time vector, shape (B,).
        f_full:     Pre-computed full-step drift f = drift(x, t) * dt_sde, same shape as x.
        dt_actual:  Actual time-grid step size (T → eps direction) used to place t_mid.

    Returns:
        dict with float keys "l2", "mae", "max_ae".
    """
    # Full step (reuse already-computed f)
    x_full = x - f_full

    # First half step — scale f_full, no model call
    x_mid = x - f_full * 0.5

    # Second half step — one extra model call at the midpoint time
    t_mid = (t_i - dt_actual * 0.5).clamp(min=0.0)
    f2, _ = rsde.discretize(x_mid, t_mid, labels=None)
    x_2h = x_mid - f2 * 0.5

    diff = (x_full - x_2h).float()
    return {
        "l2":     diff.norm().item(),
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
                # Importance sampling: draw one index per batch element from the IS
                # distribution, then look up the corresponding t value.
                indices = torch.multinomial(
                    t_probs.unsqueeze(0).expand(B, -1),
                    num_samples=1,
                    replacement=True,
                ).squeeze(-1)  # (B,)
                t = t_grid[indices]
            else:
                # Default: uniform sampling over [eps_time, T]
                t = self.eps_time + torch.rand(B, device=device) * (self.sde.T - self.eps_time)


        # Get closed-form mean and std of p_t(z | z0)
        mean, std = self.sde.marginal_prob(x0, t)  # mean: (B, ...), std: (B,)

        # Sample noise
        eps = torch.randn_like(x0)

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

        # Initialize from the prior at time T
        x = self.sde.prior_sampling(shape).to(device)
        print(f"Check prior {self.sde_type}: Mean = {x.mean()}, Std = {x.std()}")
        ts = torch.linspace(T,self.eps_time, num_steps, device = device)

        # Step size for the actual time grid (used by the consistency diagnostic)
        dt_actual = float((T - self.eps_time) / max(num_steps - 1, 1))

        k = max(num_steps//10, 1)
        start_time = time.time()

        # Tracks (t_value, global_std, dx_norm) at each logging checkpoint
        std_trajectory = []

        # Whether to run the ODE consistency check this run
        _run_consistency = debug and probability_flow and (disc_out is not None)

        # Time discretization from T -> 0
        for i in range(num_steps-1):
            t_i = ts[i].expand(B)
            f, G = rsde.discretize(x, t_i, labels=None)  # f: (B, ...), G: (B,)

            # ---- ODE consistency diagnostic (debug + probability_flow only) ----
            if _run_consistency and (i % k == 0):
                disc = _ode_consistency_discrepancy(rsde, x, t_i, f, dt_actual)
                disc["step"] = i
                disc["t"]    = t_i[0].item()
                disc_out.append(disc)
                print(
                    f"  [consistency | step {i+1:4d}/{num_steps} | t={disc['t']:.4f}]"
                    f"  L2={disc['l2']:.4e}  MAE={disc['mae']:.4e}  max_AE={disc['max_ae']:.4e}"
                )

            G_b = _expand_batch_vector_to(x, G)

            if probability_flow:# or i == 0:
                noise = 0.0
            else:
                noise = torch.randn_like(x)

            dx = -f + G_b * noise
            x  = x + dx

            # ---- log stats every 10% of steps (and the first 15) ----
            if (i % k == 0) or (i > num_steps - 10):
                x_cpu  = x.detach().cpu()
                dx_cpu = dx.detach().cpu() if isinstance(dx, torch.Tensor) else None

                global_std = x_cpu.std().item()
                # per-feature std: shape (K,) — std across batch and time dims
                per_feat_std = x_cpu.std(dim=[0, 2]).tolist()   # (K,)
                dx_norm = (dx_cpu.norm() / dx_cpu.numel()).item() if dx_cpu is not None else 0.0

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
                if i % k == 0:
                    x_np    = x_cpu.numpy()          # (B, K, L)
                    n_show  = min(x_cpu.shape[0], 5)
                    t_label = f"t={t_i[0].item():.3f}  step {i+1}/{num_steps}"

                    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

                    # Left: marginal histogram of all values — should narrow and shift as t→0
                    axes[0].hist(x_np.ravel(), bins=80, density=True, color="steelblue", alpha=0.8)
                    axes[0].set_title(f"Marginal distribution  ({t_label})")
                    axes[0].set_xlabel("value")
                    axes[0].set_ylabel("density")
                    axes[0].grid(True, linewidth=0.4)

                    # Right: first-feature trajectories for n_show samples
                    # shows whether samples are diverse or collapsing to the same path
                    for b in range(n_show):
                        alpha = 0.9 if b == 0 else 0.4
                        lw    = 1.4 if b == 0 else 0.7
                        axes[1].plot(x_np[b, 0], alpha=alpha, linewidth=lw,
                                     label=f"s{b}" if b == 0 else None)
                    axes[1].set_title(f"Sample trajectories feat-0  ({t_label})")
                    axes[1].set_xlabel("time step")
                    axes[1].set_ylabel("value")
                    axes[1].grid(True, linewidth=0.4)

                    plt.tight_layout()
                    plt.show()

        if self.enforce_observed:
            x = cond_mask * observed_data + (1.0 - cond_mask) * x

        # ── Post-loop diagnostics ──────────────────────────────────────────────
        x_cpu = x.detach().cpu()          # (B, K, L)
        B_p, K_p, L_p = x_cpu.shape

        # 1. std trajectory: should decrease from sigma_max toward data std
        if len(std_trajectory) > 1:
            t_vals   = [r[0] for r in std_trajectory]
            std_vals = [r[1] for r in std_trajectory]
            dx_vals  = [r[2] for r in std_trajectory]

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

        return x