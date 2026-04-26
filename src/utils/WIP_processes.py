import torch
import torch.nn as nn

import time
import matplotlib.pyplot as plt

from src.utils.WIP_SDE import SDE, BetaScheduleSDE, SigmaSchedule, SubVPSDE, VESDE, VPSDE, GBMLogSDE


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
            EDM EDITING
            Score function using the EDM denoiser interface.
            sigma is derived from the SDE marginal at time t and passed directly to
            model.denoise(); the score is then recovered via Tweedie's formula for VE:
                ∇_x log p_t(x) = (D_x - x) / sigma^2
            where D_x = model.denoise(x, sigma, ...) approximates E[x0 | x_t].
            """
            # sigma must be derived before the model call: denoise() takes sigma as input.
            _, sigma = self.sde.marginal_prob(x, t)   # sigma: (B,)

            D_x = model.denoise(
                x_t=x,
                sigma=sigma,
                observed_data=observed_data,
                cond_mask=cond_mask,
                observed_tp=observed_tp,
            )

            # Tweedie's formula: score = (E[x0|x] - x) / sigma^2.
            # Denominator is sigma^2 (not sigma) because D_x is in data space, not noise space.
            sigma_b = _expand_batch_vector_to(x, sigma)
            score = (D_x - x) / (sigma_b ** 2 + 1e-12)
            return score

        # Build reverse-time SDE/ODE
        rsde: SDE = self.sde.reverse(score_fn, probability_flow=probability_flow)

        # Initialize from the prior at time T
        x = self.sde.prior_sampling(shape).to(device)
        print(f"Check prior {self.sde_type}: Mean = {x.mean()}, Std = {x.std()}")
        ts = torch.linspace(T,self.eps_time, num_steps, device = device)

        k = max(num_steps//10, 1)
        start_time = time.time()

        # Tracks (t_value, global_std, dx_norm) at each logging checkpoint
        std_trajectory = []

        # Time discretization from T -> 0
        for i in range(num_steps):
            t_i = ts[i].expand(B)
            f, G = rsde.discretize(x, t_i, labels=None)  # f: (B, ...), G: (B,)

            G_b = _expand_batch_vector_to(x, G)

            if probability_flow or i == 0:
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

    # ------------------------------------------------------------------
    # EDM deterministic Heun sampler  (Karras et al. 2022, Algorithm 2)
    # EDM EDITING
    # ------------------------------------------------------------------

    @torch.no_grad()
    def edm_sampler(
        self,
        model: nn.Module,
        shape,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_tp: torch.Tensor,
        num_steps: int = 100,
        rho: float = 3.0,
        sigma_min: float = None,
        sigma_max: float = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        EDM deterministic Heun sampler (S_churn=0, purely deterministic).

        Implements Algorithm 2 from Karras et al. 2022, adapted for the
        conditional CSDI setting (B, K, L) with fixed OHL context.

        Calls model.denoise(x_t, sigma, observed_data, cond_mask, observed_tp)
        directly — no score function, no SDE reverse-time drift, no time-grid
        conversion.  The sigma schedule is ρ-power spaced, which concentrates
        steps at low sigma where the denoising is most sensitive.

        Args:
            model:         CSDIModel with EDM interface (Phase 1 forward/denoise).
            shape:         (B, K, L) — shape of samples to generate.
            observed_data: (B, K, L) — conditioning context (OHL channels).
            cond_mask:     (B, K, L) — 1=conditioning, 0=target (Close).
            observed_tp:   (B, L)    — absolute time-series positions.
            num_steps:     Number of denoising steps.  18 is the EDM paper default;
                           50–100 gives higher quality at moderate cost.
            rho:           Schedule curvature.  7.0 is the EDM paper default.
            sigma_min:     Minimum sigma.  Defaults to self.sde.sigma_schedule.sigma_min.
                           NOTE: sigma_schedule only exists on VESDE / GBMLogSDE.
                           For VP/subVP SDEs this argument must be supplied explicitly.
            sigma_max:     Maximum sigma.  Same note as sigma_min.
            device:        Inference device.  Inferred from model if None.

        Returns:
            x: (B, K, L) generated samples, with OHL channels replaced by
               observed_data if self.enforce_observed is True.
        """
        assert num_steps >= 2, "edm_sampler requires num_steps >= 2"

        if device is None:
            device = next(model.parameters()).device

        # Read sigma bounds from the SDE schedule when not provided explicitly.
        # This ensures the sampler uses the same sigma range as training.
        if sigma_min is None:
            sigma_min = float(self.sde.sigma_schedule.sigma_min)
        if sigma_max is None:
            sigma_max = float(self.sde.sigma_schedule.sigma_max)

        B = shape[0]

        # ── 1. Sigma schedule: ρ-power spacing (Karras et al. 2022, Eq. 5) ──────
        # sigma decreases from sigma_max (i=0) to sigma_min (i=num_steps-1).
        # A final sigma_N = 0 is appended so the last Euler step outputs D_x exactly:
        #   x_next = x + (0 - sigma_min) * d_cur = x - (x - D_x) = D_x
        step_idx = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            sigma_max ** (1.0 / rho)
            + step_idx / (num_steps - 1)
            * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho                                          # (num_steps,)  decreasing
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)]) # append 0 → (num_steps+1,)

        # ── 2. Initialize from the prior: x ~ N(0, sigma_max² I) ─────────────────
        x = torch.randn(*shape, device=device) * sigmas[0]  # (B, K, L)

        # ── 3. Heun loop ──────────────────────────────────────────────────────────
        for i in range(num_steps):
            sigma_cur  = sigmas[i]                             # scalar tensor
            sigma_next = sigmas[i + 1]                         # scalar tensor (0 at last step)

            # Broadcast scalar sigma to (B,) as required by model.denoise
            sigma_cur_b = sigma_cur.reshape(1).expand(B)       # (B,)

            # ── Euler step ────────────────────────────────────────────────────────
            # D_x ≈ E[x0 | x_t, sigma]: the model's denoised estimate at sigma_cur.
            D_x_cur = model.denoise(
                x_t=x,
                sigma=sigma_cur_b,
                observed_data=observed_data,
                cond_mask=cond_mask,
                observed_tp=observed_tp,
            )                                                   # (B, K, L)

            # Probability-flow ODE direction: d = (x - D_x) / sigma
            d_cur  = (x - D_x_cur) / sigma_cur
            x_next = x + (sigma_next - sigma_cur) * d_cur

            # ── 2nd-order Heun correction ─────────────────────────────────────────
            # Skipped at the last step where sigma_next = 0 to avoid log(0) and
            # division-by-zero inside the preconditioning.  The Euler step at that
            # final step already produces x_next = D_x_cur exactly.
            if i < num_steps - 1:
                sigma_next_b = sigma_next.reshape(1).expand(B)  # (B,)
                D_x_next = model.denoise(
                    x_t=x_next,
                    sigma=sigma_next_b,
                    observed_data=observed_data,
                    cond_mask=cond_mask,
                    observed_tp=observed_tp,
                )                                               # (B, K, L)
                d_prime = (x_next - D_x_next) / sigma_next
                x_next  = x + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_prime)

            x = x_next

        # ── 4. Enforce conditioning: replace OHL with ground truth ────────────────
        if self.enforce_observed:
            x = cond_mask * observed_data + (1.0 - cond_mask) * x

        return x