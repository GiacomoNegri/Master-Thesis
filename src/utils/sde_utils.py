import torch
from .WIP_processes import Diffusion_Processes
from .WIP_SDE import VESDE, SubVPSDE, VPSDE


def calculate_importance_sampling_probabilities(sde, N_timesteps: int, device, eps_time: float = 1e-3):
    """
    Compute IS probabilities p(t) ∝ g(t)² / σ²(t) on a uniform grid of N_timesteps points.

    Theory: the score-matching loss under importance sampling with q(t) ∝ g(t)²/σ²(t)
    becomes equivalent to plain unweighted MSE, but with training mass concentrated on
    small-t (hard) timesteps where σ(t) is small and the model prediction matters most.

    Works with any SDE that implements the standard interface (sde() and marginal_prob()),
    which covers VPSDE, SubVPSDE, VESDE, and GBMLogSDE.

    Args:
        sde:          An instantiated SDE object (e.g. VPSDE, VESDE, GBMLogSDE).
        N_timesteps:  Number of grid points to discretize [eps_time, T].
        device:       Torch device.
        eps_time:     Lower bound on t (same eps used in forward_process).

    Returns:
        probabilities: (N_timesteps,) normalized probability vector summing to 1.
        t_grid:        (N_timesteps,) tensor of the corresponding t values.
    """
    epsilon = 1e-8  # numerical safety for division

    t_grid = torch.linspace(eps_time, sde.T - epsilon, N_timesteps, device=device)

    # Use a dummy x of shape (N_timesteps, 1, 1) so sde() and marginal_prob() can
    # broadcast correctly — we only need the diffusion coefficient and std, not the
    # data-dependent drift.
    dummy_x = torch.zeros(N_timesteps, 1, 1, device=device)

    with torch.no_grad():
        _, g_t = sde.sde(dummy_x, t_grid)          # g_t: (N_timesteps,)
        _, sigma_t = sde.marginal_prob(dummy_x, t_grid)  # sigma_t: (N_timesteps,)

    g_sq = g_t ** 2
    sigma_sq = sigma_t ** 2 + epsilon

    weights = g_sq / sigma_sq
    probabilities = weights / weights.sum()

    return probabilities, t_grid
