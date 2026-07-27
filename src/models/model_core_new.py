# model_core.py
import torch
import torch.nn as nn
from .diff_models import diff_CSDI

class CSDIModel(nn.Module):
    """
    Pure neural component:
    - builds side information (time + feature embeddings + optional cond_mask channel)
    - formats inputs for diff_CSDI
    - exposes an EDM-style denoiser interface: D_x = c_skip*x + c_out*F_x

    Phase-1 EDM integration:
    - forward() and denoise() take sigma (noise level) instead of t (SDE time)
    - the backbone receives c_in-scaled noisy input and c_noise as the diffusion step
    - the output is the EDM denoised estimate D_x, not epsilon

    Legacy:
    - predict_eps() is preserved unchanged for the current epsilon-based sampling path;
      it will be replaced in Phase 2 when sampling is rewritten.
    """
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        # Empirical std of the (normalised) training data.
        # Used by EDM preconditioning to balance skip-connection and output scales.
        # Must be calibrated to the actual data distribution; 0.5 is a safe default
        # for window-normalised log-returns (which are approx. unit-variance but
        # c_skip/c_out remain well-conditioned for sigma_data in [0.1, 1.0]).
        # EDM EDITING
        self.sigma_data = float(config["model"].get("sigma_data", 0.5))

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # conditional mask channel

        self.embed_layer = nn.Embedding(target_dim, self.emb_feature_dim)
        # Learnable linear applied on top of the sinusoidal basis, making the
        # time embedding a combination of sinusoidal functions and learnable vectors.
        self.time_embed_proj = nn.Linear(self.emb_time_dim, self.emb_time_dim)

        config_diff = dict(config["diffusion"])
        config_diff["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

    # ------------------------------------------------------------------
    # EDM preconditioning  (Karras et al. 2022, Eq. 7)
    # EDM EDITING
    # ------------------------------------------------------------------

    def _edm_precond(self, sigma: torch.Tensor):
        """
        Compute EDM preconditioning scalars from noise level sigma.

        Args:
            sigma: (B,) per-sample noise standard deviation.

        Returns:
            c_skip:  (B,1,1)  skip-connection scale
            c_out:   (B,1,1)  output scale
            c_in:    (B,1,1)  input scale (applied to the noisy target before the backbone)
            c_noise: (B,)     noise-level embedding fed to DiffusionEmbedding
        """
        sd2   = self.sigma_data ** 2
        sigma2 = sigma ** 2
        denom  = (sigma2 + sd2).sqrt()          # (B,)

        c_skip  = (sd2        / (sigma2 + sd2)).view(-1, 1, 1)   # (B,1,1)
        c_out   = (sigma * self.sigma_data / denom).view(-1, 1, 1)  # (B,1,1)
        c_in    = (1.0 / denom).view(-1, 1, 1)                   # (B,1,1)
        c_noise = sigma.log() / 4.0                               # (B,)
        return c_skip, c_out, c_in, c_noise

    # ------------------------------------------------------------------
    # Side information & input formatting  (unchanged CSDI pipeline)
    # ------------------------------------------------------------------

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask, feature_id=None):
        # For every location (feature k, time l) the model receives:
        # a time embedding (sinusoidal + learnable), a feature identity embedding,
        # and optionally a conditioning mask value.
        # NOTE: the positional encoding (relative sequence index) is injected directly
        # inside ResidualBlock.forward_time/_forward_feature, prior to the Transformer.
        B, K, L = cond_mask.shape

        # Time embedding: sinusoidal basis → learnable projection (absolute temporal positions)
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,E)
        time_embed = self.time_embed_proj(time_embed)                      # (B,L,E) learnable
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)        # (B,L,K,E)

        if feature_id is None:  # learned embedding for each feature index
            feature_embed = self.embed_layer(torch.arange(self.target_dim, device=self.device))
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,E+F)
        side_info = side_info.permute(0, 3, 2, 1)                   # (B,side_dim,K,L)

        if not self.is_unconditional:
            side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1) # we are appending one more channel for the conditioning mask
        return side_info

    def make_diff_input(self, x_t, observed_data, cond_mask):
        if self.is_unconditional:
            return x_t.unsqueeze(1)  # (B,1,K,L), because the network receives only the noisy input
        #EDM EDITING
        cond_obs = (cond_mask * observed_data).unsqueeze(1)        # observed part  (clean)
        noisy_target = ((1 - cond_mask) * x_t).unsqueeze(1)       # target part    (c_in-scaled)
        return torch.cat([cond_obs, noisy_target], dim=1)          # (B,2,K,L)

    # ------------------------------------------------------------------
    # EDM forward pass  (Phase-1 primary interface)
    # EDM EDITING
    # ------------------------------------------------------------------

    def forward(self, x_t, sigma, observed_data, cond_mask, observed_tp, feature_id=None):
        """
        EDM-style denoiser forward pass.

        Args:
            x_t:           (B, K, L)  noisy input at noise level sigma
            sigma:         (B,)       per-sample noise standard deviation
            observed_data: (B, K, L)  clean conditioning context (O/H/L)
            cond_mask:     (B, K, L)  1 = observed/conditioning, 0 = target
            observed_tp:   (B, L)     absolute time positions
            feature_id:    optional feature indices

        Returns:
            D_x: (B, K, L)  EDM denoised estimate  D_x = c_skip*x_t + c_out*F_x
                            Meaningful on the target channel; conditioning channels
                            carry no loss signal and are masked out by target_mask.
        """
        c_skip, c_out, c_in, c_noise = self._edm_precond(sigma)

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id=feature_id)
        # Pass c_in-scaled noisy input; the clean conditioning channel in make_diff_input
        # is drawn from observed_data and is not scaled by c_in.
        diff_in = self.make_diff_input(c_in * x_t, observed_data, cond_mask)
        F_x = self.diffmodel(diff_in, side_info, c_noise)  # (B,K,L)  backbone output

        return c_skip * x_t + c_out * F_x                  # D_x: (B,K,L)

    @torch.no_grad()
    def denoise(self, x_t, sigma, observed_data, cond_mask, observed_tp, feature_id=None):
        """
        EDM-style denoiser for inference (no-grad wrapper of forward).
        Replaces predict_eps() in Phase-2 sampling rewrite.
        """
        return self.forward(x_t, sigma, observed_data, cond_mask, observed_tp, feature_id=feature_id)