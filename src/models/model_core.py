# model_core.py
import torch
import torch.nn as nn
from diff_models import diff_CSDI

class CSDIModel(nn.Module):
    """
    Pure neural component:
    - builds side information (time + feature embeddings + optional cond_mask channel)
    - formats inputs for diff_CSDI
    - predicts epsilon (noise) given x_t and conditioning
    """
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # conditional mask channel

        self.embed_layer = nn.Embedding(target_dim, self.emb_feature_dim)

        config_diff = dict(config["diffusion"])
        config_diff["side_dim"] = self.emb_total_dim
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

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
        # for every location (feature k, time l) the model receives: a time embedding, a feature identity embedding, and optionally a conditioning mask value. 
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,E), where E is the embedding time dimensions
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)        # (B,L,K,E) copy time step across all features

        if feature_id is None: #learned embedding for each feature index
            feature_embed = self.embed_layer(torch.arange(self.target_dim, device=self.device))
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,E+E) #what time it is, which variable is this?
        side_info = side_info.permute(0, 3, 2, 1)                  # (B,side_dim,K,L)

        if not self.is_unconditional:
            side_info = torch.cat([side_info, cond_mask.unsqueeze(1)], dim=1)
        return side_info

    def make_diff_input(self, x_t, observed_data, cond_mask):
        if self.is_unconditional:
            return x_t.unsqueeze(1)  # (B,1,K,L), because the network receives only the noisy input
        cond_obs = (cond_mask * observed_data).unsqueeze(1) #Observed part
        noisy_target = ((1 - cond_mask) * x_t).unsqueeze(1) #Target part
        return torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

    def forward(self, x_t, t, observed_data, cond_mask, observed_tp, feature_id=None):
        side_info = self.get_side_info(observed_tp, cond_mask, feature_id=feature_id) #build side info
        diff_in = self.make_diff_input(x_t, observed_data, cond_mask) #produces the model input channels
        return self.diffmodel(diff_in, side_info, t)  # run the denoiser and predict the noise

    @torch.no_grad()
    def predict_eps(self, x_t, t, observed_data, cond_mask, observed_tp, feature_id=None):
        side_info = self.get_side_info(observed_tp, cond_mask, feature_id=feature_id)
        diff_in = self.make_diff_input(x_t, observed_data, cond_mask)
        return self.diffmodel(diff_in, side_info, t)  # (B,K,L)