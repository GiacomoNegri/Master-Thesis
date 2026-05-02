import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer

# C: Transformer builders
# where we take the TransformerEncoder and then the LinearAttentionTransformer
# Transformer encoder: channels = 64, heads = 8, hidden size = 64, activation = GELU, layers = 1
# Linear Attention: dim = channels, depth = layers, heads = heads, max_seq_len = 256, n_local_attn_heads = 0 disabled

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels *4 , activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers) #multiplied by dim_feedforward = channels * 4 as in transformers

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

# C: create a 1D convolution and initializes weights with Kaiming/He normal initialization

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

# encode diffusion step t

class DiffusionEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, T]
        # Same frequency grid as the original lookup table, evaluated at arbitrary float t.
        half_dim = self.embedding_dim // 2
        frequencies = 10.0 ** (
            torch.arange(half_dim, device=t.device) / (half_dim - 1) * 4.0
        )  # (half_dim,) — log-spaced from 10^0 to 10^4
        args = t.unsqueeze(1) * frequencies.unsqueeze(0)  # (B, half_dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, embedding_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float — continuous SDE time, no discretisation needed
        x = self._sinusoidal_embedding(t)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # self.output_projection2 = Conv1d_with_init(self.channels, 1, 1) # removed to match paper
        # 0-initialization as in the papaer
        # nn.init.zeros_(self.output_projection2.weight)
        # Normal initalization: to avoid slow initial learning
        nn.init.normal_(self.output_projection1.bias, mean=0.0, std=0.01)

        # Stack of residual layers
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        # EDM EDITING
        # x = F.relu(x) # also removed from the diagnostic 
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # layer is an element of self.residual_layer: it applies one residual block
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        
        # EDM EDITING
        # x = F.relu(x) # Not present in the paper and removed as in 'diagnostic'
        # x = self.output_projection2(x)  # (B,1,K*L) not present in the paper
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    @staticmethod
    def _sinusoidal_pe(length, channels, device):
        """Fixed sinusoidal positional encoding: (1, channels, length) for broadcast add."""
        pe = torch.zeros(length, channels, device=device)
        pos = torch.arange(length, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, channels, 2, device=device).float()
            * (-math.log(10000.0) / channels)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.permute(1, 0).unsqueeze(0)  # (1, channels, length)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        # Positional encoding injected prior to the Transformer (relative sequence position)
        y = y + self._sinusoidal_pe(L, channel, y.device)  # (BK, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)

        # Positional encoding injected prior to the Transformer (relative feature position)
        y = y + self._sinusoidal_pe(K, channel, y.device)  # (BL, channel, K)

        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L) # Flattening

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L), mixing information

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y) #mapping back to two channels, before chunking

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip #sqrt(2) is the control variance growth across layers
