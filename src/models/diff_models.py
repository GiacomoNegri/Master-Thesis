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
        d_model=channels, nhead=heads,
        # [CHANGE 3] dim_feedforward raised from channels (1×) to 4×channels.
        # Original CSDI used 1×, making the FFN atypically narrow (33% of layer params).
        # Standard Transformer convention is 4×d_model; this restores that proportion
        # and accounts for the parameter budget freed by removing the dormant feature_layer.
        dim_feedforward=4 * channels,
        activation="gelu",
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

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
        # [CHANGE 1] Removed output_projection1 Conv1d(128→128) to match the paper's single final
        # projection. Paper text: "a final 1D convolutional layer"; figure labels the output step
        # as (C,L)→(1,L) in one step. Two sequential linear convs without a non-linearity are
        # equivalent to one; the ReLU that sat between them was the true mismatch (now gone).
        self.output_projection = Conv1d_with_init(self.channels, 1, 1)
        # Small-normal init retained from the original two-stage head: avoids the zero-output
        # trap of the vanilla CSDI initialisation (nn.init.zeros_) while keeping early
        # predictions near zero to prevent large initial loss spikes.
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.01)


        # Stack of residual layers
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    # [CHANGE 4] Propagate flag from config; defaults True so existing
                    # callers that do not set the key retain original behaviour.
                    use_feature_layer=config.get("use_feature_layer", True),
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        # [CHANGE 2] Removed F.relu here: paper figure shows Conv1d → ⊕ with no intermediate
        # activation. A ReLU at this point asymmetrically clips the latent representation before
        # any diffusion-step information is injected, biasing the signal away from zero despite
        # the target noise ε being zero-mean. Non-linearity enters naturally through the gated
        # activation (sigmoid * tanh) inside each ResidualBlock.
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # layer is an element of self.residual_layer: it applies one residual block
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers)) # this division is used to keep the variance of the sum constant regardless of depth
        x = x.reshape(B, self.channels, K * L)
        # [CHANGE 1] Single output projection replacing the former two-stage head
        # (output_projection1 Conv1d(128→128) + ReLU + output_projection2 Conv1d(128→1)).
        # Removes 16,512 parameters and the intermediate non-linearity.
        x = self.output_projection(x)   # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    # [CHANGE 4] Added use_feature_layer flag (default True for backward compatibility).
    # When False the feature_layer Transformer is never instantiated, saving ~99K params
    # per block. Set to False by CSDIModel when target_dim == 1, since forward_feature
    # returns immediately at K=1 and the weights are otherwise loaded but never executed.
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False,
                 use_feature_layer=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        self.use_feature_layer = use_feature_layer  # False → feature_layer not created
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            if use_feature_layer:
                self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            if use_feature_layer:
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
        # y: (B, channel, K*L) → (B, channel, K, L) → (B, K, channel, L) → (B*K, channel, L)
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
        # [CHANGE 4] Guard replaced: was a runtime K==1 check; now uses the static
        # use_feature_layer flag set at construction time from target_dim. When False,
        # feature_layer was never instantiated so we must return before accessing it.
        if not self.use_feature_layer:
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

        gate, filter = torch.chunk(y, 2, dim=1) # slipt into two parts for gated activation, each of shape (B,channel,K*L)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y) #mapping back to two channels, before chunking (B,2*channel,K*L)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip #sqrt(2) is the control variance growth across layers
