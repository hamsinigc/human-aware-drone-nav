"""
Attention-based Trajectory Predictor
-------------------------------------
Two attention mechanisms, as required by the project brief:

1. TEMPORAL attention: a standard Transformer encoder-decoder over each
   agent's own time sequence (captures motion patterns: speed, turning,
   stopping).
2. SOCIAL attention: a multi-head attention layer applied ACROSS agents at
   each timestep (captures interactions: avoidance, group following, etc).

The two are stacked: social attention first mixes information between
agents at each frame, then temporal attention (a seq2seq Transformer)
predicts the future for each agent using its (socially-aware) history.

Input:  past positions of all agents in a scene, shape (B, N, T_obs, 2)
Output: predicted future positions, shape (B, N, T_pred, 2)
    B = batch (scenes), N = number of agents in scene, 2 = (x, y)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for the time dimension."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B*N, T, d_model)
        return x + self.pe[:, : x.size(1), :]


class SocialAttention(nn.Module):
    """
    Multi-head attention ACROSS agents at a single timestep.
    For each timestep t, every agent attends to every other agent present
    in the scene at t. This is where "who reacts to whom" is learned.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, agent_mask=None):
        # x: (B, N, T, d_model) -> process each timestep as its own
        # "sequence over agents"
        B, N, T, D = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(B * T, N, D)  # (B*T, N, D)

        key_padding_mask = None
        if agent_mask is not None:
            # agent_mask: (B, N) True = real agent, False = padding
            key_padding_mask = (~agent_mask).unsqueeze(1).expand(B, T, N)
            key_padding_mask = key_padding_mask.reshape(B * T, N)

        attn_out, _ = self.attn(
            x_t, x_t, x_t, key_padding_mask=key_padding_mask, need_weights=False
        )
        attn_out = self.norm(x_t + attn_out)
        attn_out = attn_out.reshape(B, T, N, D).permute(0, 2, 1, 3)  # (B, N, T, D)
        return attn_out


class TemporalTransformer(nn.Module):
    """Encoder-decoder Transformer over the time dimension, per agent."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        pred_len: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec_layers)

        # learned "start of sequence" query tokens for the decoder
        self.query_embed = nn.Parameter(torch.randn(1, pred_len, d_model))

    def forward(self, x):
        # x: (B*N, T_obs, d_model)
        x = self.pos_enc(x)
        memory = self.encoder(x)

        tgt = self.query_embed.expand(x.size(0), -1, -1)
        tgt = self.pos_enc(tgt)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.pred_len
        ).to(x.device)

        out = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return out  # (B*N, T_pred, d_model)


class TrajectoryTransformer(nn.Module):
    """
    Full model: input projection -> social attention -> temporal
    transformer -> output projection to (x, y).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        obs_len: int = 8,
        pred_len: int = 12,
        n_social_layers: int = 1,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.input_proj = nn.Linear(2, d_model)
        self.social_layers = nn.ModuleList(
            [SocialAttention(d_model, n_heads, dropout) for _ in range(n_social_layers)]
        )
        self.temporal = TemporalTransformer(
            d_model, n_heads, n_enc_layers, n_dec_layers, dim_ff, dropout, pred_len
        )
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, obs_traj, agent_mask=None):
        """
        obs_traj: (B, N, T_obs, 2) raw (x, y) positions, ideally already
                  normalized/relative (see dataset.py).
        agent_mask: (B, N) bool, True where agent slot is real (not padding)
        returns: (B, N, T_pred, 2) predicted future positions
        """
        B, N, T, _ = obs_traj.shape
        x = self.input_proj(obs_traj)  # (B, N, T, d_model)

        for layer in self.social_layers:
            x = layer(x, agent_mask)

        x = x.reshape(B * N, T, -1)
        out = self.temporal(x)  # (B*N, T_pred, d_model)
        out = self.output_proj(out)  # (B*N, T_pred, 2)
        out = out.reshape(B, N, self.pred_len, 2)
        return out


if __name__ == "__main__":
    # quick smoke test
    model = TrajectoryTransformer(obs_len=8, pred_len=12)
    dummy = torch.randn(4, 5, 8, 2)  # 4 scenes, 5 agents, 8 obs steps, xy
    mask = torch.ones(4, 5, dtype=torch.bool)
    out = model(dummy, mask)
    print("Output shape:", out.shape)  # expect (4, 5, 12, 2)
