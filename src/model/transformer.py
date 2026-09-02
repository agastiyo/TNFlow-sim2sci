"""Transformer: per-channel tokens -> query-pooled context vector.

Pre-norm GELU self-attention (FFN width 4·d_model) then learned query pooling; each
spectrum is encoded independently, so any length / wavelength range works.
"""

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Tokens [B, L, d_token] -> context [B, d_model]."""

    def __init__(self, d_token: int, d_model: int = 128, n_head: int = 8,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(d_token, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, 4 * d_model, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model))
        self.pool = QueryPool(d_model=d_model)

    def forward(self, tokens: torch.Tensor, return_weights: bool = False):
        """Args: tokens: Tensor [B, L, d_token]; return_weights: also return pooling weights.

        Returns: context Tensor [B, d_model], or (context, weights [B, L]) if ``return_weights``.
        """
        x = self.embed(tokens)
        x = self.encoder(x)
        return self.pool(x, return_weights=return_weights)


class QueryPool(nn.Module):
    """Learned single-query attention pooling: [B, L, d_model] -> [B, d_model]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, x, return_weights: bool = False):
        """Args: x: Tensor [B, L, d_model]; return_weights: also return pooling weights.

        Returns: pooled Tensor [B, d_model], or (pooled, weights [B, L]) if ``return_weights``.
        """
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.d_model))
        weights = scores.softmax(dim=-1)
        pooled = (weights @ v).squeeze(1)
        if return_weights:
            return pooled, weights.squeeze(1)
        return pooled
