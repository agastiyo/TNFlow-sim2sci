"""TNFlow: spectrum -> context vector -> joint posterior over composition and grain size.

Model      spectrum [L] -> tokens [L, d_token] -> context [d_model] -> K-simplex posterior.
Sampling   one draw = composition [K] + grain [K] µm (NaN where that draw lacks the component).
           n draws  = (n, K) compositions and their matching (n, K) grain sizes.

    simp = model.simplex(spec, lams)     # encode once
    comp, grain = simp.sample(2000)      # (2000, K), (2000, K) tensors; resample freely

Everything stays on the model's device. Summaries live in src/model/posterior.py.
"""

import numpy as np
import torch
import torch.nn as nn

from src.data_utils.components import K
from src.model.tokenizers import NormTokenizer, SlopeBandTokenizer
from src.model.transformer import Transformer
from src.model.cnf import CNF, free_to_full_simplex
from src.model.grain import GrainFlow
from src.model.posterior import DEFAULT_SEED

_TRANSFORMER_PRESETS = {
    "small":  dict(d_model=64,  n_head=4,  n_layers=2),
    "medium": dict(d_model=128, n_head=8,  n_layers=4),
    "large":  dict(d_model=256, n_head=16, n_layers=8),
}
_FLOW_PRESETS = {
    "small":  dict(transforms=2, bins=8,  hidden=(64, 64)),
    "medium": dict(transforms=4, bins=16, hidden=(128, 128)),
    "large":  dict(transforms=8, bins=32, hidden=(256, 256)),
}
_TOKENIZERS = {"norm": NormTokenizer, "slope_band": SlopeBandTokenizer}

DROPOUT = 0.1
COMP_SCALE = 2.5        # pre-shrink keeping stick-breaking preimages inside the NSF spline bound
GRAIN_EMBED_DIM = 16


def _preset(table: dict, size: str, what: str) -> dict:
    try:
        return table[size]
    except KeyError:
        raise ValueError(f"{what} must be one of {list(table)}, got {size!r}")


class TNFlow(nn.Module):
    """Tokenizer -> transformer -> {composition CNF, grain flow}, sized by presets."""

    def __init__(self, transformer_size: str = "medium", comp_flow_size: str = "large",
                 grain_flow_size: str = "small", tokenizer: str = "norm",
                 log_grain_mean: float = 0.0, log_grain_std: float = 1.0):
        super().__init__()
        self.config = dict(
            transformer_size=transformer_size, comp_flow_size=comp_flow_size,
            grain_flow_size=grain_flow_size, tokenizer=tokenizer,
            log_grain_mean=log_grain_mean, log_grain_std=log_grain_std,
        )
        tf = _preset(_TRANSFORMER_PRESETS, transformer_size, "transformer_size")
        comp = _preset(_FLOW_PRESETS, comp_flow_size, "comp_flow_size")
        grain = _preset(_FLOW_PRESETS, grain_flow_size, "grain_flow_size")
        d_model = tf["d_model"]

        self.tokenizer = _preset(_TOKENIZERS, tokenizer, "tokenizer")()
        self.transformer = Transformer(self.tokenizer.d_token, d_model, tf["n_head"],
                                       tf["n_layers"], DROPOUT)
        self.cnf = CNF(features=K - 1, context=d_model, scale=COMP_SCALE,
                       transforms=comp["transforms"], bins=comp["bins"],
                       hidden_features=comp["hidden"])
        self.grain = GrainFlow(context=d_model, embed_dim=GRAIN_EMBED_DIM,
                               transforms=grain["transforms"], bins=grain["bins"],
                               hidden_features=grain["hidden"],
                               log_grain_mean=log_grain_mean, log_grain_std=log_grain_std)

    def encode(self, lam, R):
        """lam, R: Tensor [B, L] -> context Tensor [B, d_model]."""
        return self.transformer(self.tokenizer(lam, R))

    def _as_batch(self, spec, lams):
        """spec, lams: ndarray [L] or [B, L] -> (R, lam) Tensor [B, L] on the model device."""
        device = next(self.parameters()).device
        R = torch.as_tensor(np.asarray(spec), dtype=torch.float32)
        lam = torch.as_tensor(np.asarray(lams), dtype=torch.float32)
        if R.ndim == 1:
            R, lam = R.unsqueeze(0), lam.unsqueeze(0)
        return R.to(device), lam.to(device)

    def simplex(self, spec, lams, seed=DEFAULT_SEED):
        """spec, lams: ndarray|Tensor [L] -> SimplexPosterior (runs the encoder once)."""
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        R, lam = self._as_batch(spec, lams)
        with torch.no_grad():
            context = self.encode(lam, R)
        return SimplexPosterior(self, context)

    def attention(self, spec, lams):
        """spec, lams: ndarray [L] -> (lam [L], query-pooling weights [L] summing to 1)."""
        self.eval()
        R, lam = self._as_batch(spec, lams)
        with torch.no_grad():
            _, weights = self.transformer(self.tokenizer(lam, R), return_weights=True)
        return lam.squeeze(0).cpu().numpy(), weights.squeeze(0).cpu().numpy()


class SimplexPosterior:
    """The K-simplex posterior for one spectrum. Sample it as often as you like."""

    def __init__(self, model, context):
        self.model = model
        self.context = context      # [1, d_model]

    def __repr__(self):
        return f"SimplexPosterior(d_model={self.context.shape[-1]}, device={self.context.device})"

    def sample(self, n_samples: int = 2000):
        """n_samples -> (composition [n, K] summing to 1, grain [n, K] µm, NaN where absent)."""
        with torch.no_grad():
            free = self.model.cnf.dist(self.context).sample((n_samples,)).squeeze(1)
            comp = free_to_full_simplex(free)
            return comp, self.model.grain.sample_grid(self.context, comp)
