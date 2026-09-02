"""GrainFlow: per-component grain-size posterior (shared 1-D NSF over standardized log10 grain)."""

import torch
import torch.nn as nn
import zuko

from src.data_utils.components import K

GRAIN_EPS = 1e-6  # floor for log10 on the grain target

# A composition draw "contains" component k when w_k exceeds this threshold.
PRESENT_THRESH = 0.01


class GrainFlow(nn.Module):
    """1-D NSF over standardized log10(grain/µm), conditioned on [context, composition, component embed]."""

    def __init__(self, context: int, embed_dim: int = 16,
                 log_grain_mean: float = 0.0, log_grain_std: float = 1.0, **flow_kwargs):
        super().__init__()
        self.embed = nn.Embedding(K, embed_dim)
        self.flow = zuko.flows.NSF(features=1, context=context + K + embed_dim, **flow_kwargs)
        self.register_buffer("log_mean", torch.tensor(float(log_grain_mean)))
        self.register_buffer("log_std", torch.tensor(float(log_grain_std)))

    def _cond(self, c, comp, idx):
        return torch.cat([c, comp, self.embed(idx)], dim=-1)

    def _standardize(self, grain_um):
        return (torch.log10(grain_um.clamp(min=GRAIN_EPS)) - self.log_mean) / self.log_std

    def _destandardize(self, z):
        return 10.0 ** (z * self.log_std + self.log_mean)

    def log_prob(self, context, comp_context, grain_target, mask):
        """Summed log-density over present (spectrum, component) pairs, and their count.

        Args:    c: Tensor [B, d_model]; comp: Tensor [B, K]; grain: Tensor [B, K] µm; mask: BoolTensor [B, K].
        Returns: (Tensor scalar log-density sum, int count).
        """
        b, k = mask.nonzero(as_tuple=True)
        if b.numel() == 0:
            return context.new_zeros(()), 0
        z = self._standardize(grain_target[b, k]).unsqueeze(-1)
        lp = self.flow(self._cond(context[b], comp_context[b], k)).log_prob(z)
        return lp.sum(), int(b.numel())

    def sample_grid(self, c, comp, present_thresh: float = PRESENT_THRESH):
        """One grain sample (µm) per component per composition sample; NaN where absent.

        Args:    c: Tensor [1, d_model] or [d_model]; comp: Tensor [N, K]; present_thresh: float.
        Returns: Tensor [N, K] µm, NaN where comp <= present_thresh.
        """
        c = c.reshape(1, -1)
        n = comp.shape[0]
        idx = torch.arange(K, device=comp.device).repeat(n)
        crep = c.expand(n, -1).repeat_interleave(K, dim=0)
        comprep = comp.repeat_interleave(K, dim=0)
        z = self.flow(self._cond(crep, comprep, idx)).sample().squeeze(-1)
        grain = self._destandardize(z).reshape(n, K)
        return grain.masked_fill(comp <= present_thresh, float("nan"))

    def predict_median(self, c, comp, n=64):
        """c [B, d_model], comp [B, K] -> median grain [B, K] µm over n draws at a fixed comp.

        Used only by train.py's teacher-forced training curve; inference goes through sample_grid.
        """
        b = comp.shape[0]
        idx = torch.arange(K, device=comp.device).repeat(b)
        crep = c.repeat_interleave(K, dim=0)
        comprep = comp.repeat_interleave(K, dim=0)
        z = self.flow(self._cond(crep, comprep, idx)).sample((n,)).squeeze(-1)
        return self._destandardize(z).reshape(n, b, K).median(0).values
