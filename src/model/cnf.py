"""CNF: conditional normalizing flow over the K-component composition simplex.

Operates in K-1 free coordinates, wrapped in a stick-breaking transform.
"""

import torch
import torch.nn as nn
import zuko

from torch.distributions import TransformedDistribution, constraints
from torch.distributions.transforms import AffineTransform, ComposeTransform, StickBreakingTransform, Transform

EPS = 1e-4  # boundary smoothing for exact-zero components


def free_to_full_simplex(free):
    """Append the dropped coordinate (1 - sum) to recover the full K-simplex.

    Args:    free: Tensor [..., K-1].
    Returns: Tensor [..., K] (last column = 1 - free.sum(-1)).
    """
    return torch.cat([free, 1 - free.sum(-1, keepdim=True)], dim=-1)


class FreeSimplexStickBreaking(Transform):
    """Bijection R^{K-1} <-> first K-1 simplex coords (drops the redundant last coord)."""

    domain = constraints.real_vector
    codomain = constraints.independent(constraints.unit_interval, 1)
    bijective = True

    def __init__(self):
        super().__init__()
        self._sb = StickBreakingTransform()

    def _call(self, x):
        return self._sb(x)[..., :-1]

    def _inverse(self, y):
        return self._sb.inv(free_to_full_simplex(y))

    def log_abs_det_jacobian(self, x, y):
        return self._sb.log_abs_det_jacobian(x, free_to_full_simplex(y))


class CNF(nn.Module):
    """NSF on the K-simplex, conditioned on a context vector."""

    def __init__(self, features: int, context: int, scale: float = 2.5, **flow_kwargs):
        super().__init__()
        self.flow = zuko.flows.NSF(features=features, context=context, **flow_kwargs)
        self.transform = ComposeTransform([AffineTransform(0.0, scale), FreeSimplexStickBreaking()])

    def dist(self, context):
        """Composition distribution given context (TransformedDistribution over the simplex)."""
        return TransformedDistribution(self.flow(context), self.transform, validate_args=False)

    @staticmethod
    def smooth(targets):
        """Nudge exact-zero components into the open simplex so stick-breaking stays finite.

        Args:    targets: Tensor [B, K-1] free coords.
        Returns: Tensor [B, K-1] smoothed free coords.
        """
        full = free_to_full_simplex(targets) + EPS
        full = full / full.sum(-1, keepdim=True)
        return full[..., :-1]

    def log_prob(self, context, targets):
        """Log-density of (smoothed) targets under the composition distribution."""
        return self.dist(context).log_prob(self.smooth(targets))
