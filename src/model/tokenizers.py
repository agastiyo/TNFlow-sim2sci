"""Tokenizers: raw (λ, R) spectrum -> per-channel tokens [<reflectance feature>, sin(ωλ), cos(ωλ)...].

- NormTokenizer:      normalize at the NORM_LAM µm continuum (0.1µm window), then Fourier. Nothing else.
- SlopeBandTokenizer: [slope rate, smoothed relative rolling-median deviation, Fourier]; scale-free.

TNFlow selects one by name (see its ``_TOKENIZERS`` registry); the choice is stored in the checkpoint.
"""

import torch
import torch.nn as nn

N_FREQS = 6

# --- NormTokenizer ---
NORM_LAM = 0.9        # µm reference wavelength to normalize at (per Cristina; in-range for all datasets)
NORM_WINDOW = 0.1     # µm width of the window averaged to get the reference level
NORM_EPS = 1e-6       # numerical floor on the divisor (avoid div-by-zero); not a design knob

# --- SlopeBandTokenizer ---
SLOPE_LO, SLOPE_HI = 0.45, 1.6   # µm endpoints of the slope ratio R(HI)/R(LO)
ROLL_WINDOW = 1.0                # µm width of the rolling-median continuum window
SMOOTH_WINDOW = 0.1              # µm width of the rolling-mean smoothing of the deviation
DIV_EPS = 1e-8                   # numerical div-by-zero guard only (no absolute floor -> scale-invariant)


def _fourier(lam: torch.Tensor, omegas: torch.Tensor) -> list:
    """Positional sin/cos features. Returns [sin(ωλ), cos(ωλ)] each [B, L, N_FREQS]."""
    phases = lam.unsqueeze(-1) * omegas
    return [phases.sin(), phases.cos()]


class NormTokenizer(nn.Module):
    """Raw (λ, R) -> [R / R(NORM_LAM µm window), sin(ωλ), cos(ωλ)...]; width 1 + 2·N_FREQS."""

    def __init__(self):
        super().__init__()
        self.d_token = 1 + 2 * N_FREQS
        self.register_buffer("omegas", 2.0 * torch.pi * 2.0 ** torch.arange(N_FREQS))

    def forward(self, lam: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Args: lam, R: Tensor [B, L]. Returns: tokens Tensor [B, L, d_token]."""
        win = ((lam - NORM_LAM).abs() <= NORM_WINDOW / 2) & torch.isfinite(R)
        wsum = win.sum(1, keepdim=True)
        window_mean = (R * win).sum(1, keepdim=True) / wsum.clamp(min=1)
        nearest = (lam - NORM_LAM).abs().argmin(dim=1, keepdim=True)
        ref = torch.where(wsum > 0, window_mean, R.gather(1, nearest)).clamp(min=NORM_EPS)
        feats = [(R / ref).unsqueeze(-1)] + _fourier(lam, self.omegas)
        return torch.cat(feats, dim=-1)


class SlopeBandTokenizer(nn.Module):
    """Raw (λ, R) -> [slope, smoothed (R-med)/med, sin(ωλ), cos(ωλ)...]; width 2 + 2·N_FREQS.

    Scale-free by construction (slope is a ratio, band is a relative deviation), so absolute
    reflectance level never reaches the model; slope preserves the gradient, band isolates features.
    """

    def __init__(self):
        super().__init__()
        self.d_token = 2 + 2 * N_FREQS
        self.register_buffer("omegas", 2.0 * torch.pi * 2.0 ** torch.arange(N_FREQS))

    @staticmethod
    def _slope(lam: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Per-spectrum slope rate [R(nearest SLOPE_HI)/R(nearest SLOPE_LO)] / Δλ. Returns [B, 1].

        Dividing by the *actual* wavelength span makes it a per-µm rate, so datasets that don't
        reach SLOPE_LO/HI (e.g. JWST starts at 0.70µm) are put on a comparable footing.
        """
        lo_i = (lam - SLOPE_LO).abs().argmin(dim=1, keepdim=True)
        hi_i = (lam - SLOPE_HI).abs().argmin(dim=1, keepdim=True)
        ratio = R.gather(1, hi_i) / R.gather(1, lo_i).clamp(min=DIV_EPS)
        dlam = (lam.gather(1, hi_i) - lam.gather(1, lo_i)).clamp(min=DIV_EPS)
        return ratio / dlam

    @staticmethod
    def _band(lam: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Smoothed relative rolling-median deviation, per point. Returns [B, L].

        (R - med)/med over a ±ROLL_WINDOW/2 µm window, then a rolling MEAN over ±SMOOTH_WINDOW/2.
        Non-finite R is excluded from both windows; a point with an empty window (or itself
        missing) gets 0.
        """
        valid = torch.isfinite(R)
        dlam = (lam.unsqueeze(2) - lam.unsqueeze(1)).abs()              # [B, L, L]: |λ_i - λ_j|
        inwin = (dlam <= ROLL_WINDOW / 2) & valid.unsqueeze(1)
        neigh = R.unsqueeze(1).masked_fill(~inwin, float("nan"))
        med = neigh.nanmedian(dim=2).values                            # NaN only when window empty
        rel = (R - med) / med.clamp(min=DIV_EPS)
        rel = torch.where(valid & torch.isfinite(med), rel, torch.zeros_like(rel))
        smw = (dlam <= SMOOTH_WINDOW / 2) & valid.unsqueeze(1)
        smoothed = (rel.unsqueeze(1) * smw).sum(2) / smw.sum(2).clamp(min=1)
        return torch.where(valid, smoothed, torch.zeros_like(smoothed))

    def forward(self, lam: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Args: lam, R: Tensor [B, L]. Returns: tokens Tensor [B, L, d_token]."""
        L = R.shape[1]
        slope = self._slope(lam, R).unsqueeze(-1).expand(-1, L, -1)    # broadcast scalar over L
        feats = [slope, self._band(lam, R).unsqueeze(-1)] + _fourier(lam, self.omegas)
        return torch.cat(feats, dim=-1)
