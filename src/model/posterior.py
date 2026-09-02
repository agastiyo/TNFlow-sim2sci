"""Analysis of posterior draws. Everything takes the two matrices from SimplexPosterior.sample().

comp  [N, K] simplex rows (each sums to 1)
grain [N, K] µm, NaN wherever that draw does not contain the component

All torch, so it stays on whatever device the model is on. Only the BIC-GMM mode fit (scikit-learn)
and the plots (matplotlib) drop to CPU/numpy, and they do it internally.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data_utils.components import Component

DEFAULT_SEED = 10025734  # fixes posterior sampling + BIC-GMM mode fitting
MIN_GRAIN_DRAWS = 1      # draws needed to report a grain; the frequency is reported alongside


def order_by_median(comp, n=None):
    """comp [N, K] -> component indices by descending median proportion (top n, or all)."""
    order = comp.median(dim=0).values.argsort(descending=True)
    return order if n is None else order[:n]


def order_by_detection(comp, n=None):
    """comp [N, K] -> component indices by descending detection frequency (top n, or all).

    Same ranking key as validation Top-N (``metrics.detection_score``): the fraction of draws in
    which the component is present enough to receive a grain draw (w > ``PRESENT_THRESH``), tie-broken
    by mean abundance. Using it for inference display keeps what a human reads as the "top" components
    identical to what the reported metric scores. Local import avoids a model->pipeline load cycle.
    """
    from src.pipeline.metrics import detection_score
    order = detection_score(comp).argsort(descending=True)
    return order if n is None else order[:n]


def credible(x):
    """x [N] -> (median, (lo68, hi68), (lo90, hi90)) floats; NaNs if empty."""
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan"), (float("nan"),) * 2, (float("nan"),) * 2
    q = torch.quantile(x, torch.tensor([0.05, 0.16, 0.50, 0.84, 0.95], device=x.device))
    lo90, lo68, med, hi68, hi90 = q.tolist()
    return med, (lo68, hi68), (lo90, hi90)


def grain_draws(grain, c, min_draws=MIN_GRAIN_DRAWS):
    """grain [N, K], c: int -> that component's finite positive draws (empty if under min_draws)."""
    g = grain[:, c]
    g = g[torch.isfinite(g) & (g > 0)]
    return g if g.numel() >= min_draws else g[:0]


def grain_median(grain, min_draws=MIN_GRAIN_DRAWS):
    """grain [N, K] -> [K] median µm over the draws containing each component; NaN where too few."""
    med = torch.nanmedian(grain, dim=0).values
    enough = (~torch.isnan(grain)).sum(dim=0) >= min_draws
    return med.masked_fill(~enough, float("nan"))


def grain_frequency(grain):
    """grain [N, K] -> [K] fraction of draws containing each component."""
    return (~torch.isnan(grain)).to(grain.dtype).mean(dim=0)


def _bic_gmm(X, max_k, reg_covar=1e-4, seed=DEFAULT_SEED, patience=2, min_scan=4):
    """Fit a BIC-selected GMM. Returns the best GaussianMixture."""
    from sklearn.mixture import GaussianMixture

    best, best_bic, rises = None, float("inf"), 0
    for k in range(1, max_k + 1):
        gm = GaussianMixture(k, covariance_type="full", reg_covar=reg_covar, random_state=seed)
        gm.fit(X)
        if (bic := gm.bic(X)) < best_bic:
            best, best_bic, rises = gm, bic, 0
        else:
            rises += 1
            if patience is not None and k >= min_scan and rises >= patience:
                break
    return best


def modes(comp, grain, max_modes=None, reg_covar=1e-4, seed=DEFAULT_SEED,
          patience=2, min_scan=4):
    """Joint comp+grain BIC-GMM: each mode is a complete solution (composition + grain sizes).

    Fits in [comp free coords | log10(grain)] space so modes capture the full joint posterior.
    NaN grains (absent components) are zero-filled before fitting.

    Returns list of dicts sorted by descending weight:
        {"weight", "composition" [K], "grain_um" [K] (NaN where absent), "spread"}
    """
    from src.model.grain import GRAIN_EPS, PRESENT_THRESH

    comp_free = comp[:, :-1].detach().cpu().numpy()
    log_g = torch.log10(grain.clamp(min=GRAIN_EPS)).detach().cpu()
    log_g = torch.where(torch.isfinite(log_g), log_g, torch.zeros_like(log_g)).numpy()
    X = np.concatenate([comp_free, log_g], axis=1)

    Kc = comp.shape[1]
    best = _bic_gmm(X, max_modes or Kc, reg_covar, seed, patience, min_scan)

    kf = comp_free.shape[1]
    out = []
    for j in best.weights_.argsort()[::-1]:
        mu_comp = torch.as_tensor(best.means_[j, :kf], dtype=comp.dtype, device=comp.device)
        full = torch.cat([mu_comp, (1.0 - mu_comp.sum()).reshape(1)]).clamp(min=0.0)
        full = full / full.sum()

        mu_log_g = torch.as_tensor(best.means_[j, kf:], dtype=comp.dtype, device=comp.device)
        grain_um = 10.0 ** mu_log_g
        present = full > PRESENT_THRESH
        grain_um = torch.where(present, grain_um,
                               torch.tensor(float("nan"), dtype=grain_um.dtype, device=grain_um.device))

        var = torch.as_tensor(best.covariances_[j].diagonal().copy(),
                              dtype=comp.dtype, device=comp.device)
        out.append({"weight": float(best.weights_[j]),
                    "composition": full,
                    "grain_um": grain_um,
                    "spread": float(var[:kf].clamp(min=0).mean().sqrt())})
    return out


# ------------------------------------------------------------------------------- printing
def print_marginals(comp, true=None, n=5):
    """comp [N, K] -> print median + 68%/90% CIs for the top n components. true [K] optional."""
    print(f"Composition posterior  (top {n} components by detection frequency)")
    print(f"  {'component':>14s}   {'median':>6s}   {'68% CI':^16s}   {'90% CI':^16s}")
    for c in order_by_detection(comp, n).tolist():
        med, (lo68, hi68), (lo90, hi90) = credible(comp[:, c])
        tag = f"   <- TRUE {true[c]:.3f}" if (true is not None and true[c] > 0) else ""
        print(f"  {Component.decode(c):>14s}   {med:6.3f}   "
              f"[{lo68:5.3f}, {hi68:5.3f}]   [{lo90:5.3f}, {hi90:5.3f}]{tag}")


def print_grains(comp, grain, true_grains=None, n=5):
    """comp/grain [N, K] -> print grain median + CIs for the top n components.

    ``freq`` = share of draws containing the component; a low freq means the grain is conditional
    on a rare branch of the composition posterior.
    """
    freq = grain_frequency(grain)
    print(f"Grain-size posterior (µm)  (top {n} components by detection frequency, where present)")
    print(f"  {'component':>14s}   {'freq':>6s}   {'median':>8s}   {'68% CI':^18s}   {'90% CI':^18s}")
    for c in order_by_detection(comp, n).tolist():
        med, (lo68, hi68), (lo90, hi90) = credible(grain_draws(grain, c))
        tag = (f"   <- TRUE {true_grains[c]:.1f}"
               if true_grains is not None and torch.isfinite(torch.as_tensor(true_grains[c])) else "")
        print(f"  {Component.decode(c):>14s}   {freq[c]:6.1%}   {med:8.1f}   "
              f"[{lo68:7.1f}, {hi68:7.1f}]   [{lo90:7.1f}, {hi90:7.1f}]{tag}")


def print_modes(comp, grain, true_comp=None, true_grains=None, max_modes=None, seed=DEFAULT_SEED):
    """Print BIC-selected joint comp+grain modes, components above 1% each."""
    found = modes(comp, grain, max_modes=max_modes, seed=seed)
    print(f"Found {len(found)} joint mode(s) (BIC-selected, comp+grain) over {len(comp)} samples:")
    for r, m in enumerate(found, 1):
        w = m["composition"]
        g = m["grain_um"]
        parts = []
        for c in w.argsort(descending=True).tolist():
            if w[c] <= 0.01:
                continue
            s = f"{Component.decode(c)} {w[c]:.2f}"
            if torch.isfinite(g[c]):
                s += f" ({g[c]:.0f}µm)"
            parts.append(s)
        print(f"  {r}: wgt {m['weight']:.2f} | spr {m['spread']:.3f} | {', '.join(parts)}")
    if true_comp is not None:
        true_comp = torch.as_tensor(true_comp)
        parts = []
        for c in true_comp.argsort(descending=True).tolist():
            if true_comp[c] <= 0:
                continue
            s = f"{Component.decode(c)} {true_comp[c]:.2f}"
            if true_grains is not None:
                tg = torch.as_tensor(true_grains)
                if torch.isfinite(tg[c]) and tg[c] > 0:
                    s += f" ({tg[c]:.0f}µm)"
            parts.append(s)
        print(f"  (true: {', '.join(parts)})")


# ---------------------------------------------------------------------------------- plots
def hist_marginals(comp, true=None, n=3):
    """comp [N, K] -> composition histograms for the top n components. Returns: fig."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    for ax, c in zip(_axes(axes), order_by_detection(comp, n).tolist()):
        ax.hist(comp[:, c].cpu().numpy(), bins=50, range=(0.0, 1.0), density=True, alpha=0.6)
        if true is not None:
            ax.axvline(true[c], color="r", lw=2, label=f"true = {true[c]:.3f}")
            ax.legend()
        ax.set_title(f"{Component.decode(c)} marginal")
        ax.set_xlabel("proportion"); ax.set_ylabel("density")
    fig.tight_layout()
    return fig


def hist_grains(comp, grain, n=3):
    """comp/grain [N, K] -> grain histograms for the top n components. Returns: fig."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    for ax, c in zip(_axes(axes), order_by_detection(comp, n).tolist()):
        g = grain_draws(grain, c)
        if g.numel():
            ax.hist(g.cpu().numpy(), bins=40)
            ax.set_xscale("log")
        ax.set_title(f"{Component.decode(c)} grain")
        ax.set_xlabel("grain size (µm)")
    fig.tight_layout()
    return fig


def _axes(axes):
    """matplotlib returns a bare Axes for n=1; always give back something iterable."""
    return axes if hasattr(axes, "__len__") else [axes]
