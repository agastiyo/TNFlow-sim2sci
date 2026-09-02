"""Evaluation metrics. All torch, so they run on whatever device the tensors are already on."""

import torch

from src.model.grain import GRAIN_EPS, PRESENT_THRESH


def ranks_from_scores(scores):
    """scores [B, K] -> [B, K] int ranks, 0 = highest score."""
    return scores.argsort(dim=1, descending=True).argsort(dim=1)


def detection_score(comp_draws, dim=0):
    """comp_draws [S, ..., K] -> [..., K] score ranking components for Top-N / Top-5.

    Primary key is the **detection frequency**: the fraction of posterior draws in which the
    component clears ``PRESENT_THRESH`` -- i.e. is present enough that ``GrainFlow.sample_grid``
    gives it a grain draw instead of NaN. Top-N asks *which components are present*, so the
    marginal detection probability is the right quantity; posterior mean abundance conflates "how
    often" with "how much" and ranks a trace species detected in every draw below one proposed
    rarely but large.

    Frequencies are quantised to 1/S and every never-detected component sits at exactly 0, so ties
    are pervasive -- and ``ranks_from_scores`` breaks ties by enum index, which would hand an
    undetected component a lucky rank. Mean abundance enters as a secondary key scaled to stay
    strictly below one frequency step (max 1/(S+1) < 1/S), so it can only order *within* a tie
    group, never override the frequency.
    """
    s = comp_draws.shape[dim]
    freq = (comp_draws > PRESENT_THRESH).to(comp_draws.dtype).mean(dim)
    return freq + comp_draws.mean(dim) / (s + 1)


def recall_at_k(ranks, present, k):
    """ranks [B, K], present bool [B, K], k int|[B, 1] -> count of rows with every present < k."""
    return int(((~present) | (ranks < k)).all(dim=1).sum())


def tv_distance(a, b):
    """a, b [K] -> total-variation distance 0.5·Σ|a−b| (float)."""
    return float(0.5 * (a - b).abs().sum())


def mean_draw_tv(comp_draws, target):
    """comp_draws [S, K], target [K] -> mean TV distance over all draws (float).

    E[TV(draw, target)] = mean over S draws of 0.5·Σ|draw_s − target|. Unlike
    TV(mean(draws), target), this characterizes the full posterior rather than its
    point estimate — a multimodal posterior with two modes straddling the target will
    score high here even if the posterior mean happens to be close.
    """
    return float(0.5 * (comp_draws - target).abs().sum(dim=1).mean())


def central_interval(samples, p, dim=0):
    """samples [S, ...], p in (0,1) -> (lo, hi) at the (1∓p)/2 quantiles, NaNs ignored."""
    q = torch.tensor([(1 - p) / 2, (1 + p) / 2], dtype=samples.dtype, device=samples.device)
    lo, hi = torch.nanquantile(samples, q, dim=dim)
    return lo, hi


def grain_log_error(pred_um, true_um):
    """pred_um, true_um (same shape) -> |Δlog10| in dex, clamped at GRAIN_EPS."""
    return (torch.log10(pred_um.clamp(min=GRAIN_EPS))
            - torch.log10(true_um.clamp(min=GRAIN_EPS))).abs()


def energy_score(draws, target):
    """draws [S, D], target [D] -> (energy_score, sharpness), both floats.

    ES(P, y) = E‖X − y‖ − ½·E‖X − X'‖, the strictly-proper multivariate scoring rule (multivariate
    CRPS) for the JOINT predictive distribution. Lower is better; in composition (abundance) units,
    so only comparable across models on the same test set. Taken over the full K-vector (present and
    absent components alike) -- the honest "whole posterior" number, and the one metric that reflects
    correlation structure the per-component PIT is blind to. ``sharpness`` = E‖X − X'‖ (the spread
    term, un-halved); report it beside ES so a calibrated-but-smeared posterior is distinguishable
    from a genuinely tight one. O(S²) pairwise; trivial at S=500.
    """
    s = draws.shape[0]
    accuracy = torch.norm(draws - target, dim=1).mean()
    spread = torch.cdist(draws, draws).sum() / (s * (s - 1))   # mean over ordered i≠j (diag = 0)
    return float(accuracy - 0.5 * spread), float(spread)


def pit(draws, truth):
    """draws [S], truth scalar -> PIT value = fraction of draws strictly below the truth.

    Under a calibrated predictive, PIT values are Uniform(0,1). Only meaningful where the truth is a
    continuous quantity (present-component abundance, log-grain) -- for an absent component the
    posterior has an atom at the EPS floor and the PIT is degenerate, so callers restrict to present
    components rather than randomizing over the atom.
    """
    return float((draws < truth).to(draws.dtype).mean())


def pit_cvm(u):
    """u: iterable of PIT values -> Cramér–von Mises W² against Uniform(0,1); 0 = uniform.

    W² = 1/(12n) + Σ_i (u_(i) − (2i−1)/(2n))² on the sorted values. Tail-sensitive; lower is better.
    Scales mildly with n, so compare across models/splits at comparable sample counts.
    """
    u, _ = torch.sort(torch.as_tensor(u, dtype=torch.float64))
    n = u.numel()
    if n == 0:
        return float("nan")
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(1.0 / (12 * n) + ((u - (2 * i - 1) / (2 * n)) ** 2).sum())
