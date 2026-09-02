"""TNFlow validation: per-solution accuracy (BIC-GMM joint modes) + calibration + timing.

Scores both held-out splits (in-distribution test, group-holdout OOD) with 500 posterior draws
per query. Mode-finding is joint over composition + log-grain so each mode is a complete solution.

Run from the project root:  python3.10 -m src.pipeline.test
"""
import json
import time

import numpy as np
import torch

from src import config
from src.data_utils.dataloader import DataLoader, add_noise
from src.data_utils.spectrum_sample import SpectrumSample
from src.data_utils.store import mixture_id
from src.model.posterior import grain_draws, modes
from src.pipeline.checkpoint import load_model
from src.pipeline.metrics import energy_score, grain_log_error, pit, pit_cvm

N_EVAL_TEST = 5000
N_SAMPLES = 500
SEED = 10025734
PIT_BINS = 10
TRACE_THRESH = 0.05
SPLITS = ("test", "ood")
STAGES = ("encode", "sample", "modes")


def _noised(sample, rng):
    if not sample.synthetic:
        return sample
    values = add_noise(sample.values[None, :], sample.lams[None, :], rng)[0]
    return SpectrumSample(sample.lams, values, sample.target, sample.synthetic, sample.grains)


def queries_for(dl, split, noise_rng, n_eval):
    pool = {"test": dl.test, "ood": dl.ood}[split]
    rng = np.random.default_rng(SEED)
    n = min(n_eval, len(pool))
    indices = rng.choice(len(pool), size=n, replace=False)
    queries = [_noised(pool[i], noise_rng) for i in indices]

    if split == "ood":
        n_mix = len({mixture_id(s.target) for s in queries})
        desc = f"Group-holdout OOD ({len(queries)} spectra, {n_mix} held-out mixtures)"
    else:
        desc = f"In-distribution held-out ({len(queries)} spectra)"
    return queries, desc


# --------------------------------------------------------------------------- mode metrics

def _mode_comp_tv(mode, target):
    return float(0.5 * (mode["composition"] - target).abs().sum())


def _mode_grain_dex(mode, truth_g):
    g_mode = mode["grain_um"]
    ok = torch.isfinite(g_mode) & torch.isfinite(truth_g) & (truth_g > 0) & (g_mode > 0)
    if not ok.any():
        return float("nan")
    return float(grain_log_error(g_mode[ok], truth_g[ok]).mean())


# --------------------------------------------------------------------------- evaluate

def evaluate(model, samples, device, label=""):
    timings = {s: [] for s in STAGES}

    top_tv, best_tv = [], []
    top_gdex, best_gdex = [], []
    top_is_best = []
    mode_counts = []

    comp_pit_vals, grain_pit_vals = [], []
    comp_pit_trace, comp_pit_nontrace = [], []
    spread_list = []
    n_pres, n_grain = 0, 0

    for i, s in enumerate(samples, 1):
        target = torch.as_tensor(s.target, dtype=torch.float32, device=device)
        truth_g = torch.as_tensor(s.grains, dtype=torch.float32, device=device)
        present = target > 0

        t0 = time.perf_counter()
        post = model.simplex(s.values, s.lams, seed=SEED)
        t1 = time.perf_counter()

        comp, grain = post.sample(N_SAMPLES)
        t2 = time.perf_counter()

        found = modes(comp, grain)
        t3 = time.perf_counter()

        timings["encode"].append(t1 - t0)
        timings["sample"].append(t2 - t1)
        timings["modes"].append(t3 - t2)

        # --- per-solution accuracy ---
        tvs = [_mode_comp_tv(m, target) for m in found]
        best_idx = int(np.argmin(tvs))
        mode_counts.append(len(found))

        top_tv.append(tvs[0])
        best_tv.append(tvs[best_idx])
        top_gdex.append(_mode_grain_dex(found[0], truth_g))
        best_gdex.append(_mode_grain_dex(found[best_idx], truth_g))
        top_is_best.append(best_idx == 0)

        # --- calibration: composition PIT (present components only) ---
        for c in present.nonzero().flatten().tolist():
            p = pit(comp[:, c], target[c])
            comp_pit_vals.append(p)
            if float(target[c]) <= TRACE_THRESH:
                comp_pit_trace.append(p)
            else:
                comp_pit_nontrace.append(p)
        n_pres += int(present.sum())

        # --- calibration: grain PIT (log10 µm, where scorable) ---
        scorable = present & torch.isfinite(truth_g) & (truth_g > 0)
        for c in scorable.nonzero().flatten().tolist():
            g = grain_draws(grain, c)
            if not g.numel():
                continue
            n_grain += 1
            grain_pit_vals.append(pit(torch.log10(g), torch.log10(truth_g[c])))

        # --- sharpness ---
        _, spread = energy_score(comp, target)
        spread_list.append(spread)

        print(f"    {label} {i}/{len(samples)}", end="\r")
    print()

    def _nanmean(xs):
        finite = [x for x in xs if np.isfinite(x)]
        return float(np.mean(finite)) if finite else float("nan")

    def _nanstd(xs):
        finite = [x for x in xs if np.isfinite(x)]
        return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")

    mc = np.asarray(mode_counts)
    return {
        "Top-mode comp TV": _nanmean(top_tv),
        "Top-mode comp TV std": _nanstd(top_tv),
        "Best-mode comp TV": _nanmean(best_tv),
        "Best-mode comp TV std": _nanstd(best_tv),
        "Top-mode grain dex": _nanmean(top_gdex),
        "Top-mode grain dex std": _nanstd(top_gdex),
        "Best-mode grain dex": _nanmean(best_gdex),
        "Best-mode grain dex std": _nanstd(best_gdex),
        "Top = Best (%)": 100.0 * sum(top_is_best) / len(top_is_best) if top_is_best else float("nan"),
        "Mean mode count": float(mc.mean()),
        "Multimodal (%)": 100.0 * float((mc > 1).sum()) / len(mc) if len(mc) else float("nan"),
        "comp PIT W2": pit_cvm(comp_pit_vals),
        "grain PIT W2": pit_cvm(grain_pit_vals),
        "sharpness": _nanmean(spread_list),
        "sharpness std": _nanstd(spread_list),
        "comp_pit": comp_pit_vals,
        "comp_pit_trace": comp_pit_trace,
        "comp_pit_nontrace": comp_pit_nontrace,
        "grain_pit": grain_pit_vals,
        "n_present": n_pres,
        "n_grain": n_grain,
        "n_trace": len(comp_pit_trace),
        "n_nontrace": len(comp_pit_nontrace),
    }, timings


# --------------------------------------------------------------------------- PIT figures

def _pit_step(ax, pit_vals, title, ylabel=True, xlabel=True):
    """Draw one step-style PIT histogram on an axes (frequency = proportion)."""
    u = np.asarray(pit_vals)
    counts, edges = np.histogram(u, bins=PIT_BINS, range=(0.0, 1.0))
    freq = counts / len(u) if len(u) else counts.astype(float)
    ax.stairs(freq, edges, color="#4C72B0", linewidth=1.5)
    ax.axhline(1.0 / PIT_BINS, color="red", ls="--", lw=1.0)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel("Frequency")
    if xlabel:
        ax.set_xlabel("PIT value")


def pit_figure(splits_data, path):
    """2×2 PIT histogram: rows = comp / grain, cols = in-dist / OOD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True, sharey="row")

    for col, (label, comp_pit, grain_pit) in enumerate(splits_data):
        _pit_step(axes[0, col], comp_pit,
                  f"Composition — {label}", ylabel=(col == 0), xlabel=False)
        _pit_step(axes[1, col], grain_pit,
                  f"Grain (log µm) — {label}", ylabel=(col == 0), xlabel=True)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  PIT figure saved to {path}")


def pit_trace_figure(splits_data, path):
    """2×2 PIT histogram: rows = trace / non-trace comp, cols = in-dist / OOD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True, sharey="row")

    for col, (label, trace_pit, nontrace_pit) in enumerate(splits_data):
        _pit_step(axes[0, col], trace_pit,
                  f"Trace (≤{TRACE_THRESH:.0%}) — {label}", ylabel=(col == 0), xlabel=False)
        _pit_step(axes[1, col], nontrace_pit,
                  f"Non-trace (>{TRACE_THRESH:.0%}) — {label}", ylabel=(col == 0), xlabel=True)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  PIT trace/non-trace figure saved to {path}")


# --------------------------------------------------------------------------- report

ACCURACY_KEYS = ("Top-mode comp TV", "Best-mode comp TV",
                 "Top-mode grain dex", "Best-mode grain dex", "Top = Best (%)",
                 "Mean mode count", "Multimodal (%)")
CALIB_KEYS = ("comp PIT W2", "grain PIT W2", "sharpness")


def report(desc, m, timings):
    line = "=" * 66
    print("\n" + line)
    print(desc)
    print(line)

    print("  per-solution accuracy (joint comp+grain BIC-GMM modes)")
    for k in ACCURACY_KEYS:
        if "%" in k:
            print(f"    {k:>24s} : {m[k]:8.1f}")
        elif k + " std" in m:
            print(f"    {k:>24s} : {m[k]:8.4f} ± {m[k + ' std']:.4f}")
        else:
            print(f"    {k:>24s} : {m[k]:8.2f}")

    print(f"\n  calibration  ({m['n_present']} present components, {m['n_grain']} grains)")
    for k in CALIB_KEYS:
        if k + " std" in m:
            print(f"    {k:>24s} : {m[k]:8.4f} ± {m[k + ' std']:.4f}")
        else:
            print(f"    {k:>24s} : {m[k]:8.4f}")
    print("      (PIT W² 0 = uniform/calibrated; sharpness = mean pairwise draw dist)")

    print(f"\n  composition PIT: {m['n_trace']} trace (≤{TRACE_THRESH:.0%}), "
          f"{m['n_nontrace']} non-trace (>{TRACE_THRESH:.0%})")

    print("\n  composition PIT histogram (all present components; want flat)")
    for ln in _pit_hist(m["comp_pit"]):
        print(ln)
    print("\n  composition PIT histogram (trace only, ≤5%)")
    for ln in _pit_hist(m["comp_pit_trace"]):
        print(ln)
    print("\n  composition PIT histogram (non-trace only, >5%)")
    for ln in _pit_hist(m["comp_pit_nontrace"]):
        print(ln)
    print("\n  grain PIT histogram (log10 µm, where present)")
    for ln in _pit_hist(m["grain_pit"]):
        print(ln)

    if timings and timings[STAGES[0]]:
        n = len(timings[STAGES[0]])
        per_query = np.zeros(n)
        print(f"\n  inference cost per query, mean ± std over {n} queries (ms)")
        for s in STAGES:
            a = np.asarray(timings[s]) * 1e3
            per_query += a
            print(f"    {s:>14s} : {a.mean():8.1f} ± {a.std():6.1f}")
        print(f"    {'TOTAL':>14s} : {per_query.mean():8.1f} ± {per_query.std():6.1f}"
              f"   ({per_query.sum() / 1e3:.1f}s total)")
    print(line)


def _pit_hist(u, bins=PIT_BINS, width=40):
    if not u:
        return ["    (no scored components)"]
    counts, _ = np.histogram(np.asarray(u), bins=bins, range=(0.0, 1.0))
    peak = max(int(counts.max()), 1)
    ideal = len(u) / bins
    lines = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bar = "#" * int(round(width * counts[b] / peak))
        lines.append(f"    [{lo:.1f},{hi:.1f}) {counts[b]:5d} {bar}")
    lines.append(f"    (flat reference ≈ {ideal:.0f} per bin; U-shape = overconfident)")
    return lines


# --------------------------------------------------------------------------- main

def main(checkpoint=config.BEST_LOSS_CHECKPOINT):
    device = config.get_device()
    model, ckpt = load_model(checkpoint, device)
    print(f"  config: {ckpt['config']}")
    dl = DataLoader()

    noise_rng = np.random.default_rng(SEED)

    # warmup (untimed)
    warmup_sample = _noised(dl.test[0], noise_rng)
    post = model.simplex(warmup_sample.values, warmup_sample.lams, seed=SEED)
    c, g = post.sample(10)
    modes(c, g)

    splits_pit = []
    splits_trace = []

    for split in SPLITS:
        n_eval = N_EVAL_TEST if split == "test" else len(dl.ood)
        queries, desc = queries_for(dl, split, noise_rng, n_eval)
        print(f"\n{desc}")

        scored, timings = evaluate(model, queries, device, split)
        report(desc, scored, timings)

        label = "In-distribution" if split == "test" else "OOD"
        splits_pit.append((label, scored["comp_pit"], scored["grain_pit"]))
        splits_trace.append((label, scored["comp_pit_trace"], scored["comp_pit_nontrace"]))

    plot_dir = config.CHECKPOINT_DIR

    pit_data = {}
    for label, comp, grain in splits_pit:
        key = label.lower().replace("-", "").replace(" ", "_")
        pit_data[f"{key}_comp_pit"] = comp
        pit_data[f"{key}_grain_pit"] = grain
    for label, trace, nontrace in splits_trace:
        key = label.lower().replace("-", "").replace(" ", "_")
        pit_data[f"{key}_comp_pit_trace"] = trace
        pit_data[f"{key}_comp_pit_nontrace"] = nontrace

    pit_path = plot_dir / "pit_data.json"
    with open(pit_path, "w") as f:
        json.dump(pit_data, f)
    print(f"  PIT data saved to {pit_path}")

    pit_figure(splits_pit, plot_dir / "pit_histograms.png")
    pit_trace_figure(splits_trace, plot_dir / "pit_trace_histograms.png")


if __name__ == "__main__":
    main()
