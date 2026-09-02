"""Apply a checkpoint to a JWST / other-object / generated-Shkuratov spectrum: plot it, print
posterior + modes + grains.

Sources: JWST, other objects, and "Shkuratov (generated)" -- a fresh intimate-mixture spectrum built
from the MIXTURE / POROSITY block below (edit it here for a manual input).

Run from the project root:  python3.10 -m src.pipeline.inference
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

from src import config
from src.data_utils.processing import jwst, other_objects
from src.model.posterior import (DEFAULT_SEED, hist_grains, hist_marginals,
                                 print_grains, print_marginals, print_modes)
from src.pipeline.checkpoint import load_model

import warnings

warnings.filterwarnings('ignore')

N_SAMPLES = 5000
SEED = DEFAULT_SEED

# --- Shkuratov (generated) source input: edit for a manual mixture ------------------------
# name -> (proportion, grain_size_um); names must be Component enum names, proportions ~sum to 1.
MIXTURE = {
    "CO" : (1.0, 41.0),
}
POROSITY = 0.1
# ------------------------------------------------------------------------------------------

#%%
def choose(options, prompt="Select", n_cols=1):
    """Print options in a grid and prompt for an index.

    Args:    options: list[str]; prompt: input prompt text; n_cols: names per line.
    Returns: int index.
    """
    n_cols = max(1, min(n_cols, len(options)))
    width = max(len(o) for o in options) + 7
    n_rows = -(-len(options) // n_cols)  # ceil
    for r in range(n_rows):
        row = "".join(
            f"[{r * n_cols + c:2d}] {options[r * n_cols + c]}".ljust(width)
            for c in range(n_cols) if r * n_cols + c < len(options)
        )
        print(row.rstrip())
    raw = input(f"{prompt} [0-{len(options) - 1}] (default 0): ").strip()
    idx = int(raw) if raw else 0
    if not 0 <= idx < len(options):
        raise SystemExit(f"index {idx} out of range")
    return idx


def plot(sample, name):
    """Plot a spectrum. Returns: the matplotlib module (for show())."""
    fig, ax = plt.subplots()
    ax.plot(sample.lams, sample.values, lw=1)
    ax.set_title(name)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("arbitrary reflectance")
    return plt


def describe_attention(lam, weights, n=10):
    """Human-readable listing of the top-n highest-attention wavelengths.

    Args:    lam, weights: ndarray [L] (from ``TNFlow.attention``); n: how many to list.
    Returns: str.
    """
    order = np.argsort(weights)[::-1][:n]
    lines = [f"Top {n} wavelengths by query-pooling attention weight (of {len(lam)} total):"]
    for i in order:
        lines.append(f"  {lam[i]:7.4f} µm   weight {weights[i]:.4f}")
    return "\n".join(lines)


def plot_attention(sample, lam, weights, name):
    """Plot the spectrum with its per-wavelength query-pooling attention weight below it.

    Returns: the matplotlib module (for show()).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(sample.lams, sample.values, lw=1, color="#4C72B0")
    ax1.set_ylabel("reflectance")
    ax1.set_title(f"{name}")

    ax2.plot(lam, weights, lw=1, color="#C44E52")
    ax2.fill_between(lam, weights, alpha=0.3, color="#C44E52")
    ax2.set_xlabel("wavelength (µm)")
    ax2.set_ylabel("attention weight")
    ax2.set_title("Query-pooling attention (share of the context vector from each wavelength)")
    fig.tight_layout()
    return plt

#%%
class _ShkuratovSource:
    """Generate one Shkuratov intimate-mixture spectrum from the module-level MIXTURE / POROSITY.

    The mixture is the manual input block at the top of this file. Only the generation *machinery*
    (optical-constant load, wavelength grid, IntimateMixture) is reused from plot_one_shkuratov, and
    it is imported lazily -- so cana/sdoc load only when this source is selected, and the JWST /
    other-object paths pay nothing. The generated sample carries the true composition + grains as its
    target.
    """
    _cache = None

    def _gen(self):
        if _ShkuratovSource._cache is None:
            from src.data_utils.plot_one_shkuratov import (load_optical_constant, WAVELENGTHS,
                                                           IntimateMixture)
            names = list(MIXTURE.keys())
            props = [MIXTURE[n][0] for n in names]
            grains = [MIXTURE[n][1] for n in names]
            ocs = [load_optical_constant(n) for n in names]
            mix = IntimateMixture(ocs, grainsizes=grains, proportions=props, porosity=POROSITY)
            with np.errstate(invalid="ignore"):
                spec, _ = mix.make(wavelengths=WAVELENGTHS)
            label = ", ".join(f"{n} {p:.2f}" for n, p in zip(names, props))
            _ShkuratovSource._cache = (np.asarray(WAVELENGTHS, float),
                                       np.asarray(spec.r, float), label)
        return _ShkuratovSource._cache

    def designations(self):
        return [self._gen()[2]]

    def process(self):
        from src.data_utils.components import Component, K
        from src.data_utils.spectrum_sample import SpectrumSample
        lams, refl, _ = self._gen()
        target = np.zeros(K, dtype=np.float32)
        grains = np.full(K, np.nan, dtype=np.float32)
        for n, (p, g) in MIXTURE.items():
            k = Component[n].value
            target[k], grains[k] = p, g
        return [SpectrumSample(lams, refl, target, synthetic=True, grains=grains)]


def main(checkpoint=config.BEST_LOSS_CHECKPOINT):
    model, _ = load_model(checkpoint)

    sources = [("JWST", jwst), ("Other objects", other_objects),
               ("Shkuratov (generated)", _ShkuratovSource())]
    print("Choose a spectrum source:")
    src_idx = choose([label for label, _ in sources], prompt="Select a source")
    label, module = sources[src_idx]

    names = module.designations()
    samples = module.process()

    print(f"\nChoose a {label} spectrum:")
    idx = choose(names, prompt="Select a spectrum", n_cols=4)
    sample, name = samples[idx], names[idx]
    print(f"\nSelected {name} (D={len(sample.values)})")

    plt_ = plot(sample, name)
    plt_.show()

    attn_lam, attn_weights = model.attention(spec=sample.values, lams=sample.lams)
    print(f"\n===== {name}: query-pooling attention =====")
    print(describe_attention(attn_lam, attn_weights))
    plot_attention(sample, attn_lam, attn_weights, name)
    plt_.show()

    comp, grain = model.simplex(sample.values, sample.lams, seed=SEED).sample(N_SAMPLES)
    true_comp = sample.target if hasattr(sample, 'target') and sample.target is not None else None
    true_grains = sample.grains if hasattr(sample, 'grains') and sample.grains is not None else None
    print(f"\n===== {name}: posterior (median, 68% & 90% credible intervals) =====")
    print()
    print_modes(comp, grain, true_comp=true_comp, true_grains=true_grains)
    print()
    print_marginals(comp, true=true_comp, n=10)
    print()
    print_grains(comp, grain, true_grains=true_grains, n=10)

    hist_marginals(comp, n=3)
    hist_grains(comp, grain, n=3)
    plt_.show()


if __name__ == "__main__":
    main()

# %%
