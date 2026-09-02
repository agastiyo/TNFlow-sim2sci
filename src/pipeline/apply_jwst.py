"""Apply TNFlow to representative real JWST DiSCo-TNOs spectra (appendix figure + mode report).

Runs the trained model on one exemplar of each DiSCo spectral group (bowl / double-dip / cliff)
and reports **every** BIC-GMM mode, not just the top one, alongside each component's detection
frequency across the posterior draws. Real spectra are unlabelled, so nothing here is scored --
this is a qualitative illustration of the sim-to-real gap, not an evaluation.

Exemplars are the highest-band-area object of each group among those with a high PCA silhouette
coefficient in Pinilla-Alonso et al. (2025) Table 1, chosen so each group's defining species is
tested where its signal is strongest.

Run from the project root:  python3.10 -m src.pipeline.apply_jwst
"""

import json

import numpy as np
import torch

from src import config
from src.data_utils.components import Component
from src.data_utils.processing import jwst
from src.model.grain import PRESENT_THRESH
from src.model.posterior import modes
from src.pipeline.checkpoint import load_model

N_SAMPLES = 500
SEED = 10025734
NORM_LAM, NORM_WIN = 0.9, 0.05
REPORT_THRESH = 0.01          # components below this in a mode are omitted from the printed table

# designation -> (DiSCo group, band-area note from Pinilla-Alonso et al. 2025 Table 1)
TARGETS = {
    "1998SN165": ("Bowl",       "strongest H2O Fresnel band area of the bowl exemplars (0.0075)"),
    "2007UK126": ("Double-dip", "highest CO2 band area in the DiSCo sample (0.0050)"),
    "2004PF115": ("Cliff",      "highest CH3OH band area in the DiSCo sample (0.0074)"),
}

NAMES = {c.value: c.name for c in Component}


def _norm(values, lams):
    """Normalize reflectance to the NORM_LAM continuum, matching the model's tokenizer."""
    w = np.abs(lams - NORM_LAM) < NORM_WIN
    return values / values[w].mean() if w.any() else values


def analyze(model, sample, device):
    """One spectrum -> dict with every mode's composition + per-component detection frequency."""
    post = model.simplex(sample.values, sample.lams, seed=SEED)
    comp, grain = post.sample(N_SAMPLES)

    detect = (comp > PRESENT_THRESH).float().mean(dim=0)   # [K] fraction of draws containing each
    found = modes(comp, grain)

    out_modes = []
    for m in found:
        c = m["composition"].detach().cpu()
        g = m["grain_um"].detach().cpu()
        present = (c > REPORT_THRESH).nonzero().flatten().tolist()
        out_modes.append({
            "weight": float(m["weight"]),
            "spread": float(m["spread"]),
            "components": [
                {"code": int(i), "name": NAMES[int(i)], "proportion": float(c[i]),
                 "grain_um": (float(g[i]) if torch.isfinite(g[i]) else None)}
                for i in sorted(present, key=lambda j: -float(c[j]))
            ],
        })
    return {
        "n_modes": len(found),
        "modes": out_modes,
        "detection_frequency": {NAMES[i]: float(detect[i]) for i in range(len(detect))},
    }


def figure(entries, path):
    """Three-panel figure: one exemplar spectrum per DiSCo group, normalized at NORM_LAM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = [(1.5, "H$_2$O"), (2.0, "H$_2$O"), (2.27, "CH$_3$OH"),
             (3.0, "H$_2$O"), (4.27, "CO$_2$"), (4.67, "CO")]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True)
    for ax, (desig, group, sample) in zip(axes, entries):
        lam, R = sample.lams, _norm(sample.values, sample.lams)
        ax.plot(lam, R, color="#4C72B0", lw=0.8)
        for x, lbl in bands:
            ax.axvline(x, color="0.75", ls=":", lw=0.7, zorder=0)
            ax.text(x, ax.get_ylim()[1], lbl, fontsize=5.5, rotation=90,
                    va="top", ha="right", color="0.45")
        ax.set_title(f"{group} — {desig}", fontsize=9)
        ax.set_xlabel("Wavelength (µm)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"Reflectance (norm. at {NORM_LAM} µm)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}")


def main(checkpoint=config.BEST_LOSS_CHECKPOINT):
    device = config.get_device()
    model, _ = load_model(checkpoint, device)

    samples = jwst.process()
    by_desig = dict(zip(jwst.designations(), samples))

    entries, results = [], {}
    for desig, (group, note) in TARGETS.items():
        s = by_desig[desig]
        entries.append((desig, group, s))
        print(f"\n{'=' * 72}\n{group} — {desig}\n  selected: {note}"
              f"\n  spectrum: {s.lams.min():.3f}–{s.lams.max():.3f} µm, {len(s.lams)} channels")
        r = analyze(model, s, device)
        r.update(group=group, selection_note=note,
                 lam_min=float(s.lams.min()), lam_max=float(s.lams.max()),
                 n_channels=int(len(s.lams)))
        results[desig] = r

        print(f"  {r['n_modes']} mode(s) found")
        for j, m in enumerate(r["modes"], 1):
            parts = ", ".join(f"{c['name']} {c['proportion']:.3f}" for c in m["components"])
            print(f"    mode {j} (w={m['weight']:.3f}): {parts}")
        top = sorted(r["detection_frequency"].items(), key=lambda kv: -kv[1])[:8]
        print("  detection frequency (top 8): "
              + ", ".join(f"{n} {f:.2f}" for n, f in top))
        never = [n for n, f in r["detection_frequency"].items() if f == 0.0]
        print(f"  never detected in any draw ({len(never)}): {', '.join(never)}")

    out = config.CHECKPOINT_DIR / "jwst_apply.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n  results -> {out}")
    figure(entries, config.CHECKPOINT_DIR / "jwst_exemplars.png")


if __name__ == "__main__":
    main()
