"""JWST DiSCo-TNOs PRISM spectra -> unlabelled SpectrumSamples (apply-time only)."""

import numpy as np

from src.config import DATA_DIR as DATA_ROOT
from src.data_utils.components import K
from src.data_utils.spectrum_sample import SpectrumSample

DATA_DIR = DATA_ROOT / "JWST_DiSCO-TNOs" / "v13" / "DISCO_ALL"


def _paths():
    return sorted(DATA_DIR.glob("*_spec_solcorr.txt"))


def designations():
    """Object designation per spectrum, index-aligned with ``process()`` (list[str])."""
    return [p.stem.split("_PRISM")[0] for p in _paths()]


def process():
    """Returns: list[SpectrumSample] (real, unlabelled; target all-zero)."""
    samples = []
    for path in _paths():
        arr = np.loadtxt(path, comments="#", usecols=(0, 1))
        target = np.zeros(K, dtype=np.float32)
        samples.append(SpectrumSample(arr[:, 0], arr[:, 1], target, synthetic=False))
    print(f"  jwst: {len(samples)} samples")
    return samples
