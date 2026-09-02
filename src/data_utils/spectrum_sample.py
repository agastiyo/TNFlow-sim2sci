"""SpectrumSample: a (wavelength, reflectance) spectrum + its composition target."""

import numpy as np


class SpectrumSample:
    """One spectrum and the mixture it came from.

    lams [D] µm · values [D] reflectance · target [K] proportions (sum 1) ·
    synthetic bool · grains [K] µm per component (NaN where unknown).
    """

    def __init__(self, lams, values, target, synthetic, grains=None):
        self.lams = np.asarray(lams, dtype=np.float32)
        self.values = np.asarray(values, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.synthetic = bool(synthetic)
        if grains is None:
            grains = np.full(self.target.shape, np.nan, dtype=np.float32)
        self.grains = np.asarray(grains, dtype=np.float32)
        assert self.lams.shape == self.values.shape, \
            f"lams ({self.lams.shape}) and values ({self.values.shape}) length mismatch"
        assert self.grains.shape == self.target.shape, \
            f"grains ({self.grains.shape}) and target ({self.target.shape}) length mismatch"

    def __repr__(self):
        kind = "synthetic" if self.synthetic else "lab"
        return f"SpectrumSample({kind}, D={len(self.values)})"
