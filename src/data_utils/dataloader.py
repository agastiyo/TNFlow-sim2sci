"""SpectrumSample list -> train / test batches.

One workhorse, ``_batch``, builds length-homogeneous batches and applies every augmentation
(each behind a flag, synthetic rows only). ``__call__`` returns one epoch of two ``_batch`` calls.
"""

from collections import namedtuple

import numpy as np
import torch

from src.data_utils import store
from src.data_utils.components import K

ATTN_BL2_BUDGET = 300_000_000  # self-attention ~ O(B·L²); caps rows/batch on long buckets (L up to ~9601)

NOISE_SIGMA0_MAX = 0.04
NOISE_ADD_MAX = 0.04
NOISE_LAM_REF = 0.9


def add_noise(values, lams, rng):
    """R -> R*(1+eps_mult) + eps_add.

    Args:    values, lams: ndarray [B, L]; rng: np.random.Generator.
    Returns: ndarray [B, L] noised reflectance.
    """
    b = values.shape[0]
    sig0 = rng.normal(0.0, NOISE_SIGMA0_MAX, (b, 1))
    eps_mult = rng.standard_normal(values.shape) * sig0 * np.sqrt(lams / NOISE_LAM_REF)
    a = rng.normal(0.0, NOISE_ADD_MAX, (b, 1))
    eps_add = rng.standard_normal(values.shape) * a * np.median(values, axis=1, keepdims=True)
    return values * (1.0 + eps_mult) + eps_add


Epoch = namedtuple("Epoch", [
    "train_spectra", "train_lams", "train_targets", "train_grains",
    "test_spectra", "test_lams", "test_targets", "test_grains",
])


class DataLoader:
    """Builds train/test/ood sample lists from a DatasetSpec and serves length-bucketed batches."""

    def __init__(self, spec=None, batches=512, noise_seed=10025734):
        self.batches = batches
        self.rng = np.random.default_rng(noise_seed)
        splits = store.build_splits(store.load_manifest(), spec or store.DEFAULT_SPEC)
        self.train = store.load_samples(splits["train"])
        self.test = store.load_samples(splits["test"])
        self.ood = store.load_samples(splits["ood"])

    def _batch(self, samples, shuffle=False, noise=False, numpy=False):
        """One split -> (spectra, lams, targets, grains), each a list of length-homogeneous batches.

        Noise augmentation acts on synthetic rows only and is re-drawn each call. To add another
        augmentation: give it a flag and a block in the augmentation section below. Val splits pass
        ``noise=False`` (clean).
        """
        if not samples:
            return [], [], [], []
        if shuffle:
            samples = [samples[i] for i in self.rng.permutation(len(samples))]

        n = len(samples)
        spectra, lams = np.empty(n, object), np.empty(n, object)
        targets, grains = np.zeros((n, K), np.float32), np.zeros((n, K), np.float32)
        synth = np.empty(n, bool)
        for i, s in enumerate(samples):
            spectra[i], lams[i], targets[i], grains[i], synth[i] = \
                s.values, s.lams, s.target, s.grains, s.synthetic

        lengths = np.fromiter((len(r) for r in spectra), int, n)
        rows = -(-n // min(self.batches, n))  # ceil: target rows per length-homogeneous chunk
        sb, lb, tb, gb = [], [], [], []
        for length in np.unique(lengths):
            idx = np.nonzero(lengths == length)[0]
            cap = max(1, min(rows, ATTN_BL2_BUDGET // (int(length) ** 2)))
            for start in range(0, len(idx), cap):
                chunk = idx[start:start + cap]
                sc, lc, syn = np.stack(spectra[chunk]), np.stack(lams[chunk]), synth[chunk]

                # ---- augmentation: synthetic rows only, re-drawn each call ----
                if noise:
                    sc = np.where(syn[:, None], add_noise(sc, lc, self.rng), sc)

                sb.append(sc.astype(np.float32)); lb.append(lc.astype(np.float32))
                tb.append(targets[chunk]); gb.append(grains[chunk])

        if shuffle:  # interleave resolution buckets so an epoch isn't one bucket at a time
            order = self.rng.permutation(len(sb))
            sb, lb, tb, gb = [sb[i] for i in order], [lb[i] for i in order], [tb[i] for i in order], [gb[i] for i in order]
        if numpy:
            return sb, lb, tb, gb
        return ([torch.tensor(x) for x in sb], [torch.tensor(x) for x in lb],
                [torch.tensor(x) for x in tb], [torch.tensor(x) for x in gb])

    def __call__(self):
        """One epoch: train is shuffled + noise-augmented; test stays clean."""
        return Epoch(
            *self._batch(self.train, shuffle=True, noise=True),
            *self._batch(self.test, shuffle=False, noise=True),
        )
