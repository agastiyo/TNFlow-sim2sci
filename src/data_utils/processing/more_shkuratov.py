"""more_shkuratov: CANA/SDOC synthetic spectra -> SpectrumSamples (one per shk_*.txt).

Ingests the files produced by ``src/data_utils/generate_shkuratov.py`` (format below), which is
distinct from the ``bestfit.*.txt`` MODEL grid read by :mod:`processing.shkuratov`::

    POROSITY
    <value>                     # per-spectrum scalar, not used by the label
    COMPONENTS
    <name> <proportion> <grain_um>
    ...
    SPECTRUM
    <wavelength_um> <reflectance>
    ...

All 14 generated materials are in the K=20 Component enum, so every well-formed spectrum is
ingested; a component in ``DROPPED_TITLES`` still counts as *dropped* and a truly unrecognized
title as *unknown* (a real bug). Porosity is parsed but discarded -- SpectrumSample carries no
porosity field.
"""

import numpy as np

from src.config import DATA_DIR as DATA_ROOT
from src.data_utils.components import Component, DroppedComponentError, K
from src.data_utils.spectrum_sample import SpectrumSample

DATA_DIR = DATA_ROOT / "train" / "more_shkuratov_080326"

_HEADERS = frozenset({"POROSITY", "COMPONENTS", "SPECTRUM"})


def _read_sections(path):
    """Split a shk_*.txt file into {HEADER: [body lines]}.

    Headers are the fixed keywords in ``_HEADERS``; every other line is body of the current
    section. (Unlike the bestfit reader, COMPONENTS body lines start with a name, not a number,
    so header detection is keyword-based rather than "first token is non-numeric".)
    """
    sections, current = {}, None
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line in _HEADERS:
                current = line
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(line)
    return sections


def _parse_spectrum(body):
    pairs = [(float(r[0]), float(r[1]))
             for r in (line.split() for line in body) if len(r) >= 2]
    if not pairs:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    arr = np.asarray(pairs, np.float32)
    return arr[:, 0], arr[:, 1]


def _parse_components(body):
    target = np.zeros(K, np.float32)
    grains = np.full(K, np.nan, np.float32)
    for line in body:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        name, proportion, grain = tokens[0], tokens[1], tokens[2]
        code = Component.encode(name)   # raises DroppedComponentError / ValueError
        target[code] = float(proportion)
        grains[code] = float(grain)
    return target, grains


def process():
    """Returns: list[SpectrumSample] (synthetic).

    Skip accounting mirrors :func:`processing.shkuratov.process` so a silent enum bug can't hide
    inside the deliberate-drop count.
    """
    samples, empty, dropped, unknown = [], 0, 0, 0
    for path in sorted(DATA_DIR.glob("shk_*.txt")):
        sections = _read_sections(path)
        lams, values = _parse_spectrum(sections.get("SPECTRUM", []))
        if len(values) == 0:
            empty += 1
            continue
        try:
            target, grains = _parse_components(sections.get("COMPONENTS", []))
        except DroppedComponentError:
            dropped += 1      # contains a component in DROPPED_TITLES
            continue
        except ValueError:
            unknown += 1      # a title the enum does not know -- investigate if nonzero
            continue
        samples.append(SpectrumSample(lams, values, target, synthetic=True, grains=grains))
    print(f"  more_shkuratov: {len(samples)} samples "
          f"({dropped} dropped-component, {empty} empty, {unknown} unknown-title)")
    if unknown:
        print(f"  WARNING: {unknown} spectra had unrecognized component titles")
    return samples
