"""Shkuratov RT model grid -> synthetic SpectrumSamples (one per bestfit.*.txt)."""

import numpy as np

from src.config import DATA_DIR as DATA_ROOT
from src.data_utils.components import Component, DroppedComponentError, K
from src.data_utils.spectrum_sample import SpectrumSample

DATA_DIR = DATA_ROOT / "train"


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def _read_sections(path):
    """Split a bestfit file into {KEYWORD: [body lines]}."""
    sections, current = {}, None
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            first = line.split()[0]
            if not _is_number(first):
                current = first
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(line)
    return sections


def _parse_model(body):
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
        if len(tokens) < 2:
            continue
        *numbers, material = tokens
        code = Component.encode(material)
        target[code] = float(numbers[3])  # 4th numeric field: proportion
        grains[code] = float(numbers[0])  # 1st numeric field: grain size (µm)
    return target, grains


def process():
    """Returns: list[SpectrumSample] (synthetic).

    Spectra containing a component dropped in the K=25 -> K=15 reduction are excluded here,
    and counted separately from genuine parse failures so a silent enum bug can't hide inside
    the deliberate-drop count.
    """
    samples, empty, dropped, unknown = [], 0, 0, 0
    for path in sorted(DATA_DIR.rglob("bestfit.*.txt")):
        sections = _read_sections(path)
        lams, values = _parse_model(sections.get("MODEL", []))
        if len(values) == 0:
            empty += 1
            continue
        try:
            target, grains = _parse_components(sections.get("COMPONENTS", []))
        except DroppedComponentError:
            dropped += 1      # deliberate: contains a component we removed
            continue
        except ValueError:
            unknown += 1      # a title the enum does not know -- investigate if nonzero
            continue
        samples.append(SpectrumSample(lams, values, target, synthetic=True, grains=grains))
    print(f"  shkuratov: {len(samples)} samples "
          f"({dropped} dropped-component, {empty} empty, {unknown} unknown-title)")
    if unknown:
        print(f"  WARNING: {unknown} spectra had unrecognized component titles")
    return samples
