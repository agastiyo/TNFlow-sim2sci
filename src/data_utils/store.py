"""Parquet-backed sample store + declarative dataset splits.

Build:  python3.10 -m src.data_utils.store   (writes manifest.parquet + spectra.parquet)

``mixture`` + ``proportion`` ids together identify a composition; ``build_splits`` keeps every
spectrum sharing a composition (a duplicate group) in the same split.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src.data_utils.components import Component
from src.data_utils.processing import more_shkuratov, shkuratov
from src.data_utils.spectrum_sample import SpectrumSample

SOURCES = {"shkuratov": shkuratov, "more_shkuratov": more_shkuratov}

PROPORTION_DECIMALS = 6   # round before hashing so float noise can't split a true duplicate group


@dataclass(frozen=True)
class DatasetSpec:
    sources: tuple = ("shkuratov", "more_shkuratov")
    val_frac: float = 0.2
    seed: int = 10025734
    holdout_mixtures: tuple = ()      # explicit component-set ids to hold out as OOD
    n_holdout_mixtures: int = 8       # or randomly hold out this many distinct mixtures (seeded)


DEFAULT_SPEC = DatasetSpec()


def mixture_id(target) -> str:
    """Component-set id: sorted present-component names joined by '+' (proportions/grain ignored)."""
    return "+".join(sorted(Component.decode(int(i)) for i in np.nonzero(target)[0]))


def proportion_id(target) -> str:
    """Proportion id: the present components' proportions, in component-index order, rounded.

    Orthogonal to ``mixture_id`` (which components vs. how much of each)."""
    active = np.nonzero(target)[0]
    vals = np.round(np.asarray(target, dtype=np.float64)[active], PROPORTION_DECIMALS)
    return "|".join(f"{v:.{PROPORTION_DECIMALS}f}" for v in vals)


def build(rebuild=False):
    if config.MANIFEST_PATH.exists() and not rebuild:
        return
    rows, lams, values, targets, grains = [], [], [], [], []
    for source, module in SOURCES.items():
        for i, s in enumerate(module.process()):
            sid = f"{source}:{i:06d}"
            rows.append(dict(id=sid, source=source, synthetic=s.synthetic,
                             mixture=mixture_id(s.target), proportion=proportion_id(s.target),
                             n_components=int(np.count_nonzero(s.target)),
                             n_channels=len(s.values), lam_min=float(s.lams.min()), lam_max=float(s.lams.max())))
            lams.append(s.lams); values.append(s.values); targets.append(s.target); grains.append(s.grains)

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(config.MANIFEST_PATH, index=False)
    f32 = pa.list_(pa.float32())
    pq.write_table(pa.table({
        "id": manifest["id"].tolist(),
        "synthetic": manifest["synthetic"].tolist(),
        "lams": pa.array(lams, f32), "values": pa.array(values, f32),
        "target": pa.array(targets, f32), "grains": pa.array(grains, f32),
    }), config.SPECTRA_PATH)
    print(f"Built {len(manifest)} samples -> {config.MANIFEST_PATH.name} + {config.SPECTRA_PATH.name}")


def load_manifest() -> pd.DataFrame:
    return pd.read_parquet(config.MANIFEST_PATH)


_payload = None


def _unpack(col):
    """List<float32> column -> (flat float32 values, int offsets [N+1]) for O(1) row slicing."""
    arr = col.combine_chunks()
    return arr.values.to_numpy(zero_copy_only=False), arr.offsets.to_numpy()


def _load_payload():
    global _payload
    if _payload is None:
        t = pq.read_table(config.SPECTRA_PATH)
        ids, syn = t.column("id").to_pylist(), t.column("synthetic").to_pylist()
        cols = {n: _unpack(t.column(n)) for n in ("lams", "values", "target", "grains")}
        idx = {sid: i for i, sid in enumerate(ids)}
        _payload = (idx, syn, cols)
    return _payload


def load_samples(ids) -> list:
    idx, syn, cols = _load_payload()
    out = []
    for sid in ids:
        i = idx[sid]
        row = {n: v[o[i]:o[i + 1]] for n, (v, o) in cols.items()}
        out.append(SpectrumSample(row["lams"], row["values"], row["target"],
                                  synthetic=syn[i], grains=row["grains"]))
    return out


def build_splits(manifest, spec) -> dict:
    """(manifest, spec) -> {"train", "test", "ood"} id lists, deterministic from spec.seed.

    1. hold out ``n_holdout_mixtures`` whole component sets -> ood
    2. group the remainder by (mixture, proportion)
    3. shuffle the groups and split ~val_frac by row count, keeping every group intact

    Groups (not rows) are the split unit to prevent near-duplicate leakage.
    """
    rows = manifest[manifest["source"].isin(spec.sources)]
    rng = np.random.default_rng(spec.seed)
    ood_mix = set(spec.holdout_mixtures)
    if spec.n_holdout_mixtures:
        candidates = sorted(set(rows["mixture"]) - ood_mix)
        ood_mix |= set(rng.choice(candidates, spec.n_holdout_mixtures, replace=False))

    is_ood = rows["mixture"].isin(ood_mix)
    ood = rows.loc[is_ood, "id"].tolist()

    rest = rows.loc[~is_ood]
    groups = rest.groupby(["mixture", "proportion"], sort=True)["id"].apply(list)
    keys = list(groups.index)
    order = rng.permutation(len(keys))

    target = int(len(rest) * spec.val_frac)
    test, train, n_test = [], [], 0
    for j in order:
        members = groups.iloc[j]
        if n_test < target:
            test.extend(members); n_test += len(members)
        else:
            train.extend(members)
    return {"train": train, "test": test, "ood": ood}


if __name__ == "__main__":
    build(rebuild=True)
