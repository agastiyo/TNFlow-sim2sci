"""Component material <-> integer code mapping for the CNF target.

K=20, codes 0..19. Originally reduced K=25 -> K=15; five materials
now available as SDOC optical constants and generated in bulk by generate_shkuratov.py -- CO,
C2H6, C2H4, NH3, Pyroxene (codes 15..19) -- have since been re-added. Spectra containing a
still-dropped component are skipped at build time: ``encode`` raises ``DroppedComponentError``
for their titles, which the source processors catch.
"""

from enum import Enum

import numpy as np

# File titles that aren't valid Python identifiers -> canonical member name.
_TITLE_ALIASES: dict[str, str] = {
    "N2_beta_(roush/brown)": "N2",
}

# Titles still dropped after K=25 -> K=15 reduction and re-addition of CO/C2H6/C2H4/NH3/Pyroxene.
# In-N2 mixtures (C2H6inN2, CH4inN2, N2:CH4:CO) stay dropped: distinct from the pure ices.
DROPPED_TITLES: frozenset[str] = frozenset({
    "C2H6inN2", "CH4inN2", "HCN", "Irr_N2CH4CO",
    "NH4OH", "PM80", "TagishLake",
    # alias spellings of the above
    "Ethane_in_N2", "Tagish_Lake", "NH4OH_3pc", "NH4OH_BS",
    "HCN_MastersonKanna", "N2:CH4:CO",
})


class DroppedComponentError(ValueError):
    """Raised by ``encode`` for a component removed in the K=25 -> K=15 reduction.

    Subclasses ValueError so existing ``except ValueError`` skip paths keep working.
    """


class Component(Enum):
    """Molecule (member name) -> integer code (member value). K=20, codes 0..19."""

    # simple molecular ices
    N2 = 0
    COinN2 = 1
    CO2 = 2
    H2O = 3

    # small organics / hydrocarbons
    CH4 = 4
    CH3OH = 5

    # salt
    NaCl = 6

    # amorphous carbon family
    AC = 7
    HAC = 8

    # tholins (complex refractory organics)
    TTH = 9
    IT2 = 10
    TrT = 11

    # silicate minerals
    Oliv = 12
    serp = 13

    # meteoritic / complex multi-component samples
    PM100 = 14

    # expanded set: re-added once SDOC optical constants + bulk synthetic data became available
    CO = 15         # pure carbon monoxide ice (distinct from COinN2=1)
    C2H6 = 16       # ethane
    C2H4 = 17       # ethylene
    NH3 = 18        # ammonia
    Pyroxene = 19   # pyroxene silicate

    # aliases: file titles / alternate spellings sharing a canonical value
    N2_43K_Quirico = 0
    COinN2_Quirico = 1
    CO2_hansen = 2
    H20_KBO_40K = 3
    H20_KBO_100K = 3
    H20_KBO_120K_RC = 3
    H2O_amorph = 3
    H20_JC = 3
    CH4_Grundy = 4
    CH4_43K_Quirico = 4
    ch3oh_90k_brown = 5
    NaCl1 = 6
    AC_v0 = 7
    Titan_Tholin = 9
    TTHlpshortlr = 9
    Icetholin2 = 10
    Triton_Tholin = 11
    Oliv1 = 12
    oliv = 12
    Josh_serp = 13
    Ethane = 16       # old bestfit spelling for pure ethane
    NH3_JR = 18       # old bestfit spelling for pure ammonia

    @classmethod
    def encode(cls, title: str) -> int:
        """Integer code for a material title.

        Raises:  DroppedComponentError if the title is in DROPPED_TITLES (caller should
                 skip the spectrum); ValueError if unrecognized entirely (a real bug).
        """
        if title in DROPPED_TITLES:
            raise DroppedComponentError(
                f"Component {title!r} is in DROPPED_TITLES; "
                f"spectra containing it are excluded from the dataset."
            )
        if title in _TITLE_ALIASES:
            title = _TITLE_ALIASES[title]
        try:
            return cls[title].value
        except KeyError:
            raise ValueError(
                f"Unknown component title {title!r}: not present in the "
                f"Component enum. Add it to {__name__}.Component."
            ) from None

    @classmethod
    def decode(cls, code: int) -> str:
        """Material title (canonical member name) for an integer code."""
        return cls(code).name


K = len(set(c.value for c in Component))  # target vector length (aliases share values)

# Assumed grain size (µm) for lab spectra (thin films, no measured grain).
# Retained for the Tier-2 lab-data path; lab sources are not built into the current dataset.
LAB_GRAIN_UM = 20.0


def target_from_mix(mix: dict[str, float]) -> np.ndarray:
    """Build a K-length, sum-to-1 target vector from a {material: weight} mixture.

    Args:    mix: {material title: weight}.
    Returns: ndarray [K] float32, normalized to sum 1.
    """
    target = np.zeros(K, dtype=np.float32)
    total = sum(mix.values())
    for material, weight in mix.items():
        target[Component.encode(material)] = weight / total
    return target


def grains_from_mix(mix: dict[str, float], grain_um: float = LAB_GRAIN_UM) -> np.ndarray:
    """Grain vector: ``grain_um`` for each material in ``mix``, NaN elsewhere.

    Args:    mix: {material title: weight}; grain_um: grain size to assign.
    Returns: ndarray [K] float32 (NaN where the material is absent).
    """
    grains = np.full(K, np.nan, dtype=np.float32)
    for material in mix:
        grains[Component.encode(material)] = grain_um
    return grains
