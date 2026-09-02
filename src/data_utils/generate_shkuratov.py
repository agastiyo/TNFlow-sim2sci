"""Generate synthetic reflectance spectra with the CANA Shkuratov intimate-mixture model.

Draws random mixtures of optical constants from SDOC (Small Database of Optical Constants)
and runs them through :class:`cana.composition.IntimateMixture` (Shkuratov 1999 model).

Batches: 10,000 spectra each for 2, 3, 4 and 5 components (40,000 total). For every spectrum
the components, their proportions, per-component grain sizes and the (shared) porosity are
drawn at random; each spectrum is written as its own text file under ``data/train/more_shkuratov``.

File format (one file per spectrum)::

    POROSITY
    <value>
    COMPONENTS
    <name> <proportion> <grain_size_um>
    ...
    SPECTRUM
    <wavelength_um> <reflectance>
    ...

NOTE ON DEPENDENCIES
--------------------
CANA declares ``numba`` as a dependency but uses it only as a no-op ``@jit`` decorator on a
pure-numpy static method. numba/llvmlite does not build on this machine, so we install cana
and sdoc with ``--no-deps`` and inject a tiny no-op ``numba`` shim (below) before importing
cana. The Shkuratov math is unaffected.

Run with::

    python3.10 -m src.data_utils.generate_shkuratov
"""

from __future__ import annotations

import sys
import types

# --- no-op numba shim (must be installed before importing cana) ----------------------------
if "numba" not in sys.modules:  # pragma: no cover - environment shim
    _numba = types.ModuleType("numba")

    def _jit(*args, **kwargs):
        # Support both @jit and @jit(nopython=True) decorator forms.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    _numba.jit = _jit
    _numba.njit = _jit
    sys.modules["numba"] = _numba
# -------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import sdoc
from cana.composition import IntimateMixture, read_constant

from src.config import DATA_DIR

# --- configuration -------------------------------------------------------------------------

OUT_DIR = DATA_DIR / "train" / "more_shkuratov_080326"

N_PER_BATCH = 50_000
N_COMPONENTS_BATCHES = (2, 3, 4)

# Wavelength grid (um): 0.35 -> 5.0 at 0.01 step. SDOC constants end at 4.99 um; values beyond
# that are clamped by the interpolation (edge-held), which is fine at the very red end.
WAVELENGTHS = np.round(np.arange(0.35, 5.0 + 1e-9, 0.01), 2)

# Grain size (um): log-uniform in [5, 100], drawn independently per component.
GRAIN_MIN_UM, GRAIN_MAX_UM = 5.0, 100.0
# Porosity: uniform in [0.0, 0.7], one value per spectrum.
POROSITY_MIN, POROSITY_MAX = 0.0, 0.7
# Floor on the smallest mixture proportion so no component is effectively absent.
MIN_PROPORTION = 0.05
# Max redraws when a parameter combo yields a non-finite spectrum (Shkuratov albedo formula
# has no real solution -- aux < 1 -- for some highly absorbing / porous mixtures).
MAX_REDRAWS = 50

SEED = 10025734

# Component library: name written to the file -> SDOC optical-constant id.
#   * The first 9 are the components present in src/data_utils/components.py that have genuine
#     optical constants in SDOC (names kept encodable by Component.encode).
#   * The last 5 are extra SDOC materials (per user request) to widen diversity. Several of
#     these (CO, C2H6, NH3) are in components.py's DROPPED_TITLES and C2H4 / Pyroxene are not
#     in the enum at all -- they will not encode to the K=15 target without a mapping update.
# Where a material has several SDOC entries (temperature/phase variants) one representative is
# chosen; swap the id here to use a different variant.
COMPONENT_LIBRARY: dict[str, str] = {
    # -- present in components.py, real SDOC match --
    "N2": "nit_0",       # Nitrogen, 36.5 K
    "CO2": "cod_1",      # Carbon dioxide, Hansen et al. 1997
    "H2O": "wat_2",      # Water, amorphous 40 K (Mastrapa 2008/2009)
    "CH4": "met_0",      # Methane, crystalline 39 K (Grundy 2002)
    "CH3OH": "mol_0",    # Methanol, 100 K (Hudgins 1993 / Brown 1995)
    "AC": "car_0",       # Amorphous carbon (Rouleau & Martin 1991)
    "TTH": "tit_0",      # Titan tholin (Khare et al. 1984)
    "TrT": "trt_0",      # Triton tholin (Khare)
    "Oliv": "oli_0",     # Olivine Mg_2y Fe_2-2y SiO4, y=0.5 (Dorschner 1995)
    # -- extra SDOC materials (not all in components.py) --
    "C2H6": "eta_0",     # Ethane, crystalline 30 K (Hudson 2014)
    "C2H4": "eth_0",     # Ethylene, amorphous 30 K (Hudson 2014)
    "NH3": "amo_3",      # Ammonia, amorphous 40 K (Roser 2021)
    "CO": "com_0",       # Carbon monoxide, 10 K (PubChem)
    "Pyroxene": "pyr_1",  # Pyroxene Mg_x Fe_1-x SiO3, x=0.50 (Dorschner 1995)
}

# -------------------------------------------------------------------------------------------


def load_optical_constants() -> dict[str, object]:
    """Read every optical constant in ``COMPONENT_LIBRARY`` from SDOC once, pre-rebased.

    Returns a ``{name: OpticalConstant}`` dict interpolated onto ``WAVELENGTHS`` so the per-
    spectrum mixture builds do not re-read/re-interpolate the raw SDOC data every time.
    """
    sdb = sdoc.SDOC(mode="r")
    ocs: dict[str, object] = {}
    for name, cid in COMPONENT_LIBRARY.items():
        label, data = sdb.get_constant(cid)
        oc = read_constant(data, label=name)
        ocs[name] = oc.rebase(baseaxis=WAVELENGTHS)
    sdb.close()
    return ocs


def draw_proportions(rng: np.random.Generator, k: int) -> np.ndarray:
    """Draw ``k`` proportions summing to 1, each at least ``MIN_PROPORTION`` (Dirichlet)."""
    while True:
        p = rng.dirichlet(np.ones(k))
        if p.min() >= MIN_PROPORTION:
            return p


def draw_grainsizes(rng: np.random.Generator, k: int) -> np.ndarray:
    """Draw ``k`` grain sizes (um), log-uniform in [GRAIN_MIN_UM, GRAIN_MAX_UM]."""
    log_lo, log_hi = np.log10(GRAIN_MIN_UM), np.log10(GRAIN_MAX_UM)
    return 10.0 ** rng.uniform(log_lo, log_hi, size=k)


def write_spectrum(path: Path, names, proportions, grains, porosity, reflectance) -> None:
    """Write one spectrum file in the COMPONENTS / SPECTRUM text format."""
    lines = [
        "POROSITY",
        f"{porosity:.6f}",
        "COMPONENTS",
    ]
    for name, prop, grain in zip(names, proportions, grains):
        lines.append(f"{name} {prop:.6f} {grain:.4f}")
    lines.append("SPECTRUM")
    for wl, refl in zip(WAVELENGTHS, reflectance):
        lines.append(f"{wl:.2f} {refl:.6f}")
    path.write_text("\n".join(lines) + "\n")


def generate() -> None:
    """Generate all batches and write files to ``OUT_DIR``."""
    rng = np.random.default_rng(SEED)
    ocs = load_optical_constants()
    names_all = list(COMPONENT_LIBRARY.keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total, redraws = 0, 0
    for k in N_COMPONENTS_BATCHES:
        print(f"Generating {N_PER_BATCH} spectra with {k} components ...", flush=True)
        for i in range(N_PER_BATCH):
            # Redraw the parameters (and, as a last resort, the component set) until the model
            # returns a fully finite spectrum. NaNs arise where the Shkuratov albedo has no
            # real solution; such combos are physically degenerate and are simply resampled.
            for attempt in range(MAX_REDRAWS):
                names = list(rng.choice(names_all, size=k, replace=False))
                proportions = draw_proportions(rng, k)
                grains = draw_grainsizes(rng, k)
                porosity = float(rng.uniform(POROSITY_MIN, POROSITY_MAX))

                mix = IntimateMixture(
                    [ocs[n] for n in names],
                    grainsizes=list(grains),
                    proportions=list(proportions),
                    porosity=porosity,
                )
                with np.errstate(invalid="ignore"):
                    spec, _albedo = mix.make(wavelengths=WAVELENGTHS)
                reflectance = np.asarray(spec.r, dtype=float)
                if np.isfinite(reflectance).all():
                    break
                redraws += 1
            else:
                raise RuntimeError(
                    f"Could not draw a finite {k}-component spectrum in {MAX_REDRAWS} attempts"
                )

            total += 1
            path = OUT_DIR / f"shk_{k}c_{i:05d}.txt"
            write_spectrum(path, names, proportions, grains, porosity, reflectance)

            if (i + 1) % 1000 == 0:
                print(f"  {k}c: {i + 1}/{N_PER_BATCH}", flush=True)

    print(f"Done: wrote {total} spectra to {OUT_DIR} ({redraws} redraws)", flush=True)


if __name__ == "__main__":
    generate()
