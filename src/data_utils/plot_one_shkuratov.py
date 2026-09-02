"""Generate a single Shkuratov intimate-mixture spectrum and plot it.

Edit the MIXTURE block below to change components, proportions, grain sizes and
porosity, then run::

    python3.10 -m src.data_utils.plot_one_shkuratov
"""

from __future__ import annotations

import sys
import types

# --- no-op numba shim (must be installed before importing cana) ----------------------------
if "numba" not in sys.modules:  # pragma: no cover - environment shim
    _numba = types.ModuleType("numba")

    def _jit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    _numba.jit = _jit
    _numba.njit = _jit
    sys.modules["numba"] = _numba
# -------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import sdoc
from cana.composition import IntimateMixture, read_constant

# Wavelength grid (um): 0.35 -> 5.0 at 0.01 step.
WAVELENGTHS = np.round(np.arange(0.35, 5.0 + 1e-9, 0.01), 2)

# Component library: name -> SDOC optical-constant id (see generate_shkuratov.py).
COMPONENT_LIBRARY: dict[str, str] = {
    "N2": "nit_0",
    "CO2": "cod_1",
    "H2O": "wat_2",
    "CH4": "met_0",
    "CH3OH": "mol_0",
    "AC": "car_0",
    "TTH": "tit_0",
    "TrT": "trt_0",
    "Oliv": "oli_0",
    "C2H6": "eta_0",
    "C2H4": "eth_0",
    "NH3": "amo_3",
    "CO": "com_0",
    "Pyroxene": "pyr_1",
}

# --- MIXTURE: edit this block for different mixtures ----------------------------------------
# name -> (proportion, grain_size_um). Proportions should sum to ~1.
MIXTURE = {
    "CH3OH": (0.50, 15.3),
    "H2O" : (0.24, 5.0),
    "CO2" : (0.10, 15.3),
    "TrT" : (0.08, 20.8),
    "AC" : (0.05, 28.2),
    "CO" : (0.02, 74.2)
}
POROSITY = 0.1
# -------------------------------------------------------------------------------------------


def load_optical_constant(name: str):
    """Read one optical constant from SDOC, rebased onto WAVELENGTHS."""
    sdb = sdoc.SDOC(mode="r")
    _label, data = sdb.get_constant(COMPONENT_LIBRARY[name])
    oc = read_constant(data, label=name)
    sdb.close()
    return oc.rebase(baseaxis=WAVELENGTHS)


def main() -> None:
    names = list(MIXTURE.keys())
    proportions = [MIXTURE[n][0] for n in names]
    grains = [MIXTURE[n][1] for n in names]

    ocs = [load_optical_constant(n) for n in names]
    mix = IntimateMixture(
        ocs,
        grainsizes=grains,
        proportions=proportions,
        porosity=POROSITY,
    )
    with np.errstate(invalid="ignore"):
        spec, _albedo = mix.make(wavelengths=WAVELENGTHS)
    reflectance = np.asarray(spec.r, dtype=float)

    label = ", ".join(f"{n} {p:.2f}" for n, p in zip(names, proportions))
    plt.figure(figsize=(8, 5))
    plt.plot(WAVELENGTHS, reflectance, lw=1.2)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflectance")
    plt.title(f"Shkuratov mixture: {label} (porosity {POROSITY})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
