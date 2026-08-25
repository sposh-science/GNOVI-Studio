"""Explicit X-ray radiation/wavelength model for XRD analysis.

GNOVI never guesses a wavelength for a Bragg-law/d-spacing calculation --
every function in `modules.xrd` that needs one takes an explicit `Radiation`
(or a bare `wavelength_angstrom` float), never a silent Cu K-alpha default.
A future GUI may default a *new* analysis to a Cu K-alpha preset, but that
is a UI convenience layered on top of this module, not something this
module does itself.

Wavelength is always in angstrom (A) -- the near-universal unit convention
in XRD literature/software, and GNOVI has no general unit-conversion
framework to route through instead (see PROJECT_GUIDE.md's XRD section).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class InvalidRadiationError(ValueError):
    """Raised for a non-physical or malformed `Radiation` (non-finite,
    zero, or negative wavelength; empty label)."""


@dataclass(frozen=True)
class Radiation:
    """One explicit radiation/wavelength context.

    `label` is a human-readable name (a preset's name, e.g. "Cu Ka1", or
    "Custom") -- purely descriptive, never consulted to derive
    `wavelength_angstrom`: the numeric value is always the actual source of
    truth, the same "descriptive snapshot, never re-derived from it" split
    `AnalysisResult.source_dataset_name` already uses relative to
    `source_dataset_id`.
    """

    label: str
    wavelength_angstrom: float

    def __post_init__(self) -> None:
        if not self.label:
            raise InvalidRadiationError("Radiation.label must not be empty")
        if not math.isfinite(self.wavelength_angstrom):
            raise InvalidRadiationError(
                f"Radiation wavelength must be finite (got {self.wavelength_angstrom!r})"
            )
        if self.wavelength_angstrom <= 0:
            raise InvalidRadiationError(
                f"Radiation wavelength must be positive (got {self.wavelength_angstrom})"
            )

    def to_dict(self) -> dict:
        return {"label": self.label, "wavelength_angstrom": self.wavelength_angstrom}

    @classmethod
    def from_dict(cls, data: dict) -> "Radiation":
        return cls(label=data["label"], wavelength_angstrom=float(data["wavelength_angstrom"]))

    @classmethod
    def custom(cls, wavelength_angstrom: float) -> "Radiation":
        """An explicit, non-preset wavelength -- still validated exactly
        like a preset (see `__post_init__`), just not one of the named
        `RADIATION_PRESETS` below."""
        return cls(label="Custom", wavelength_angstrom=wavelength_angstrom)


# --- Characteristic K-alpha wavelengths --------------------------------------
#
# Values are the standard laboratory-XRD characteristic emission-line
# wavelengths as tabulated by Bearden, J.A. (1967), "X-Ray Wavelengths",
# Rev. Mod. Phys. 39, 78 -- the reference tabulation essentially every XRD
# textbook/software (including GSAS-II and Profex) still cites for these
# numbers. Verified against current secondary literature before being
# hard-coded here (not merely recalled).
#
# Ka1/Ka2 are the two distinct characteristic lines; the commonly-quoted
# single "Ka" value used when Ka1/Ka2 are not separately resolved is their
# intensity-weighted average (Ka1 has ~2x Ka2's intensity, hence the 2:1
# weighting) -- computed here from Ka1/Ka2 directly, rather than a separate
# hard-coded third number, so the two can never silently disagree with each
# other. See `Radiation`'s own docstring / this module's docstring for why
# GNOVI distinguishes Ka1 from the weighted average rather than treating
# them as interchangeable.

CU_KALPHA1_ANGSTROM = 1.540562
CU_KALPHA2_ANGSTROM = 1.544390
CU_KALPHA_ANGSTROM = (2 * CU_KALPHA1_ANGSTROM + CU_KALPHA2_ANGSTROM) / 3

CO_KALPHA1_ANGSTROM = 1.788965
CO_KALPHA2_ANGSTROM = 1.792850
CO_KALPHA_ANGSTROM = (2 * CO_KALPHA1_ANGSTROM + CO_KALPHA2_ANGSTROM) / 3

MO_KALPHA1_ANGSTROM = 0.709300
MO_KALPHA2_ANGSTROM = 0.713590
MO_KALPHA_ANGSTROM = (2 * MO_KALPHA1_ANGSTROM + MO_KALPHA2_ANGSTROM) / 3

# Preset id -> Radiation. Ids are stable keys (never the display label) so
# a future UI/persisted setting can reference a preset by id even if its
# label text is later reworded.
RADIATION_PRESETS: dict[str, Radiation] = {
    "cu_ka1": Radiation("Cu Ka1", CU_KALPHA1_ANGSTROM),
    "cu_ka": Radiation("Cu Ka (weighted)", CU_KALPHA_ANGSTROM),
    "co_ka1": Radiation("Co Ka1", CO_KALPHA1_ANGSTROM),
    "co_ka": Radiation("Co Ka (weighted)", CO_KALPHA_ANGSTROM),
    "mo_ka1": Radiation("Mo Ka1", MO_KALPHA1_ANGSTROM),
    "mo_ka": Radiation("Mo Ka (weighted)", MO_KALPHA_ANGSTROM),
}


def radiation_from_preset(preset_id: str) -> Radiation:
    """Look up a named preset by its stable id (see `RADIATION_PRESETS`).
    Raises `InvalidRadiationError` for an unknown id -- never a silent
    fallback to Cu Ka or any other default (see this module's docstring)."""
    try:
        return RADIATION_PRESETS[preset_id]
    except KeyError:
        raise InvalidRadiationError(
            f"Unknown radiation preset {preset_id!r}; expected one of {sorted(RADIATION_PRESETS)}"
        ) from None
