"""Raman peak detection -- a deliberately small wrapper around
`scipy.signal.find_peaks`, pure numerical code (no Qt, no Matplotlib).

A detected (or manually added) peak is a SEED/CANDIDATE (`RamanPeakSeed`),
not a final, scientifically measured peak position -- peak-profile fitting
is explicitly outside this milestone (see Issue #72). This module never
claims otherwise. `RamanPeakSeed.width_samples` (when SciPy computes it) is
in ARRAY-INDEX units, not cm^-1 -- it is a detection diagnostic, never a
substitute for a fitted linewidth/FWHM.

Deliberately independent of `modules.xrd`: the underlying `find_peaks`
wrapping is structurally similar to `modules.xrd.peaks.detect_peaks`, but
Raman does not import XRD's result types (`XRDPeakSeed`) or any other
XRD-specific dataclass -- this module owns its own peak representation
with Raman-appropriate field names (`raman_shift`, not `two_theta`), so a
Raman-only install/use never carries an XRD-shaped dependency.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

ORIGIN_AUTOMATIC = "automatic"
ORIGIN_MANUAL = "manual"


class InvalidPeakDetectionError(ValueError):
    """Raised for invalid peak-detection input/parameters: mismatched
    array shapes, non-finite data, or a non-physical parameter (e.g. a
    negative `distance`) that SciPy itself rejects."""


@dataclass
class RamanPeakSeed:
    """One peak candidate -- either SciPy `find_peaks` found it
    (`origin=ORIGIN_AUTOMATIC`), or a caller added it directly
    (`origin=ORIGIN_MANUAL`, `index=None`, no SciPy detection metadata).

    `raman_shift` is the peak's position in cm^-1 (the x-axis unit Raman
    spectra are conventionally reported in) -- deliberately not called
    `position`/`x`, matching the same "name the field for what it
    physically is" convention `modules.xrd.peaks.XRDPeakSeed.two_theta`
    already established for its own domain.

    `enabled` lets a candidate stay in the list (so a detection pass is
    never silently lost) while being excluded from later analysis -- the
    same "soft exclude, don't delete" semantics
    `modules.xrd.peaks.XRDPeakSeed.enabled` already uses.

    `id` is this seed's own stable identity, independent of `index`
    (which is only meaningful relative to the exact array it was detected
    in) -- a future GUI can reference a specific seed (e.g. "remove this
    one") by `id` even after the underlying data/detection has changed.
    """

    raman_shift: float
    intensity: float
    origin: str
    index: int | None = None
    prominence: float | None = None
    width_samples: float | None = None
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {
            "raman_shift": self.raman_shift,
            "intensity": self.intensity,
            "origin": self.origin,
            "index": self.index,
            "prominence": self.prominence,
            "width_samples": self.width_samples,
            "enabled": self.enabled,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RamanPeakSeed":
        return cls(
            raman_shift=data["raman_shift"],
            intensity=data["intensity"],
            origin=data["origin"],
            index=data.get("index"),
            prominence=data.get("prominence"),
            width_samples=data.get("width_samples"),
            enabled=data.get("enabled", True),
            id=data.get("id") or uuid.uuid4().hex,
        )

    @classmethod
    def manual(cls, raman_shift: float, intensity: float) -> "RamanPeakSeed":
        """A user-added seed, not tied to any detection-array position --
        see this class's own docstring and the module docstring on why
        this is a SEED ("analyze a peak near here"), never a claim about
        the true peak center."""
        return cls(raman_shift=raman_shift, intensity=intensity, origin=ORIGIN_MANUAL)


def detect_raman_peaks(
    raman_shift: np.ndarray,
    intensity: np.ndarray,
    *,
    prominence: float | None = None,
    distance: float | None = None,
    height: float | None = None,
    width: float | None = None,
) -> list[RamanPeakSeed]:
    """Detect peak candidates via `scipy.signal.find_peaks`.

    Primary parameters (the ones a researcher should normally set):
    `prominence` (how much a peak stands out above its surroundings --
    the most physically meaningful threshold for "is this a real peak")
    and `distance` (minimum separation, in samples, between detected
    peaks). `height`/`width` are advanced/optional filters, left `None`
    (SciPy's own "not applied") unless a caller explicitly sets them --
    this wrapper does not expose every `find_peaks` parameter, only these
    four, matching the same "deliberately small API" convention
    `modules.xrd.peaks.detect_peaks` already established.

    Returns structured `RamanPeakSeed` candidates (never raw SciPy
    indices) with `origin=ORIGIN_AUTOMATIC` -- ordered exactly as SciPy
    returns them (ascending index / ascending `raman_shift`, since
    `raman_shift` is assumed monotonic increasing, as an imported Raman
    spectrum always is).

    `width`, when SciPy computes it, is returned on each seed as
    `width_samples` -- an ARRAY-INDEX detection diagnostic, never cm^-1,
    never a fitted linewidth/FWHM. There is no fitted quantity anywhere
    in this function; peak-profile fitting is explicitly outside this
    milestone (see Issue #72).

    Raises `InvalidPeakDetectionError` for a shape mismatch, non-finite
    input, or a parameter SciPy itself rejects (e.g. negative `distance`)
    -- wrapped for the same reason `modules.xrd.peaks.detect_peaks` wraps
    a SciPy failure, never left as a raw SciPy exception a caller has to
    know to expect.
    """
    raman_shift = np.asarray(raman_shift, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if raman_shift.shape != intensity.shape:
        raise InvalidPeakDetectionError(
            f"raman_shift and intensity must have the same shape "
            f"(got {raman_shift.shape} and {intensity.shape})"
        )
    if not np.all(np.isfinite(raman_shift)) or not np.all(np.isfinite(intensity)):
        raise InvalidPeakDetectionError(
            "raman_shift and intensity must be entirely finite -- clean or "
            "remove non-finite values before peak detection"
        )

    try:
        indices, properties = find_peaks(
            intensity, prominence=prominence, distance=distance, height=height, width=width
        )
    except Exception as exc:  # SciPy raises plain ValueError for bad params
        raise InvalidPeakDetectionError(f"Peak detection failed: {exc}") from exc

    prominences = properties.get("prominences")
    widths = properties.get("widths")

    seeds: list[RamanPeakSeed] = []
    for position, idx in enumerate(indices):
        seeds.append(
            RamanPeakSeed(
                raman_shift=float(raman_shift[idx]),
                intensity=float(intensity[idx]),
                origin=ORIGIN_AUTOMATIC,
                index=int(idx),
                prominence=float(prominences[position]) if prominences is not None else None,
                width_samples=float(widths[position]) if widths is not None else None,
            )
        )
    return seeds
