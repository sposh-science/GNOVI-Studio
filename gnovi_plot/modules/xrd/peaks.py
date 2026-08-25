"""XRD peak detection -- a deliberately small wrapper around
`scipy.signal.find_peaks`, pure numerical code (no Qt, no Matplotlib).

A detected (or manually added) peak is a SEED/CANDIDATE (`XRDPeakSeed`),
not a final, scientifically measured peak position -- profile fitting
(Gaussian/Lorentzian/pseudo-Voigt) is a later milestone's job (see
PROJECT_GUIDE.md's XRD roadmap notes); this module never claims otherwise.
`XRDPeakSeed.width_samples` (when SciPy computes it) is in ARRAY-INDEX
units, not degrees -- it is a detection diagnostic, never a substitute for
a fitted FWHM.
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
class XRDPeakSeed:
    """One peak candidate -- either SciPy `find_peaks` found it
    (`origin=ORIGIN_AUTOMATIC`), or a caller added it directly
    (`origin=ORIGIN_MANUAL`, `index=None`, no SciPy detection metadata).

    `enabled` lets a candidate stay in the list (so a detection pass is
    never silently lost) while being excluded from later analysis --
    the same "soft exclude, don't delete" semantics as
    `plotting.series3d.Series3D.stale`'s own convention of keeping rather
    than discarding state a later step might want back.

    `id` is this seed's own stable identity, independent of `index` (which
    is only meaningful relative to the exact array it was detected in) --
    a future XRD-2 GUI can reference a specific seed (e.g. "remove this
    one") by `id` even after the underlying data/detection has changed.
    """

    two_theta: float
    intensity: float
    origin: str
    index: int | None = None
    prominence: float | None = None
    width_samples: float | None = None
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {
            "two_theta": self.two_theta,
            "intensity": self.intensity,
            "origin": self.origin,
            "index": self.index,
            "prominence": self.prominence,
            "width_samples": self.width_samples,
            "enabled": self.enabled,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "XRDPeakSeed":
        return cls(
            two_theta=data["two_theta"],
            intensity=data["intensity"],
            origin=data["origin"],
            index=data.get("index"),
            prominence=data.get("prominence"),
            width_samples=data.get("width_samples"),
            enabled=data.get("enabled", True),
            id=data.get("id") or uuid.uuid4().hex,
        )

    @classmethod
    def manual(cls, two_theta: float, intensity: float) -> "XRDPeakSeed":
        """A user-added seed, not tied to any detection-array position --
        see this class's own docstring and the module docstring on why
        this is a SEED ("analyze a peak near here"), never a claim about
        the true peak center."""
        return cls(two_theta=two_theta, intensity=intensity, origin=ORIGIN_MANUAL)


def detect_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    prominence: float | None = None,
    distance: float | None = None,
    height: float | None = None,
    width: float | None = None,
) -> list[XRDPeakSeed]:
    """Detect peak candidates via `scipy.signal.find_peaks`.

    Primary parameters (the ones a researcher should normally set):
    `prominence` (how much a peak stands out above its surroundings --
    the most physically meaningful threshold for "is this a real peak")
    and `distance` (minimum separation, in samples, between detected
    peaks). `height`/`width` are advanced/optional filters, left `None`
    (SciPy's own "not applied") unless a caller explicitly sets them --
    this wrapper does not expose every `find_peaks` parameter, only these
    four, matching the "deliberately small API" this milestone commits to.

    Returns structured `XRDPeakSeed` candidates (never raw SciPy indices)
    with `origin=ORIGIN_AUTOMATIC` -- ordered exactly as SciPy returns
    them (ascending index / ascending `two_theta`, since `two_theta` is
    assumed monotonic increasing, as an imported XRD pattern always is).

    Raises `InvalidPeakDetectionError` for a shape mismatch, non-finite
    input, or a parameter SciPy itself rejects (e.g. negative `distance`)
    -- wrapped for the same reason `analysis.fitting.fit_curve` wraps a
    solver failure into `FitError`, never left as a raw SciPy exception a
    caller has to know to expect.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if two_theta.shape != intensity.shape:
        raise InvalidPeakDetectionError(
            f"two_theta and intensity must have the same shape "
            f"(got {two_theta.shape} and {intensity.shape})"
        )
    if not np.all(np.isfinite(two_theta)) or not np.all(np.isfinite(intensity)):
        raise InvalidPeakDetectionError(
            "two_theta and intensity must be entirely finite -- clean or "
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

    seeds: list[XRDPeakSeed] = []
    for position, idx in enumerate(indices):
        seeds.append(
            XRDPeakSeed(
                two_theta=float(two_theta[idx]),
                intensity=float(intensity[idx]),
                origin=ORIGIN_AUTOMATIC,
                index=int(idx),
                prominence=float(prominences[position]) if prominences is not None else None,
                width_samples=float(widths[position]) if widths is not None else None,
            )
        )
    return seeds
