"""Bragg's-law d-spacing calculation -- pure numerical code, no Qt, no
Matplotlib, no Dataset.

For ordinary first-order (n = 1) powder XRD: n*lambda = 2*d*sin(theta), so
d = lambda / (2 * sin(theta)). The experimental x-axis is conventionally
2*theta in DEGREES; theta = two_theta / 2, and the trigonometric
calculation is always done in radians internally (see `d_spacing`) -- the
one place in this module degree/radian conversion happens, so no caller
needs to think about it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class InvalidBraggInputError(ValueError):
    """Raised for a non-physical or non-finite Bragg-law input: a
    non-finite/non-positive wavelength, or a 2*theta outside the physically
    valid (0, 180) degree range for first-order diffraction."""


def d_spacing(
    two_theta_deg: float | Sequence[float] | np.ndarray, wavelength_angstrom: float
) -> float | np.ndarray:
    """d-spacing (angstrom) for `two_theta_deg` (degrees) at
    `wavelength_angstrom` -- first-order Bragg's law, `d = lambda / (2 *
    sin(theta))` with `theta = two_theta / 2`.

    Accepts a scalar or an array-like `two_theta_deg`; returns a `float`
    for scalar input, a `np.ndarray` (same shape) for array input -- never
    silently squashes an array to a scalar or vice versa.

    Raises `InvalidBraggInputError` if `wavelength_angstrom` is not finite
    and positive, or if any `two_theta_deg` value is not finite or falls
    outside the open interval (0, 180) degrees -- at or beyond either bound
    theta is 0 deg or >=90 deg, where sin(theta) is 0 (undefined d) or the
    diffraction geometry is no longer physically meaningful for ordinary
    powder XRD. Never silently clips or coerces an invalid value.
    """
    if not math.isfinite(wavelength_angstrom):
        raise InvalidBraggInputError(
            f"wavelength_angstrom must be finite (got {wavelength_angstrom!r})"
        )
    if wavelength_angstrom <= 0:
        raise InvalidBraggInputError(
            f"wavelength_angstrom must be positive (got {wavelength_angstrom})"
        )

    scalar_input = np.ndim(two_theta_deg) == 0
    two_theta = np.asarray(two_theta_deg, dtype=float)

    if not np.all(np.isfinite(two_theta)):
        raise InvalidBraggInputError("two_theta_deg must be finite")
    if np.any(two_theta <= 0) or np.any(two_theta >= 180):
        raise InvalidBraggInputError(
            "two_theta_deg must be strictly between 0 and 180 degrees for "
            "first-order Bragg diffraction (got a value outside that range)"
        )

    theta_rad = np.radians(two_theta / 2.0)
    d = wavelength_angstrom / (2.0 * np.sin(theta_rad))
    return float(d) if scalar_input else d
