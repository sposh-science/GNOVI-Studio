"""XRD data-preparation primitives: background/baseline correction and
optional smoothing -- pure numerical code, no Qt, no Matplotlib.

NON-NEGOTIABLE: every function here returns NEW arrays. None of them ever
mutate their input arrays in place, and none of them ever touch a
`Dataset` -- callers (a future XRD-2 GUI) are responsible for turning an
accepted result into a derived `Dataset`, following the same convention
`gui.widgets.analysis_panel`'s "Add Fit Curve to Plot" already established
for `analysis.fitting.FitResult`: preprocessing/analysis output is a
first-class value the caller decides what to do with, never a live edit to
the source Dataset's own data.

Phase 1 supports exactly two background methods (`polynomial_baseline`,
`arpls_baseline`) and one smoothing method (`savgol_smooth`) -- deliberately
not a menu of algorithms; see PROJECT_GUIDE.md's XRD section for why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

try:
    from pybaselines import Baseline as _PybaselinesBaseline

    _PYBASELINES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the optional
    # 'xrd' extra is not installed; covered by
    # test_arpls_baseline_raises_a_clear_error_without_pybaselines when it
    # is (that test monkeypatches _PYBASELINES_AVAILABLE instead of
    # requiring an environment without the dependency).
    _PYBASELINES_AVAILABLE = False


class InvalidPreprocessingError(ValueError):
    """Raised for invalid preprocessing input/parameters: mismatched array
    shapes, non-finite data, an empty/out-of-bounds baseline-point
    selection, or an invalid smoothing window/polynomial order."""


class PybaselinesNotAvailableError(ImportError):
    """Raised by `arpls_baseline` when the optional `pybaselines` package
    (see the `xrd` extra in pyproject.toml) is not installed. Never raised
    at GNOVI import/startup time -- `pybaselines` is imported lazily, only
    when arPLS is actually requested, so its absence never breaks ordinary
    GNOVI startup or any non-XRD feature."""


@dataclass
class BaselineResult:
    """A computed background estimate plus the corrected signal it implies
    -- both `polynomial_baseline` and `arpls_baseline` return this same
    shape, so a caller (a future XRD-2 preview) never needs method-specific
    branching to show raw/baseline/corrected together.

    `two_theta`/`raw_intensity` echo the exact input arrays this baseline
    was computed against (not copies of a caller's mutable state -- see
    each function's own docstring), so a preview never needs to separately
    track "what was this baseline computed from".
    """

    two_theta: np.ndarray
    raw_intensity: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    method: str
    parameters: dict


@dataclass
class SmoothResult:
    """A smoothed intensity array plus the exact parameters used --
    `savgol_smooth`'s return value. Deliberately does not claim to be
    "the" signal: see `savgol_smooth`'s own docstring on why smoothed data
    must never be presented as raw experimental data."""

    two_theta: np.ndarray
    raw_intensity: np.ndarray
    smoothed_intensity: np.ndarray
    method: str
    parameters: dict


def _validate_xy(two_theta: np.ndarray, intensity: np.ndarray) -> None:
    if two_theta.shape != intensity.shape:
        raise InvalidPreprocessingError(
            f"two_theta and intensity must have the same shape "
            f"(got {two_theta.shape} and {intensity.shape})"
        )
    if not np.all(np.isfinite(intensity)):
        raise InvalidPreprocessingError(
            "intensity must be entirely finite -- clean or remove non-finite "
            "values before running background correction or smoothing"
        )
    if not np.all(np.isfinite(two_theta)):
        raise InvalidPreprocessingError("two_theta must be entirely finite")


def polynomial_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    baseline_indices: list[int],
    *,
    degree: int = 2,
) -> BaselineResult:
    """Fit a degree-`degree` polynomial through `intensity` at only the
    caller-specified `baseline_indices` (positions into `two_theta`/
    `intensity`), then evaluate that polynomial across the FULL
    `two_theta` range to produce a full-length baseline.

    This is deliberately NOT "fit a polynomial through the whole pattern"
    -- doing that would fit the peaks themselves, not the background, and
    silently present the result as if it meant something scientifically.
    `baseline_indices` is the explicit region/point selection the caller
    (a human, or a future automatic peak-avoiding heuristic -- not this
    function) has decided represents background, not peaks. XRD-1 only
    provides this primitive; automatic baseline-region selection is
    explicitly deferred (see PROJECT_GUIDE.md's XRD roadmap notes).

    Raises `InvalidPreprocessingError` for a shape mismatch, non-finite
    input, an out-of-bounds/duplicate-free-but-too-small `baseline_indices`
    (fewer than `degree + 2` points -- the same "more points than free
    parameters" bar `analysis.fitting.fit_curve` already enforces, so an
    exact fit through every point can't be presented as a genuine
    background), or `degree < 0`.

    Never mutates `two_theta`/`intensity`.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    _validate_xy(two_theta, intensity)

    if degree < 0:
        raise InvalidPreprocessingError(f"degree must be >= 0 (got {degree})")

    indices = sorted({int(i) for i in baseline_indices})
    if not indices:
        raise InvalidPreprocessingError("baseline_indices must not be empty")
    if indices[0] < 0 or indices[-1] >= len(two_theta):
        raise InvalidPreprocessingError(
            f"baseline_indices out of bounds for {len(two_theta)}-point data "
            f"(got range {indices[0]}-{indices[-1]})"
        )
    required = degree + 2
    if len(indices) < required:
        raise InvalidPreprocessingError(
            f"Not enough baseline points for a degree-{degree} polynomial "
            f"(found {len(indices)}, need at least {required})"
        )

    baseline_x = two_theta[indices]
    baseline_y = intensity[indices]
    coeffs = np.polyfit(baseline_x, baseline_y, degree)
    baseline = np.polyval(coeffs, two_theta)
    corrected = intensity - baseline

    return BaselineResult(
        two_theta=two_theta,
        raw_intensity=intensity,
        baseline=baseline,
        corrected=corrected,
        method="polynomial",
        parameters={"degree": degree, "baseline_indices": indices},
    )


def arpls_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    lam: float = 1e5,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> BaselineResult:
    """Asymmetric Reweighted Penalized Least Squares baseline, via the
    optional `pybaselines` dependency (`pybaselines.Baseline.arpls`) --
    GNOVI does not reimplement the algorithm (see PROJECT_GUIDE.md's
    Scientific Python Library Policy).

    Exposes only `lam` (the smoothness penalty -- the one parameter that
    actually matters for how aggressively the baseline follows the data;
    higher = smoother/less aggressive), plus `max_iter`/`tol` (convergence
    controls, defaulted to pybaselines' own defaults) -- not every
    `pybaselines.Baseline.arpls` parameter, matching the "minimal
    scientifically meaningful parameter surface" this milestone commits to.

    Raises `PybaselinesNotAvailableError` if `pybaselines` is not
    installed (see the `xrd` extra) -- never raised at GNOVI import time,
    only when this function is actually called. Raises
    `InvalidPreprocessingError` for a shape mismatch or non-finite input.

    Never mutates `two_theta`/`intensity`.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    _validate_xy(two_theta, intensity)

    if not _PYBASELINES_AVAILABLE:
        raise PybaselinesNotAvailableError(
            "arPLS baseline correction requires the optional 'pybaselines' "
            "package. Install it with: pip install gnovi-plot[xrd]"
        )

    fitter = _PybaselinesBaseline(two_theta)
    baseline, _params = fitter.arpls(intensity, lam=lam, max_iter=max_iter, tol=tol)
    baseline = np.asarray(baseline, dtype=float)
    corrected = intensity - baseline

    return BaselineResult(
        two_theta=two_theta,
        raw_intensity=intensity,
        baseline=baseline,
        corrected=corrected,
        method="arpls",
        parameters={"lam": lam, "max_iter": max_iter, "tol": tol},
    )


def savgol_smooth(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
) -> SmoothResult:
    """Savitzky-Golay smoothing (`scipy.signal.savgol_filter`) -- OFF by
    default conceptually: this function is never called automatically by
    anything else in `modules.xrd`, only when a caller explicitly asks for
    it.

    `window_length` MUST be odd and positive; `polyorder` MUST be less
    than `window_length`. Neither is silently corrected (e.g. an even
    window silently bumped to odd) -- both raise `InvalidPreprocessingError`
    with a clear message instead, so a caller's mistake is never hidden by
    a guessed correction.

    Smoothing changes peak width/shape -- the returned `SmoothResult`
    always carries `method="savgol"` and the exact parameters used, so
    nothing downstream can mistake `smoothed_intensity` for the raw
    experimental `raw_intensity` it's also carrying alongside it.

    Never mutates `two_theta`/`intensity`.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    _validate_xy(two_theta, intensity)

    if window_length <= 0 or window_length % 2 == 0:
        raise InvalidPreprocessingError(
            f"window_length must be a positive odd integer (got {window_length})"
        )
    if window_length > len(intensity):
        raise InvalidPreprocessingError(
            f"window_length ({window_length}) must not exceed the data length ({len(intensity)})"
        )
    if polyorder < 0 or polyorder >= window_length:
        raise InvalidPreprocessingError(
            f"polyorder must satisfy 0 <= polyorder < window_length (got "
            f"polyorder={polyorder}, window_length={window_length})"
        )

    smoothed = savgol_filter(intensity, window_length=window_length, polyorder=polyorder)

    return SmoothResult(
        two_theta=two_theta,
        raw_intensity=intensity,
        smoothed_intensity=smoothed,
        method="savgol",
        parameters={"window_length": window_length, "polyorder": polyorder},
    )
