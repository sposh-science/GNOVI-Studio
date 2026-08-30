"""Quantitative single-peak XRD profile fitting -- the numerical
foundation (XRD-3A).

Pure NumPy/SciPy code: no Qt, no Matplotlib, no `Dataset`. The public
entry point is `fit_xrd_peak`, which fits ONE symmetric profile
(Gaussian / Lorentzian / pseudo-Voigt) plus an optional local baseline
(none / constant / linear) inside an explicit 2theta window and returns an
`XRDPeakFitResult` (an `AnalysisResult` subclass, kind ``"xrd_peak_fit"``,
registered through the existing polymorphic mechanism).

Scientific conventions held here:

* **Area is the canonical amplitude.** Every profile is AREA-NORMALIZED:
  the fitted parameter ``area`` is exactly the analytical integrated
  intensity of the peak component above the local baseline. Peak
  ``height`` is a DERIVED quantity, computed from ``area`` and ``fwhm``
  (and ``eta`` for pseudo-Voigt). There is no independent height
  parameter, and no parameter called "amplitude".
* **FWHM is the fitted profile width**, in degrees 2theta
  (``fwhm_units = "degrees_2theta"``) -- never `scipy.signal.find_peaks`'
  detection width, and never silently converted to radians.
* **Local baseline is a fit term, not a background algorithm.** It is
  conceptually separate from `modules.xrd.preprocessing` (whole-pattern
  background/smoothing). The reported ``area`` never includes any
  baseline contribution.
* **Detection is not measurement.** A detected `XRDPeakSeed` only seeds
  the initial center/width; the returned center/FWHM/area come from the
  fit.
* **Covariance-derived standard errors are labelled as such** -- never
  "measurement uncertainty" / "confidence interval". A parameter's
  standard error is reported as ``None`` when `scipy.optimize.curve_fit`
  cannot estimate a usable covariance at all (then all are ``None``) or
  when that parameter sits at a fit bound. Extreme parameter correlation
  and low degrees of freedom add a caution but do NOT null otherwise
  finite standard errors. GNOVI does not attempt to detect peak overlap
  from the fit residuals -- overlap is flagged only from explicitly
  supplied neighbouring-peak positions.

Explicitly NOT implemented here (see PROJECT_GUIDE.md's XRD roadmap
notes): the researcher-facing peak-fitting workspace/GUI, multi-peak or
overlapping-peak deconvolution, Scherrer crystallite size, instrumental
broadening correction, Williamson-Hall, Ka1/Ka2 doublet modelling,
asymmetric profiles, Poisson weighting / reduced chi-square, phase
identification, Rietveld refinement, and QPA.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import curve_fit

from gnovi_plot.analysis.results import (
    ENGINE_GNOVI,
    AnalysisResult,
    ResidualData,
    register_result_kind,
)
from gnovi_plot.core.app_info import __version__ as _APP_VERSION
from gnovi_plot.modules.xrd.bragg import InvalidBraggInputError
from gnovi_plot.modules.xrd.bragg import d_spacing as _bragg_d_spacing
from gnovi_plot.modules.xrd.peaks import XRDPeakSeed
from gnovi_plot.modules.xrd.radiation import Radiation

# --- model / baseline identifiers -----------------------------------------

GAUSSIAN = "gaussian"
LORENTZIAN = "lorentzian"
PSEUDO_VOIGT = "pseudo_voigt"
PROFILE_MODELS: tuple[str, ...] = (GAUSSIAN, LORENTZIAN, PSEUDO_VOIGT)

BASELINE_NONE = "none"
BASELINE_CONSTANT = "constant"
BASELINE_LINEAR = "linear"
BASELINE_MODELS: tuple[str, ...] = (BASELINE_NONE, BASELINE_CONSTANT, BASELINE_LINEAR)

#: The x-axis unit the fitted FWHM (and center) are expressed in. XRD
#: patterns are conventionally 2theta in degrees; a later Scherrer
#: milestone performs whatever radian conversion it needs explicitly.
FWHM_UNITS_TWO_THETA_DEG = "degrees_2theta"

#: Stable, human-readable statement of the exact profile convention each
#: model uses -- recorded verbatim in `XRDPeakFitResult.parameters
#: ["profile_convention"]` so a researcher (or a downstream tool) is never
#: left guessing which of the several "pseudo-Voigt" parameterizations in
#: common use GNOVI means.
_PROFILE_CONVENTION: dict[str, str] = {
    GAUSSIAN: "area-normalized Gaussian; FWHM = Gamma (degrees 2theta); height derived from area and FWHM",
    LORENTZIAN: "area-normalized Lorentzian; FWHM = Gamma (degrees 2theta); height derived from area and FWHM",
    PSEUDO_VOIGT: (
        "area-normalized pseudo-Voigt = (1 - eta) * Gaussian + eta * Lorentzian, "
        "sharing one center and one FWHM (= Gamma, degrees 2theta); "
        "eta is the Lorentzian fraction: eta = 0 is a pure Gaussian, eta = 1 is a pure "
        "Lorentzian; height derived from area, FWHM and eta"
    ),
}

OPERATION_PEAK_FIT = "xrd_peak_fit"

#: Default number of samples in a regenerated fitted curve (mirrors
#: `analysis.fitting.DEFAULT_CURVE_SAMPLES`).
DEFAULT_CURVE_SAMPLES = 200

#: Default half-width of an auto-proposed fit window, in units of the
#: estimated FWHM: window = center +/- this * FWHM0 (approved XRD-3
#: strategy).
DEFAULT_WINDOW_FWHM_MULTIPLE = 4.0

# --- exact area-normalized profile shape constants -----------------------
#
# Gaussian:  G_N(x0) = 2*sqrt(ln2) / (Gamma*sqrt(pi))           (height per unit area)
# Lorentzian: L_N(x0) = 2 / (pi*Gamma)
_LN2 = math.log(2.0)
_FOUR_LN2 = 4.0 * _LN2
_GAUSS_HEIGHT_PER_AREA_GAMMA = 2.0 * math.sqrt(_LN2) / math.sqrt(math.pi)  # ~0.939437
_LORENTZ_HEIGHT_PER_AREA_GAMMA = 2.0 / math.pi  # ~0.636620


class XRDFitError(ValueError):
    """Raised when an XRD peak fit cannot be performed or trusted: an
    unknown model/baseline, a reversed/zero-width/empty fit window, too
    few finite points for the parameter count, data with no positive peak
    above the local baseline, or a solver that fails to converge. Callers
    should surface the message, not guess -- mirrors
    `analysis.fitting.FitError` and `modules.xrd.peaks.
    InvalidPeakDetectionError`.
    """


# --- normalized profiles (each integrates to 1, FWHM is exactly Gamma) ---


def gaussian_normalized(x, center: float, fwhm: float) -> np.ndarray:
    """Area-normalized Gaussian with FWHM ``fwhm``:
    ``2*sqrt(ln2)/(Gamma*sqrt(pi)) * exp(-4 ln2 * ((x-x0)/Gamma)^2)``.
    Integrates to 1 over (-inf, +inf)."""
    x = np.asarray(x, dtype=float)
    return (_GAUSS_HEIGHT_PER_AREA_GAMMA / fwhm) * np.exp(
        -_FOUR_LN2 * ((x - center) / fwhm) ** 2
    )


def lorentzian_normalized(x, center: float, fwhm: float) -> np.ndarray:
    """Area-normalized Lorentzian with FWHM ``fwhm``:
    ``(1/pi) * (Gamma/2) / ((x-x0)^2 + (Gamma/2)^2)``. Integrates to 1
    over (-inf, +inf)."""
    x = np.asarray(x, dtype=float)
    hwhm = fwhm / 2.0
    return (hwhm / math.pi) / ((x - center) ** 2 + hwhm**2)


def pseudo_voigt_normalized(x, center: float, fwhm: float, eta: float) -> np.ndarray:
    """Area-normalized pseudo-Voigt: ``(1-eta)*G_N + eta*L_N`` with a
    shared center and shared FWHM. Integrates to 1 for every ``eta`` in
    [0, 1]; its FWHM is exactly ``fwhm`` for every ``eta``."""
    return (1.0 - eta) * gaussian_normalized(x, center, fwhm) + eta * lorentzian_normalized(
        x, center, fwhm
    )


def peak_component(
    x, model: str, area: float, center: float, fwhm: float, eta: float | None = None
) -> np.ndarray:
    """The peak component ``area * <normalized profile>`` -- integrates to
    exactly ``area`` above a zero baseline."""
    if model == GAUSSIAN:
        return area * gaussian_normalized(x, center, fwhm)
    if model == LORENTZIAN:
        return area * lorentzian_normalized(x, center, fwhm)
    if model == PSEUDO_VOIGT:
        if eta is None:
            raise XRDFitError("pseudo_voigt peak_component requires eta")
        return area * pseudo_voigt_normalized(x, center, fwhm, eta)
    raise XRDFitError(f"Unknown profile model {model!r}; expected one of {PROFILE_MODELS}")


def derived_height(model: str, area: float, fwhm: float, eta: float | None = None) -> float:
    """Exact peak height (value of the peak component at its center) for a
    profile of integrated ``area`` and FWHM ``fwhm`` -- derived, never a
    fitted parameter.

    * Gaussian:   ``H = area * 2*sqrt(ln2)/(Gamma*sqrt(pi))``
    * Lorentzian: ``H = area * 2/(pi*Gamma)``
    * pseudo-Voigt: the eta-weighted combination of the two.
    """
    g = _GAUSS_HEIGHT_PER_AREA_GAMMA / fwhm
    lo = _LORENTZ_HEIGHT_PER_AREA_GAMMA / fwhm
    if model == GAUSSIAN:
        return float(area * g)
    if model == LORENTZIAN:
        return float(area * lo)
    if model == PSEUDO_VOIGT:
        if eta is None:
            raise XRDFitError("pseudo_voigt derived_height requires eta")
        return float(area * ((1.0 - eta) * g + eta * lo))
    raise XRDFitError(f"Unknown profile model {model!r}; expected one of {PROFILE_MODELS}")


def baseline_values(
    x, baseline_model: str, x_ref: float, c0: float = 0.0, c1: float = 0.0
) -> np.ndarray:
    """Local baseline ``B(x)``: 0 (``none``), ``c0`` (``constant``), or
    ``c0 + c1*(x - x_ref)`` (``linear``). ``x_ref`` is the fit-window
    midpoint -- a well-conditioned local origin so the slope term is
    numerically decoupled from the intercept."""
    x = np.asarray(x, dtype=float)
    if baseline_model == BASELINE_NONE:
        return np.zeros_like(x)
    if baseline_model == BASELINE_CONSTANT:
        return np.full_like(x, float(c0))
    if baseline_model == BASELINE_LINEAR:
        return float(c0) + float(c1) * (x - x_ref)
    raise XRDFitError(
        f"Unknown baseline model {baseline_model!r}; expected one of {BASELINE_MODELS}"
    )


# --- fit window ----------------------------------------------------------


@dataclass(frozen=True)
class FitWindow:
    """An explicit 2theta fit interval, end-inclusive."""

    two_theta_min: float
    two_theta_max: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.two_theta_min, self.two_theta_max)


def _local_step(two_theta: np.ndarray, around_index: int | None, radius: int = 64) -> float | None:
    """Median absolute step between consecutive x samples, near
    ``around_index`` when given. Uses ``abs(diff)`` so a descending 2theta
    array (unconventional, but valid input to the public
    `propose_fit_window`) yields the same step as its ascending
    equivalent, rather than every difference being filtered out as
    non-positive. ``None`` when no finite non-zero step exists."""
    tt = np.asarray(two_theta, dtype=float)
    if around_index is not None and 0 <= around_index < tt.size:
        lo = max(0, around_index - radius)
        hi = min(tt.size, around_index + radius + 1)
        seg = tt[lo:hi]
    else:
        seg = tt
    d = np.abs(np.diff(seg))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return None
    return float(np.median(d))


def estimate_seed_fwhm(two_theta, seed: XRDPeakSeed) -> float | None:
    """Convert a detected `XRDPeakSeed.width_samples` (array-index units,
    from `scipy.signal.find_peaks`) into an estimated FWHM in the x
    (2theta) units, using the local median x spacing near the seed.
    Returns ``None`` for a manual seed (no ``width_samples``) or if the x
    spacing cannot be determined."""
    if seed.width_samples is None or seed.width_samples <= 0:
        return None
    spacing = _local_step(np.asarray(two_theta, dtype=float), seed.index)
    if spacing is None:
        return None
    return float(seed.width_samples) * spacing


def propose_fit_window(
    two_theta,
    seed: XRDPeakSeed,
    *,
    neighbor_two_thetas: tuple[float, ...] | list[float] = (),
    width_multiple: float = DEFAULT_WINDOW_FWHM_MULTIPLE,
    fallback_fwhm: float | None = None,
) -> FitWindow:
    """Propose an initial fit window ``center +/- width_multiple *
    estimated_FWHM`` for ``seed``, then:

    * clip to the data's [min, max] 2theta range;
    * clip each side at the midpoint toward the nearest neighbouring
      detected peak in ``neighbor_two_thetas`` (so a window never spans
      two reflections);
    * fall back to ``fallback_fwhm`` (then to a data-scaled estimate) when
      the seed carries no usable width.

    Irregular x spacing is handled -- the sample->x conversion uses the
    local median spacing near the seed. Raises `XRDFitError` if the window
    collapses to zero width after clipping (the caller should then specify
    a window explicitly).

    Note: ``seed.index`` is interpreted against ``two_theta`` as passed
    here; supply the same pattern the seed was detected in.
    """
    tt = np.asarray(two_theta, dtype=float)
    finite = tt[np.isfinite(tt)]
    if finite.size < 2:
        raise XRDFitError("two_theta must contain at least two finite values to propose a window.")

    center = float(seed.two_theta)
    fwhm0 = estimate_seed_fwhm(tt, seed)
    if (fwhm0 is None or fwhm0 <= 0) and fallback_fwhm is not None and fallback_fwhm > 0:
        fwhm0 = float(fallback_fwhm)
    if fwhm0 is None or fwhm0 <= 0:
        span = float(finite.max() - finite.min())
        spacing = _local_step(tt, seed.index) or (span / max(finite.size - 1, 1))
        fwhm0 = max(span / 100.0, 5.0 * spacing)

    lo = center - width_multiple * fwhm0
    hi = center + width_multiple * fwhm0
    lo = max(lo, float(finite.min()))
    hi = min(hi, float(finite.max()))

    for raw in neighbor_two_thetas:
        n = float(raw)
        if not math.isfinite(n) or n == center:
            continue
        if n < center:
            lo = max(lo, 0.5 * (n + center))
        else:
            hi = min(hi, 0.5 * (center + n))

    if not (hi > lo):
        raise XRDFitError(
            "Proposed fit window collapsed to zero width after clipping; "
            "specify a fit window explicitly."
        )
    return FitWindow(two_theta_min=float(lo), two_theta_max=float(hi))


# --- parameter space ----------------------------------------------------


def _param_names(model: str, baseline_model: str) -> list[str]:
    names = ["area", "center", "fwhm"]
    if model == PSEUDO_VOIGT:
        names.append("eta")
    if baseline_model == BASELINE_CONSTANT:
        names.append("baseline_c0")
    elif baseline_model == BASELINE_LINEAR:
        names.extend(["baseline_c0", "baseline_c1"])
    return names


def _model_callable(model: str, baseline_model: str, x_ref: float, names: list[str]):
    def f(x, *theta):
        p = dict(zip(names, theta))
        peak = peak_component(x, model, p["area"], p["center"], p["fwhm"], p.get("eta"))
        base = baseline_values(
            x, baseline_model, x_ref, p.get("baseline_c0", 0.0), p.get("baseline_c1", 0.0)
        )
        return peak + base

    return f


@dataclass
class _ParamSpace:
    names: list[str]
    p0: list[float]
    lower: list[float]
    upper: list[float]


def _initial_space(
    x_w: np.ndarray,
    y_w: np.ndarray,
    model: str,
    baseline_model: str,
    x_ref: float,
    *,
    seed_center: float | None,
    seed_fwhm: float | None,
    overrides: dict[str, float] | None,
    bound_overrides: dict[str, tuple[float, float]] | None,
) -> tuple[_ParamSpace, list[str]]:
    """Deterministic initial guess + bounds for the windowed data.
    Returns the parameter space and a list of human-readable warnings."""
    warnings: list[str] = []
    names = _param_names(model, baseline_model)
    window_width = float(x_w[-1] - x_w[0])
    n = x_w.size

    # Provisional local baseline from the window "wings" (outer thirds),
    # which avoid the peak -- the classic foot-to-foot construction.
    wing = max(3, n // 6)
    wing_idx = np.r_[0:wing, n - wing : n]
    xw_wings, yw_wings = x_w[wing_idx], y_w[wing_idx]
    if baseline_model == BASELINE_LINEAR and np.unique(xw_wings).size >= 2:
        m, b = np.polyfit(xw_wings, yw_wings, 1)
        c1_0 = float(m)
        c0_0 = float(b + m * x_ref)
    elif baseline_model in (BASELINE_LINEAR, BASELINE_CONSTANT):
        c0_0 = float(np.median(yw_wings))
        c1_0 = 0.0
    else:
        c0_0 = 0.0
        c1_0 = 0.0
    base_guess = baseline_values(x_w, baseline_model, x_ref, c0_0, c1_0)
    y_sig = y_w - base_guess

    peak_height = float(np.max(y_sig))
    noise = 1.4826 * float(np.median(np.abs(np.diff(y_w) - np.median(np.diff(y_w))))) if n > 2 else 0.0
    if float(np.ptp(y_w)) <= 0.0 or peak_height <= 0.0:
        raise XRDFitError(
            "The data inside the fit window shows no positive peak above the local baseline "
            "(flat or non-positive signal). XRD peak fitting handles ordinary positive "
            "powder-XRD peaks only."
        )
    if noise > 0.0 and peak_height < 5.0 * noise:
        warnings.append(
            "Weak peak signal relative to local noise; fitted parameters and standard errors "
            "may be unreliable."
        )
    if baseline_model == BASELINE_NONE:
        # Warn when the data clearly do not return to ~zero within the
        # window while no local baseline was requested -- for a POSITIVE or
        # a NEGATIVE offset alike. Compares the window-edge ("wing") level
        # to the peak's height above that level, so an ordinary positive
        # peak sitting on ~zero never trips it.
        edge_level = float(np.median(yw_wings))
        peak_above_edge = float(np.max(y_w)) - edge_level
        if peak_above_edge > 0.0 and abs(edge_level) > 0.1 * peak_above_edge:
            warnings.append(
                "Baseline model is 'none' but the data do not return near zero at the window "
                "edges (edge level "
                f"{edge_level:.4g} vs peak height {peak_above_edge:.4g}); consider a constant or "
                "linear local baseline."
            )

    # center
    if overrides and "center" in overrides:
        center_0 = float(overrides["center"])
    elif seed_center is not None and x_w[0] <= seed_center <= x_w[-1]:
        center_0 = float(seed_center)
    else:
        center_0 = float(x_w[int(np.argmax(y_sig))])
        if seed_center is not None:
            warnings.append(
                "Seed center lies outside the fit window; initialized the center from the "
                "in-window maximum instead."
            )

    # fwhm
    local_spacing = _local_step(x_w, None) or (window_width / max(n - 1, 1))
    if overrides and "fwhm" in overrides:
        fwhm_0 = float(overrides["fwhm"])
    elif seed_fwhm is not None and 0.0 < seed_fwhm <= window_width:
        fwhm_0 = float(seed_fwhm)
    else:
        above = x_w[y_sig >= 0.5 * peak_height]
        fwhm_0 = float(above.max() - above.min()) if above.size >= 2 else window_width / 4.0
    fwhm_lo = max(2.0 * local_spacing, window_width * 1e-3)
    fwhm_hi = window_width
    fwhm_0 = float(min(max(fwhm_0, fwhm_lo * 1.5), fwhm_hi * 0.9))

    # area (positive); trapezoid of the positive signal, else height*width form
    trap = float(np.trapezoid(np.clip(y_sig, 0.0, None), x_w))
    area_0 = trap if trap > 0.0 else peak_height * fwhm_0 / _GAUSS_HEIGHT_PER_AREA_GAMMA
    if overrides and "area" in overrides:
        area_0 = float(overrides["area"])
    area_0 = max(area_0, 1e-12)

    guesses = {"area": area_0, "center": center_0, "fwhm": fwhm_0}
    lowers = {"area": 0.0, "center": float(x_w[0]), "fwhm": fwhm_lo}
    uppers = {"area": np.inf, "center": float(x_w[-1]), "fwhm": fwhm_hi}

    if model == PSEUDO_VOIGT:
        eta_0 = float(overrides["eta"]) if (overrides and "eta" in overrides) else 0.5
        guesses["eta"] = min(max(eta_0, 0.0), 1.0)
        lowers["eta"], uppers["eta"] = 0.0, 1.0

    if baseline_model in (BASELINE_CONSTANT, BASELINE_LINEAR):
        guesses["baseline_c0"] = float(overrides["baseline_c0"]) if (overrides and "baseline_c0" in overrides) else c0_0
        lowers["baseline_c0"], uppers["baseline_c0"] = -np.inf, np.inf
    if baseline_model == BASELINE_LINEAR:
        guesses["baseline_c1"] = float(overrides["baseline_c1"]) if (overrides and "baseline_c1" in overrides) else c1_0
        lowers["baseline_c1"], uppers["baseline_c1"] = -np.inf, np.inf

    if bound_overrides:
        for name, (blo, bhi) in bound_overrides.items():
            if name in lowers:
                lowers[name], uppers[name] = float(blo), float(bhi)

    # keep each guess strictly inside its bounds (curve_fit/trf requires it)
    for name in names:
        lo, hi = lowers[name], uppers[name]
        g = guesses[name]
        if math.isfinite(lo):
            g = max(g, lo + abs(lo) * 1e-9 + 1e-12)
        if math.isfinite(hi):
            g = min(g, hi - abs(hi) * 1e-9 - 1e-12)
        guesses[name] = g

    space = _ParamSpace(
        names=names,
        p0=[guesses[name] for name in names],
        lower=[lowers[name] for name in names],
        upper=[uppers[name] for name in names],
    )
    return space, warnings


# --- diagnostics helpers ----------------------------------------------


def _goodness_of_fit(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """``(r_squared, rss, rmse)`` -- one pass over the residuals, mirroring
    `analysis.fitting._fit_quality` (including its constant-y guard)."""
    residual = y - y_pred
    ss_res = float(np.sum(residual**2))
    n = len(y)
    rmse = float(np.sqrt(ss_res / n)) if n else 0.0
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        scale = float(np.sum(y**2)) or 1.0
        r_squared = 1.0 if ss_res <= 1e-9 * scale else 0.0
    else:
        r_squared = 1.0 - ss_res / ss_tot
    return r_squared, ss_res, rmse


def _standard_errors(pcov, names: list[str]) -> dict[str, float] | None:
    """``{name: sqrt(var)}`` or ``None`` when the covariance is unusable
    (non-finite, wrong shape, or a negative variance) -- the same
    "uncertainties not available" contract as
    `analysis.fitting._errors_from_covariance`."""
    if pcov is None:
        return None
    cov = np.asarray(pcov, dtype=float)
    if cov.shape != (len(names), len(names)) or not np.all(np.isfinite(cov)):
        return None
    variances = np.diag(cov)
    if np.any(variances < 0):
        return None
    return {name: float(np.sqrt(v)) for name, v in zip(names, variances)}


def _max_offdiagonal_correlation(pcov, n_params: int) -> float | None:
    """Largest absolute off-diagonal Pearson correlation
    ``R_ij = C_ij / sqrt(C_ii * C_jj)`` of the parameter covariance.

    Dimensionless, so -- unlike ``numpy.linalg.cond(pcov)`` -- it is not
    inflated by the parameters' very different natural scales (area,
    centre, FWHM, baseline slope). ``None`` when the covariance cannot be
    turned into a correlation matrix (non-finite, wrong shape, or a
    non-positive variance)."""
    if pcov is None:
        return None
    cov = np.asarray(pcov, dtype=float)
    if cov.shape != (n_params, n_params) or not np.all(np.isfinite(cov)):
        return None
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0.0):
        return None
    corr = cov / np.outer(sd, sd)
    off = corr[~np.eye(n_params, dtype=bool)]
    return float(np.max(np.abs(off))) if off.size else 0.0


def _at_bound(value: float, lo: float, hi: float, *, rtol: float = 1e-4, atol: float = 1e-9) -> bool:
    scale = max(
        abs(lo) if math.isfinite(lo) else 0.0,
        abs(hi) if math.isfinite(hi) else 0.0,
        abs(value),
        1.0,
    )
    tol = rtol * scale + atol
    if math.isfinite(lo) and abs(value - lo) <= tol:
        return True
    if math.isfinite(hi) and abs(value - hi) <= tol:
        return True
    return False


def _height_error(
    model: str, values: dict[str, float], pcov, names: list[str], unreliable: set[str]
) -> float | None:
    """Standard error of the DERIVED height by first-order propagation
    through the (area, fwhm[, eta]) sub-covariance -- cross-terms
    included (never independent-error quadrature). ``None`` when the
    covariance is unusable or a parameter it depends on is at a bound."""
    if pcov is None:
        return None
    cov = np.asarray(pcov, dtype=float)
    if cov.shape != (len(names), len(names)) or not np.all(np.isfinite(cov)):
        return None
    area = values["area"]
    gamma = values["fwhm"]
    kg = _GAUSS_HEIGHT_PER_AREA_GAMMA
    kl = _LORENTZ_HEIGHT_PER_AREA_GAMMA
    if model == GAUSSIAN:
        dep = ["area", "fwhm"]
        grad = np.array([kg / gamma, -area * kg / gamma**2])
    elif model == LORENTZIAN:
        dep = ["area", "fwhm"]
        grad = np.array([kl / gamma, -area * kl / gamma**2])
    else:
        eta = values["eta"]
        dep = ["area", "fwhm", "eta"]
        mix = (1.0 - eta) * kg + eta * kl
        grad = np.array(
            [mix / gamma, -area * mix / gamma**2, area * (kl - kg) / gamma]
        )
    if any(d in unreliable for d in dep):
        return None
    idx = [names.index(d) for d in dep]
    sub = cov[np.ix_(idx, idx)]
    var = float(grad @ sub @ grad)
    if not math.isfinite(var) or var < 0.0:
        return None
    return math.sqrt(var)


def _neighbour_overlap_warnings(
    center: float,
    fwhm: float,
    window: tuple[float, float],
    neighbor_two_thetas: tuple[float, ...] | list[float],
) -> list[str]:
    """A single conservative, hedged warning when an explicitly supplied
    neighbouring detected-peak position lies inside the fitted window.

    A neighbour within ~0.2*FWHM of the fitted centre is treated as the
    same reflection (e.g. the seed being fitted, if a caller passes it
    through in ``neighbor_two_thetas``); anything else inside the window
    is a distinct peak.

    GNOVI deliberately does NOT try to infer overlap from the fit
    residuals: a residual-scale heuristic collapses toward floating-point
    roundoff on a near-perfect fit and would warn about a correct isolated
    single-peak fit. Overlap is only ever flagged from explicit
    neighbouring-peak information, and never claimed as proven.
    """
    lo, hi = window
    inside = [
        float(n)
        for n in neighbor_two_thetas
        if math.isfinite(float(n)) and lo < float(n) < hi and abs(float(n) - center) > 0.2 * fwhm
    ]
    if not inside:
        return []
    return [
        "Another detected peak lies inside the fit window; single-peak fit parameters "
        "may not represent an isolated reflection."
    ]


def _bragg_d_and_error(
    radiation: Radiation | None, center: float, center_error: float | None
) -> tuple[float | None, float | None, list[str]]:
    """d-spacing from the fitted centre plus the FIT standard error
    propagated analytically through Bragg's law. ``(None, None, [])`` when
    no radiation context was supplied (never assumes a wavelength)."""
    if radiation is None:
        return None, None, []
    try:
        d = float(_bragg_d_spacing(center, radiation.wavelength_angstrom))
    except InvalidBraggInputError:
        return None, None, [
            "Fitted centre 2theta is outside the valid Bragg range; d-spacing was not computed."
        ]
    if center_error is None:
        return d, None, []
    theta_rad = math.radians(center / 2.0)
    # d = lambda / (2 sin(theta)),  theta = center_deg * pi / 360
    # dd/d(center_deg) = -d * cot(theta) * (pi / 360)
    dd_dcenter = -d / math.tan(theta_rad) * (math.pi / 360.0)
    return d, abs(dd_dcenter) * center_error, []


# --- result -------------------------------------------------------------


@register_result_kind
@dataclass
class XRDPeakFitResult(AnalysisResult):
    """One fitted single XRD peak profile: an area-normalized Gaussian /
    Lorentzian / pseudo-Voigt plus a local baseline, fitted in an explicit
    2theta window.

    ``area`` is the canonical fitted amplitude -- the full analytical,
    infinite-domain integrated intensity of the peak component above the
    fitted local baseline. Because the profiles are area-normalized, a
    finite fit window does not directly contain all of that area: the
    fitted model constrains the wings beyond the window through the
    profile shape. For a Gaussian the difference is negligible; for a
    Lorentzian (and a Lorentzian-rich pseudo-Voigt) it is not -- e.g. a
    +/- 4*FWHM window around a pure Lorentzian encloses only ~92% of the
    reported ``area``. A reported ``area`` is therefore sensitive to the
    chosen profile ``model``, the ``fit_window``, and the
    ``baseline_model``; quantitative comparison of integrated areas should
    use a consistent fitting procedure and profile convention where
    possible, and the model choice should be reported alongside the value.

    ``height`` is DERIVED (see `derived_height`). ``fwhm`` is the fitted
    profile width in ``fwhm_units`` (degrees 2theta). Standard errors are
    covariance-derived (``None`` where not reliably estimable -- see
    ``warnings``); they are fit standard errors, not experimental
    measurement uncertainties or confidence intervals.

    For pseudo-Voigt, ``eta`` is the Lorentzian fraction of an
    area-normalized ``(1 - eta) * Gaussian + eta * Lorentzian`` sharing one
    center and one FWHM: ``eta = 0`` is a pure Gaussian, ``eta = 1`` a pure
    Lorentzian. The exact convention is recorded verbatim in
    ``parameters["profile_convention"]``.

    Dense fitted-curve arrays are never stored -- the curve is regenerated
    from ``model`` / ``params`` / ``baseline_model`` / the window via
    `evaluate_total` / `sample_fit_curve`.

    Not a multi-peak result: a future overlapping-peak deconvolution gets
    its own result kind rather than a ``components`` list here.
    """

    kind: ClassVar[str] = "xrd_peak_fit"

    model: str
    baseline_model: str
    fit_window: tuple[float, float]
    window_x_ref: float
    n_points: int
    n_params: int
    dof: int
    params: dict[str, float]
    param_errors: dict[str, float | None] | None
    area: float
    area_error: float | None
    center_2theta: float
    center_error: float | None
    fwhm: float
    fwhm_error: float | None
    fwhm_units: str
    height: float
    height_error: float | None
    rss: float
    rmse: float
    r_squared: float
    converged: bool
    solver_message: str
    warnings: list[str]
    radiation: Radiation | None = None
    eta: float | None = None
    eta_error: float | None = None
    d_spacing: float | None = None
    d_spacing_error: float | None = None
    source_peak_id: str | None = None
    source_result_id: str | None = None
    curve_x_min: float | None = None
    curve_x_max: float | None = None
    curve_num_points: int | None = None

    # --- display contract ------------------------------------------------

    def summary(self) -> str:
        def pm(value: float, err: float | None, spec: str = ".5g") -> str:
            return f"{value:{spec}} ± {err:.2g}" if err is not None else f"{value:{spec}}"

        model_label = self.model.replace("_", "-")
        parts = [
            f"{model_label} peak fit: 2θ = {pm(self.center_2theta, self.center_error)}°",
            f"FWHM = {pm(self.fwhm, self.fwhm_error)}°",
            f"area = {pm(self.area, self.area_error, '.4g')}",
            f"R² = {self.r_squared:.4f}",
        ]
        return ", ".join(parts)

    def details(self) -> list[tuple[str, str]]:
        """A BOUNDED, fixed set of rows (never one row per anything) -- the
        same layout discipline as `XRDAnalysisResult.details`."""

        def pm(value: float | None, err: float | None, spec: str = ".6g") -> str:
            if value is None:
                return "—"
            return f"{value:{spec}} ± {err:.3g}" if err is not None else f"{value:{spec}}"

        errs = self.param_errors or {}
        rows: list[tuple[str, str]] = [
            ("Model", self.model.replace("_", "-")),
            ("Baseline", self.baseline_model),
            ("Fit window (°2θ)", f"{self.fit_window[0]:.4f} – {self.fit_window[1]:.4f}"),
            ("Points fitted", str(self.n_points)),
            ("Center (°2θ)", pm(self.center_2theta, self.center_error)),
            ("FWHM (°2θ)", pm(self.fwhm, self.fwhm_error)),
            ("Area (integrated intensity)", pm(self.area, self.area_error)),
            ("Height (derived)", pm(self.height, self.height_error)),
        ]
        if self.model == PSEUDO_VOIGT:
            rows.append(
                ("η (Lorentzian fraction; 0=Gaussian, 1=Lorentzian)", pm(self.eta, self.eta_error, ".4f"))
            )
        if self.baseline_model in (BASELINE_CONSTANT, BASELINE_LINEAR):
            rows.append(("Baseline c₀", pm(self.params.get("baseline_c0"), errs.get("baseline_c0"))))
        if self.baseline_model == BASELINE_LINEAR:
            rows.append(("Baseline c₁ (per °2θ)", pm(self.params.get("baseline_c1"), errs.get("baseline_c1"))))
        if self.d_spacing is not None:
            rows.append(
                ("d-spacing (Å)", pm(self.d_spacing, self.d_spacing_error) + " (propagated fit s.e.)")
            )
        else:
            rows.append(("d-spacing (Å)", "— (no radiation context)"))
        rows.extend(
            [
                ("R²", f"{self.r_squared:.6f}"),
                ("RMSE", f"{self.rmse:.6g}"),
                ("RSS", f"{self.rss:.6g}"),
                ("Degrees of freedom", str(self.dof)),
                ("Converged", "Yes" if self.converged else "No"),
            ]
        )
        if self.warnings:
            rows.append((f"Warnings ({len(self.warnings)})", self.warnings[0]))
        return rows

    def supports_residuals(self) -> bool:
        return True

    def compute_residuals(self, x, y) -> ResidualData:
        return compute_residuals(self, x, y)

    def residual_window_subtitle(self) -> str:
        return f"{self.model.replace('_', '-')} peak fit"

    # --- serialization -------------------------------------------------

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "model": self.model,
                "baseline_model": self.baseline_model,
                "fit_window": list(self.fit_window),
                "window_x_ref": self.window_x_ref,
                "n_points": self.n_points,
                "n_params": self.n_params,
                "dof": self.dof,
                "params": {k: float(v) for k, v in self.params.items()},
                "param_errors": (
                    None
                    if self.param_errors is None
                    else {k: (None if v is None else float(v)) for k, v in self.param_errors.items()}
                ),
                "area": self.area,
                "area_error": self.area_error,
                "center_2theta": self.center_2theta,
                "center_error": self.center_error,
                "fwhm": self.fwhm,
                "fwhm_error": self.fwhm_error,
                "fwhm_units": self.fwhm_units,
                "height": self.height,
                "height_error": self.height_error,
                "rss": self.rss,
                "rmse": self.rmse,
                "r_squared": self.r_squared,
                "converged": self.converged,
                "solver_message": self.solver_message,
                "warnings": list(self.warnings),
                "radiation": self.radiation.to_dict() if self.radiation is not None else None,
                "eta": self.eta,
                "eta_error": self.eta_error,
                "d_spacing": self.d_spacing,
                "d_spacing_error": self.d_spacing_error,
                "source_peak_id": self.source_peak_id,
                "source_result_id": self.source_result_id,
                "curve_x_min": self.curve_x_min,
                "curve_x_max": self.curve_x_max,
                "curve_num_points": self.curve_num_points,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "XRDPeakFitResult":
        row_range = data.get("row_range")
        window = data.get("fit_window", [0.0, 0.0])
        radiation = data.get("radiation")
        param_errors = data.get("param_errors")
        return cls(
            source_dataset_id=data["source_dataset_id"],
            source_dataset_name=data.get("source_dataset_name"),
            source_series_id=data.get("source_series_id"),
            source_series_label=data.get("source_series_label"),
            x_column=data["x_column"],
            y_column=data["y_column"],
            row_range=tuple(row_range) if row_range is not None else None,
            source_panel_id=data.get("source_panel_id"),
            result_id=data.get("result_id") or uuid.uuid4().hex,
            engine=data.get("engine", ENGINE_GNOVI),
            engine_version=data.get("engine_version"),
            operation=data.get("operation", OPERATION_PEAK_FIT),
            parameters=dict(data.get("parameters", {})),
            model=data["model"],
            baseline_model=data.get("baseline_model", BASELINE_NONE),
            fit_window=(float(window[0]), float(window[1])),
            window_x_ref=float(data.get("window_x_ref", 0.5 * (float(window[0]) + float(window[1])))),
            n_points=int(data.get("n_points", 0)),
            n_params=int(data.get("n_params", 0)),
            dof=int(data.get("dof", 0)),
            params={k: float(v) for k, v in data.get("params", {}).items()},
            param_errors=(
                None
                if param_errors is None
                else {k: (None if v is None else float(v)) for k, v in param_errors.items()}
            ),
            area=float(data["area"]),
            area_error=data.get("area_error"),
            center_2theta=float(data["center_2theta"]),
            center_error=data.get("center_error"),
            fwhm=float(data["fwhm"]),
            fwhm_error=data.get("fwhm_error"),
            fwhm_units=data.get("fwhm_units", FWHM_UNITS_TWO_THETA_DEG),
            height=float(data["height"]),
            height_error=data.get("height_error"),
            rss=float(data.get("rss", 0.0)),
            rmse=float(data.get("rmse", 0.0)),
            r_squared=float(data.get("r_squared", 0.0)),
            converged=bool(data.get("converged", False)),
            solver_message=data.get("solver_message", ""),
            warnings=list(data.get("warnings", [])),
            radiation=Radiation.from_dict(radiation) if radiation is not None else None,
            eta=data.get("eta"),
            eta_error=data.get("eta_error"),
            d_spacing=data.get("d_spacing"),
            d_spacing_error=data.get("d_spacing_error"),
            source_peak_id=data.get("source_peak_id"),
            source_result_id=data.get("source_result_id"),
            curve_x_min=data.get("curve_x_min"),
            curve_x_max=data.get("curve_x_max"),
            curve_num_points=data.get("curve_num_points"),
        )


# --- curve regeneration ----------------------------------------------


def evaluate_baseline(result: XRDPeakFitResult, x) -> np.ndarray:
    """The fitted local baseline B(x) alone."""
    p = result.params
    return baseline_values(
        x, result.baseline_model, result.window_x_ref, p.get("baseline_c0", 0.0), p.get("baseline_c1", 0.0)
    )


def evaluate_peak_component(result: XRDPeakFitResult, x) -> np.ndarray:
    """The fitted peak component alone (baseline removed) -- integrates to
    ``result.area``."""
    p = result.params
    return peak_component(x, result.model, p["area"], p["center"], p["fwhm"], p.get("eta"))


def evaluate_total(result: XRDPeakFitResult, x) -> np.ndarray:
    """The full fitted signal: peak component + local baseline."""
    return evaluate_peak_component(result, x) + evaluate_baseline(result, x)


def sample_fit_curve(
    result: XRDPeakFitResult,
    x_min: float | None = None,
    x_max: float | None = None,
    num_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A smooth ``(x, y_total)`` curve for the fitted model, defaulting to
    the stored fit-window range and sample count. Deterministic: fully
    reproducible from the result's parameters."""
    x_min = result.curve_x_min if x_min is None else x_min
    x_max = result.curve_x_max if x_max is None else x_max
    if x_min is None or x_max is None:
        x_min, x_max = result.fit_window
    num_points = result.curve_num_points if num_points is None else num_points
    num_points = num_points or DEFAULT_CURVE_SAMPLES
    if num_points < 2:
        raise XRDFitError(f"num_points must be at least 2 (got {num_points})")
    x = np.linspace(float(x_min), float(x_max), int(num_points))
    return x, evaluate_total(result, x)


def compute_residuals(result: XRDPeakFitResult, x, y) -> ResidualData:
    """``observed - fitted`` at ``(x, y)`` using the fitted total model --
    the same evaluation `sample_fit_curve` uses. Pure; the caller resolves
    ``x``/``y`` (e.g. from the source series' current data)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fitted = evaluate_total(result, x_arr)
    return ResidualData(x=x_arr, observed=y_arr, fitted=fitted, residuals=y_arr - fitted)


# --- the fit ---------------------------------------------------------


def _resolve_window(fit_window) -> tuple[float, float]:
    if isinstance(fit_window, FitWindow):
        lo, hi = fit_window.two_theta_min, fit_window.two_theta_max
    else:
        lo, hi = float(fit_window[0]), float(fit_window[1])
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise XRDFitError(f"fit_window bounds must be finite (got {lo!r}, {hi!r})")
    if not (hi > lo):
        raise XRDFitError(
            f"fit_window must have max > min (got min={lo}, max={hi}); reversed or zero-width."
        )
    return lo, hi


def fit_xrd_peak(
    two_theta,
    intensity,
    model: str,
    *,
    fit_window,
    baseline: str = BASELINE_LINEAR,
    radiation: Radiation | None = None,
    source_dataset_id: str,
    x_column: str,
    y_column: str,
    source_dataset_name: str | None = None,
    source_series_id: str | None = None,
    source_series_label: str | None = None,
    row_range: tuple[int, int] | None = None,
    source_panel_id: str | None = None,
    seed: XRDPeakSeed | None = None,
    source_peak_id: str | None = None,
    source_result_id: str | None = None,
    initial_params: dict[str, float] | None = None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    neighbor_two_thetas: tuple[float, ...] | list[float] = (),
    curve_num_points: int = DEFAULT_CURVE_SAMPLES,
) -> XRDPeakFitResult:
    """Fit one ``model`` profile (+ local ``baseline``) to ``(two_theta,
    intensity)`` inside ``fit_window`` and return an `XRDPeakFitResult`.

    ``two_theta`` / ``intensity`` are plain array-likes (a pandas Series is
    fine). All ``source_*`` / ``row_range`` / ``x_column`` / ``y_column``
    values are opaque provenance threaded straight into the result -- this
    function never imports `Dataset`, Qt, or anything from
    `gnovi_plot.plotting`.

    ``seed`` (optional) seeds the initial centre and width; ``fit_window``
    is always explicit (use `propose_fit_window` to derive one).
    ``initial_params`` / ``param_bounds`` override the deterministic
    defaults per key (``area`` / ``center`` / ``fwhm`` / ``eta`` /
    ``baseline_c0`` / ``baseline_c1``). ``neighbor_two_thetas`` feeds the
    conservative overlap warning only.

    Raises `XRDFitError` for an unknown model/baseline, a
    reversed/zero-width/empty window, fewer than ``max(2*P, 10)`` finite
    points, data with no positive peak above the local baseline, or a
    solver that fails to converge.
    """
    if model not in PROFILE_MODELS:
        raise XRDFitError(f"Unknown profile model {model!r}; expected one of {PROFILE_MODELS}")
    if baseline not in BASELINE_MODELS:
        raise XRDFitError(f"Unknown baseline model {baseline!r}; expected one of {BASELINE_MODELS}")

    w_min, w_max = _resolve_window(fit_window)
    x_ref = 0.5 * (w_min + w_max)

    x_all = np.asarray(two_theta, dtype=float)
    y_all = np.asarray(intensity, dtype=float)
    if x_all.shape != y_all.shape:
        raise XRDFitError(
            f"two_theta and intensity must have the same shape (got {x_all.shape} and {y_all.shape})"
        )
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    n_dropped = int(finite.size - int(np.count_nonzero(finite)))
    x_all, y_all = x_all[finite], y_all[finite]
    order = np.argsort(x_all, kind="stable")
    x_all, y_all = x_all[order], y_all[order]
    if x_all.size < 2:
        raise XRDFitError("Fewer than two finite (two_theta, intensity) pairs after cleaning.")

    in_window = (x_all >= w_min) & (x_all <= w_max)
    x_w, y_w = x_all[in_window], y_all[in_window]
    if x_w.size == 0:
        raise XRDFitError(
            f"The fit window [{w_min:g}, {w_max:g}] contains no finite data points."
        )

    names = _param_names(model, baseline)
    n_params = len(names)
    min_points = max(2 * n_params, 10)
    if x_w.size < min_points:
        raise XRDFitError(
            f"Not enough finite points in the fit window to fit a {model} + {baseline}-baseline "
            f"model (found {x_w.size}, need at least {min_points} = max(2*P, 10) for P={n_params} "
            "free parameters). Widen the window."
        )

    warnings: list[str] = []
    if n_dropped:
        warnings.append(
            f"{n_dropped} non-finite (two_theta, intensity) pair(s) were excluded before fitting."
        )
    data_lo, data_hi = float(x_all.min()), float(x_all.max())
    edge_spacing = _local_step(x_all, None) or 0.0
    if w_min < data_lo - edge_spacing or w_max > data_hi + edge_spacing:
        warnings.append(
            "Fit window extends beyond the data range; the effective window is the data's "
            f"[{data_lo:g}, {data_hi:g}] overlap."
        )
    x_at_max = float(x_w[int(np.argmax(y_w))])
    if (x_at_max - data_lo) <= edge_spacing or (data_hi - x_at_max) <= edge_spacing:
        warnings.append(
            "The in-window maximum sits at a data boundary; centre and FWHM may be biased by an "
            "asymmetric window."
        )

    seed_center = None if seed is None else float(seed.two_theta)
    seed_fwhm = None if seed is None else estimate_seed_fwhm(x_all, seed)

    space, init_warnings = _initial_space(
        x_w,
        y_w,
        model,
        baseline,
        x_ref,
        seed_center=seed_center,
        seed_fwhm=seed_fwhm,
        overrides=initial_params,
        bound_overrides=param_bounds,
    )
    warnings.extend(init_warnings)

    dof = x_w.size - n_params
    if dof <= 0:
        raise XRDFitError(
            f"Non-positive degrees of freedom (points={x_w.size}, parameters={n_params})."
        )

    f = _model_callable(model, baseline, x_ref, names)
    try:
        popt, pcov, _infodict, errmsg, ier = curve_fit(
            f,
            x_w,
            y_w,
            p0=space.p0,
            bounds=(space.lower, space.upper),
            max_nfev=20000,
            x_scale="jac",
            full_output=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise XRDFitError(f"XRD {model} peak fit did not converge: {exc}") from exc

    converged = ier in (1, 2, 3, 4)
    solver_message = (errmsg or "").strip() or ("converged" if converged else "did not converge")
    if not converged:
        raise XRDFitError(f"XRD {model} peak fit did not converge: {solver_message}")

    values = {name: float(v) for name, v in zip(names, popt)}
    y_pred = f(x_w, *popt)
    r_squared, rss, rmse = _goodness_of_fit(y_w, y_pred)

    # --- standard errors, with explicit reliability handling ---
    raw_errors = _standard_errors(pcov, names)
    cov_ok = raw_errors is not None
    unreliable: set[str] = set()
    if not cov_ok:
        warnings.append(
            "Fit covariance could not be estimated (singular or non-finite); parameter standard "
            "errors are unavailable."
        )
    else:
        # A dimensionless parameter-correlation check -- NOT `cond(pcov)`,
        # which is dominated by the parameters' very different natural
        # scales (area ~1e3, centre ~30, FWHM ~0.1, baseline slope ...) and
        # so reads "ill-conditioned" even for a perfectly determined fit.
        max_corr = _max_offdiagonal_correlation(pcov, len(names))
        if max_corr is not None and max_corr > 0.999:
            warnings.append(
                "Fit parameters are extremely correlated (|r| > 0.999); covariance-derived "
                "standard errors should be interpreted cautiously."
            )
        for name in names:
            lo = space.lower[names.index(name)]
            hi = space.upper[names.index(name)]
            if _at_bound(values[name], lo, hi):
                unreliable.add(name)
        if "eta" in unreliable and model == PSEUDO_VOIGT:
            endpoint = (
                "the Gaussian endpoint (η≈0)"
                if values["eta"] < 0.5
                else "the Lorentzian endpoint (η≈1)"
            )
            warnings.append(
                f"η converged to {endpoint}; its covariance-derived standard error is not "
                "reported. The other fitted parameters are unaffected."
            )
        other_at_bound = sorted(unreliable - {"eta"})
        if other_at_bound:
            warnings.append(
                "Parameter(s) at a fit bound (" + ", ".join(other_at_bound) + "); their "
                "covariance-derived standard errors are not reported."
            )
    if dof < 10:
        warnings.append(
            f"Few degrees of freedom (dof={dof}); the fit is numerically permissible but "
            "standard errors are indicative only."
        )

    if raw_errors is None:
        param_errors: dict[str, float | None] | None = None
    else:
        param_errors = {
            name: (None if name in unreliable else raw_errors[name]) for name in names
        }

    def _err(name: str) -> float | None:
        if param_errors is None:
            return None
        return param_errors.get(name)

    area = values["area"]
    area_error = _err("area")
    center = values["center"]
    center_error = _err("center")
    fwhm = values["fwhm"]
    fwhm_error = _err("fwhm")
    eta = values.get("eta") if model == PSEUDO_VOIGT else None
    eta_error = _err("eta") if model == PSEUDO_VOIGT else None

    height = derived_height(model, area, fwhm, eta)
    height_error = _height_error(model, values, pcov if cov_ok else None, names, unreliable)
    if cov_ok and height_error is None and not unreliable:
        warnings.append(
            "Derived height standard error could not be propagated from the fit covariance."
        )

    d_val, d_err, d_warnings = _bragg_d_and_error(radiation, center, center_error)
    warnings.extend(d_warnings)

    warnings.extend(
        _neighbour_overlap_warnings(center, fwhm, (w_min, w_max), neighbor_two_thetas)
    )

    resolved_peak_id = source_peak_id if source_peak_id is not None else (seed.id if seed is not None else None)

    parameters = {
        "model": model,
        "baseline_model": baseline,
        "fit_window": [w_min, w_max],
        "window_x_ref": x_ref,
        "fwhm_units": FWHM_UNITS_TWO_THETA_DEG,
        "amplitude_convention": (
            "area_normalized: fitted 'area' is the integrated intensity of the peak component "
            "above the local baseline; height is derived"
        ),
        "profile_convention": _PROFILE_CONVENTION[model],
        # Solver identity only. The SciPy-internal termination message and
        # function-evaluation count are deliberately NOT stored in this
        # reproducibility dict -- they are SciPy-version/platform-dependent
        # noise that would make two otherwise-identical fits compare unequal
        # after a save/reload. The at-a-glance `solver_message` /
        # `converged` fields on the result itself carry that diagnostic.
        "solver": "scipy.optimize.curve_fit (trf)",
        "initial_params": {name: float(v) for name, v in zip(space.names, space.p0)},
        "param_bounds": {
            name: [
                (None if not math.isfinite(space.lower[i]) else float(space.lower[i])),
                (None if not math.isfinite(space.upper[i]) else float(space.upper[i])),
            ]
            for i, name in enumerate(space.names)
        },
        "source_peak_id": resolved_peak_id,
        "source_result_id": source_result_id,
        "radiation": radiation.to_dict() if radiation is not None else None,
    }

    return XRDPeakFitResult(
        source_dataset_id=source_dataset_id,
        source_dataset_name=source_dataset_name,
        source_series_id=source_series_id,
        source_series_label=source_series_label,
        x_column=x_column,
        y_column=y_column,
        row_range=row_range,
        source_panel_id=source_panel_id,
        result_id=uuid.uuid4().hex,
        engine=ENGINE_GNOVI,
        engine_version=_APP_VERSION,
        operation=OPERATION_PEAK_FIT,
        parameters=parameters,
        model=model,
        baseline_model=baseline,
        fit_window=(w_min, w_max),
        window_x_ref=x_ref,
        n_points=int(x_w.size),
        n_params=n_params,
        dof=int(dof),
        params=values,
        param_errors=param_errors,
        area=float(area),
        area_error=area_error,
        center_2theta=float(center),
        center_error=center_error,
        fwhm=float(fwhm),
        fwhm_error=fwhm_error,
        fwhm_units=FWHM_UNITS_TWO_THETA_DEG,
        height=float(height),
        height_error=height_error,
        rss=float(rss),
        rmse=float(rmse),
        r_squared=float(r_squared),
        converged=bool(converged),
        solver_message=solver_message,
        warnings=warnings,
        radiation=radiation,
        eta=None if eta is None else float(eta),
        eta_error=eta_error,
        d_spacing=d_val,
        d_spacing_error=d_err,
        source_peak_id=resolved_peak_id,
        source_result_id=source_result_id,
        curve_x_min=w_min,
        curve_x_max=w_max,
        curve_num_points=int(curve_num_points),
    )
