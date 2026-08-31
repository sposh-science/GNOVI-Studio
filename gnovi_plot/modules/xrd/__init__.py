"""GNOVI's native XRD numerical foundation: radiation/wavelength handling,
Bragg's-law d-spacing, background/smoothing preprocessing, peak detection,
and single-peak profile fitting (Gaussian/Lorentzian/pseudo-Voigt) -- pure
NumPy/SciPy(+optional pybaselines) numerical code, no Qt, no Matplotlib,
no XRD GUI.

Explicitly NOT part of this foundation (see PROJECT_GUIDE.md's XRD roadmap
notes): the researcher-facing peak-fitting workspace/GUI, multi-peak or
overlapping-peak deconvolution, Scherrer crystallite-size calculation,
instrumental broadening correction, Williamson-Hall analysis, phase
identification, Rietveld refinement, quantitative phase analysis, or any
external-engine integration (GSAS-II/pyFAI/Profex/BGMN).

This is a thin re-export of the module's public API; the actual
implementation lives in the individual submodules below.
"""

from __future__ import annotations

from gnovi_plot.modules.xrd.bragg import InvalidBraggInputError, d_spacing
from gnovi_plot.modules.xrd.fitting import (
    BASELINE_CONSTANT,
    BASELINE_LINEAR,
    BASELINE_MODELS,
    BASELINE_NONE,
    FWHM_UNITS_TWO_THETA_DEG,
    GAUSSIAN,
    LORENTZIAN,
    OPERATION_PEAK_FIT,
    PROFILE_MODELS,
    PSEUDO_VOIGT,
    FitWindow,
    LocalWidthEstimate,
    XRDFitError,
    XRDPeakFitResult,
    derived_height,
    estimate_local_peak_width,
    estimate_seed_fwhm,
    evaluate_baseline,
    evaluate_peak_component,
    evaluate_total,
    fit_xrd_peak,
    gaussian_normalized,
    lorentzian_normalized,
    peak_component,
    propose_fit_window,
    pseudo_voigt_normalized,
    sample_fit_curve,
)
from gnovi_plot.modules.xrd.peaks import (
    ORIGIN_AUTOMATIC,
    ORIGIN_MANUAL,
    InvalidPeakDetectionError,
    XRDPeakSeed,
    detect_peaks,
)
from gnovi_plot.modules.xrd.preprocessing import (
    BaselineResult,
    InvalidPreprocessingError,
    PybaselinesNotAvailableError,
    SmoothResult,
    arpls_baseline,
    polynomial_baseline,
    savgol_smooth,
)
from gnovi_plot.modules.xrd.radiation import (
    RADIATION_PRESETS,
    InvalidRadiationError,
    Radiation,
    radiation_from_preset,
)
from gnovi_plot.modules.xrd.results import (
    OPERATION_PEAK_DETECTION,
    XRDAnalysisResult,
    build_xrd_analysis_result,
)

__all__ = [
    "InvalidBraggInputError",
    "d_spacing",
    "ORIGIN_AUTOMATIC",
    "ORIGIN_MANUAL",
    "InvalidPeakDetectionError",
    "XRDPeakSeed",
    "detect_peaks",
    "BaselineResult",
    "InvalidPreprocessingError",
    "PybaselinesNotAvailableError",
    "SmoothResult",
    "arpls_baseline",
    "polynomial_baseline",
    "savgol_smooth",
    "RADIATION_PRESETS",
    "InvalidRadiationError",
    "Radiation",
    "radiation_from_preset",
    "OPERATION_PEAK_DETECTION",
    "XRDAnalysisResult",
    "build_xrd_analysis_result",
    "GAUSSIAN",
    "LORENTZIAN",
    "PSEUDO_VOIGT",
    "PROFILE_MODELS",
    "BASELINE_NONE",
    "BASELINE_CONSTANT",
    "BASELINE_LINEAR",
    "BASELINE_MODELS",
    "FWHM_UNITS_TWO_THETA_DEG",
    "OPERATION_PEAK_FIT",
    "FitWindow",
    "LocalWidthEstimate",
    "XRDFitError",
    "XRDPeakFitResult",
    "fit_xrd_peak",
    "propose_fit_window",
    "estimate_local_peak_width",
    "estimate_seed_fwhm",
    "derived_height",
    "gaussian_normalized",
    "lorentzian_normalized",
    "pseudo_voigt_normalized",
    "peak_component",
    "evaluate_total",
    "evaluate_peak_component",
    "evaluate_baseline",
    "sample_fit_curve",
]
