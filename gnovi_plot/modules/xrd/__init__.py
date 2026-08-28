"""GNOVI's native XRD numerical foundation: radiation/wavelength handling,
Bragg's-law d-spacing, background/smoothing preprocessing, and peak
detection -- pure NumPy/SciPy(+optional pybaselines) numerical code, no Qt,
no Matplotlib, no XRD GUI yet.

Explicitly NOT part of this milestone (see PROJECT_GUIDE.md's XRD roadmap
notes): profile/peak-shape fitting, Scherrer crystallite-size calculation,
phase identification, Rietveld refinement, quantitative phase analysis, or
any external-engine integration (GSAS-II/pyFAI/Profex/BGMN).

This is a thin re-export of the module's public API; the actual
implementation lives in the individual submodules below.
"""

from __future__ import annotations

from gnovi_plot.modules.xrd.bragg import InvalidBraggInputError, d_spacing
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
]
