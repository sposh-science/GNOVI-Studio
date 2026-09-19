"""GNOVI's native Raman numerical foundation: peak detection and the
Raman analysis result model -- pure NumPy/SciPy numerical code, no Qt, no
Matplotlib, no Raman GUI.

Explicitly NOT part of this foundation yet (see Issue #72's own scope
notes): background/smoothing preprocessing wiring, peak-profile fitting
(Gaussian/Lorentzian/pseudo-Voigt), fitted linewidth/FWHM, material
identification, spectral database matching, automatic vibrational-mode
assignment, D/G ratio interpretation, crystallite-size calculation,
stress/strain interpretation, phase identification, chemometrics/PCA,
Raman mapping, or batch processing.

This is a thin re-export of the module's public API; the actual
implementation lives in the individual submodules below.
"""

from __future__ import annotations

from gnovi_plot.modules.raman.peaks import (
    ORIGIN_AUTOMATIC,
    ORIGIN_MANUAL,
    InvalidPeakDetectionError,
    RamanPeakSeed,
    detect_raman_peaks,
)
from gnovi_plot.modules.raman.results import (
    OPERATION_PEAK_DETECTION,
    PEAK_TABLE_COLUMNS,
    RamanAnalysisResult,
    build_raman_analysis_result,
)

__all__ = [
    "ORIGIN_AUTOMATIC",
    "ORIGIN_MANUAL",
    "InvalidPeakDetectionError",
    "RamanPeakSeed",
    "detect_raman_peaks",
    "OPERATION_PEAK_DETECTION",
    "PEAK_TABLE_COLUMNS",
    "RamanAnalysisResult",
    "build_raman_analysis_result",
]
