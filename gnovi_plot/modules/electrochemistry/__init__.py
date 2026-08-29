"""GNOVI's native Electrochemistry numerical foundation.

Cyclic Voltammetry (CV) is the first technique; the package is named
``electrochemistry`` rather than ``cv`` because Linear Sweep Voltammetry,
chronoamperometry/-potentiometry, GCD / charge-discharge, and EIS are the
intended later members of the same family (see PROJECT_GUIDE.md). Each
future technique gets its own module file beside ``cv.py``; there is
deliberately no ``ElectrochemicalExperiment`` base class, no
``ElectrochemicalResult`` intermediate superclass, and no plugin system --
one technique does not justify an abstraction layer.

CV-1 (this milestone) is the NUMERICAL FOUNDATION ONLY -- pure NumPy/SciPy
code (plus pandas only where GNOVI's ``Dataset`` infrastructure already
uses it), no Qt, no Matplotlib, no CV GUI.

Explicitly NOT part of CV-1 (see PROJECT_GUIDE.md's electrochemistry
roadmap notes): any CV GUI, a smoothing workflow, multi-scan-rate
aggregation / ``scan_rate.py``, Randles-Sevcik diffusion-coefficient
calculation, reversibility classification, the Nicholson peak-current
ratio, LSV/GCD/EIS, vendor import adapters, and reference-electrode
potential conversion.

This is a thin re-export of the module's public API; the implementation
lives in the submodules below.
"""

from __future__ import annotations

from gnovi_plot.modules.electrochemistry.common import (
    CHARGE_UNITS,
    CURRENT_UNITS,
    POTENTIAL_UNITS,
    SCAN_RATE_UNITS,
    SWEEP_FALLING,
    SWEEP_RISING,
    ChargeIntegrationError,
    CurrentSignConvention,
    ElectrochemistryError,
    ElectrodeContext,
    InvalidElectrodeContextError,
    SweepSegment,
    SweepSegmentationError,
    UnknownUnitError,
    convert_units,
    current_to_amperes,
    integrate_current,
    oxidative_sign,
    potential_to_volts,
    scan_rate_to_v_per_s,
    segment_sweeps,
)
from gnovi_plot.modules.electrochemistry.cv import (
    ORIGIN_AUTOMATIC,
    ORIGIN_MANUAL,
    PROCESS_ANODIC,
    PROCESS_CATHODIC,
    PROCESS_UNASSIGNED,
    RATIO_BASIS_CORRECTED,
    RATIO_BASIS_RAW,
    CVAnalysisError,
    CVBaseline,
    CVCoupleMetrics,
    CVPeakMeasurement,
    CVPeakSeed,
    Cycle,
    InvalidCVInputError,
    ambiguous_segmentation,
    couple_metrics,
    default_prominence,
    detect_cv_peaks,
    local_linear_baseline,
    measure_peak,
    mv_to_sample_distance,
    pair_cycles,
)
from gnovi_plot.modules.electrochemistry.results import (
    CV_OPERATION_PEAK_ANALYSIS,
    CYCLE_CONFIDENCE_DETECTED,
    CYCLE_CONFIDENCE_EXPLICIT,
    CYCLE_CONFIDENCE_MANUAL,
    CVBaselineInfo,
    CVCycleAnalysisResult,
    CVPeakResult,
    CVSweepInfo,
    assign_couple,
    build_cv_cycle_analysis_result,
    couple_from_peak_results,
    peak_result_from_seed,
)

__all__ = [
    # common
    "CHARGE_UNITS",
    "CURRENT_UNITS",
    "POTENTIAL_UNITS",
    "SCAN_RATE_UNITS",
    "SWEEP_FALLING",
    "SWEEP_RISING",
    "ChargeIntegrationError",
    "CurrentSignConvention",
    "ElectrochemistryError",
    "ElectrodeContext",
    "InvalidElectrodeContextError",
    "SweepSegment",
    "SweepSegmentationError",
    "UnknownUnitError",
    "convert_units",
    "current_to_amperes",
    "integrate_current",
    "oxidative_sign",
    "potential_to_volts",
    "scan_rate_to_v_per_s",
    "segment_sweeps",
    # cv
    "ORIGIN_AUTOMATIC",
    "ORIGIN_MANUAL",
    "PROCESS_ANODIC",
    "PROCESS_CATHODIC",
    "PROCESS_UNASSIGNED",
    "RATIO_BASIS_CORRECTED",
    "RATIO_BASIS_RAW",
    "CVAnalysisError",
    "CVBaseline",
    "CVCoupleMetrics",
    "CVPeakMeasurement",
    "CVPeakSeed",
    "Cycle",
    "InvalidCVInputError",
    "ambiguous_segmentation",
    "couple_metrics",
    "default_prominence",
    "detect_cv_peaks",
    "local_linear_baseline",
    "measure_peak",
    "mv_to_sample_distance",
    "pair_cycles",
    # results
    "CV_OPERATION_PEAK_ANALYSIS",
    "CYCLE_CONFIDENCE_DETECTED",
    "CYCLE_CONFIDENCE_EXPLICIT",
    "CYCLE_CONFIDENCE_MANUAL",
    "CVBaselineInfo",
    "CVCycleAnalysisResult",
    "CVPeakResult",
    "CVSweepInfo",
    "assign_couple",
    "build_cv_cycle_analysis_result",
    "couple_from_peak_results",
    "peak_result_from_seed",
]
