"""XRD peak-detection analysis result -- the `AnalysisResult` subclass
this milestone adds (see `analysis.results.AnalysisResult`, `analysis.
panel_results.PanelResultHistory`).

Deliberately no `modules/xrd/provenance.py`: engine-neutral provenance
(`engine`/`engine_version`/`operation`/`parameters`) lives on
`AnalysisResult` itself (see that class's own docstring) because nothing
about it is XRD-specific -- a future non-XRD analysis-result subtype
benefits from the exact same fields. `XRDAnalysisResult` below only adds
what's genuinely XRD-specific: which radiation was used, and the detected
peaks themselves.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import ClassVar

from gnovi_plot.analysis.results import ENGINE_GNOVI, AnalysisResult, register_result_kind
from gnovi_plot.core.app_info import __version__ as _APP_VERSION
from gnovi_plot.modules.xrd.peaks import XRDPeakSeed
from gnovi_plot.modules.xrd.radiation import Radiation

OPERATION_PEAK_DETECTION = "xrd_peak_detection"


@register_result_kind
@dataclass
class XRDAnalysisResult(AnalysisResult):
    """The result of one XRD peak-detection pass against a panel's data:
    which radiation/wavelength was assumed, and the peak candidates found
    (see `modules.xrd.peaks.detect_peaks`) -- NOT a fitted-peak result
    (no FWHM/area/d-spacing/uncertainty here; profile fitting is a later
    milestone, see PROJECT_GUIDE.md's XRD roadmap notes). Deliberately
    composable rather than a single monolithic object with placeholder
    `None` fields for not-yet-built features: a later milestone adding
    peak fitting extends `XRDPeakSeed`'s successor / adds its own
    `AnalysisResult` subclass, not this one.

    `AnalysisResult.parameters` (inherited) carries the preprocessing and
    detection settings actually used (e.g. `{"detection":
    {"prominence": ..., "distance": ...}, "preprocessing": {"method":
    "arpls", "lam": ...}}` or `{"preprocessing": None}` if raw data was
    analyzed directly) -- reusing the generic base field rather than a
    second, XRD-specific settings field, since it already exists and is
    already part of every `AnalysisResult`'s persisted provenance.
    """

    kind: ClassVar[str] = "xrd_peaks"

    radiation: Radiation
    peaks: list[XRDPeakSeed]

    def summary(self) -> str:
        enabled = sum(1 for p in self.peaks if p.enabled)
        return (
            f"XRD peak detection: {enabled}/{len(self.peaks)} peak candidate(s), "
            f"{self.radiation.label} (λ = {self.radiation.wavelength_angstrom:.6g} Å)"
        )

    def details(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = [
            ("Radiation", f"{self.radiation.label} (λ = {self.radiation.wavelength_angstrom:.6g} Å)"),
            ("Peak candidates", str(len(self.peaks))),
        ]
        for position, peak in enumerate(self.peaks, start=1):
            state = "" if peak.enabled else " [disabled]"
            prominence_text = f", prominence={peak.prominence:.4g}" if peak.prominence is not None else ""
            rows.append(
                (
                    f"Peak {position}{state}",
                    f"2θ = {peak.two_theta:.4f}°, I = {peak.intensity:.6g}, "
                    f"origin={peak.origin}{prominence_text}",
                )
            )
        return rows

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "radiation": self.radiation.to_dict(),
                "peaks": [peak.to_dict() for peak in self.peaks],
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "XRDAnalysisResult":
        row_range = data.get("row_range")
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
            operation=data.get("operation", OPERATION_PEAK_DETECTION),
            parameters=dict(data.get("parameters", {})),
            radiation=Radiation.from_dict(data["radiation"]),
            peaks=[XRDPeakSeed.from_dict(peak) for peak in data.get("peaks", [])],
        )


def build_xrd_analysis_result(
    *,
    source_dataset_id: str,
    x_column: str,
    y_column: str,
    radiation: Radiation,
    peaks: list[XRDPeakSeed],
    source_dataset_name: str | None = None,
    source_series_id: str | None = None,
    source_series_label: str | None = None,
    row_range: tuple[int, int] | None = None,
    source_panel_id: str | None = None,
    parameters: dict | None = None,
) -> XRDAnalysisResult:
    """Construct a fresh `XRDAnalysisResult` -- the XRD counterpart of
    `analysis.fitting.fit_curve`'s own result-construction convention:
    `result_id` is always freshly generated here (never caller-supplied),
    `engine`/`engine_version`/`operation` are always the native-GNOVI
    values (see `AnalysisResult.engine`'s own docstring), and every
    provenance argument is threaded straight through, opaque to this
    function -- it never imports `Dataset`, Qt, or anything from
    `gnovi_plot.plotting`.

    Pure construction only -- this does not run detection itself (see
    `modules.xrd.peaks.detect_peaks`) or preprocessing (see
    `modules.xrd.preprocessing`); the caller supplies already-computed
    `peaks` and whatever `parameters` it wants recorded.
    """
    return XRDAnalysisResult(
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
        operation=OPERATION_PEAK_DETECTION,
        parameters=dict(parameters) if parameters is not None else {},
        radiation=radiation,
        peaks=peaks,
    )
