"""Raman peak-detection analysis result -- the `AnalysisResult` subclass
this milestone adds (see `analysis.results.AnalysisResult`, `analysis.
panel_results.PanelResultHistory`).

Deliberately no `modules/raman/provenance.py`: engine-neutral provenance
(`engine`/`engine_version`/`operation`/`parameters`) lives on
`AnalysisResult` itself (see that class's own docstring) because nothing
about it is Raman-specific. `RamanAnalysisResult` below only adds what's
genuinely Raman-specific: the detected peaks themselves.

Unlike `modules.xrd.results.XRDAnalysisResult`, there is no Raman
equivalent of XRD radiation/wavelength or Bragg's-law d-spacing -- Raman
shift is already reported directly in cm^-1 by the instrument, with no
per-measurement calibration constant this result needs to carry.

`AnalysisResult.parameters` (inherited) is this milestone's only
provenance surface for detection settings (e.g. `{"detection":
{"prominence": ..., "distance": ...}}`). Preprocessing provenance
(baseline/smoothing) is deliberately NOT assumed or hard-coded into
`details()` here -- that GUI/preprocessing state doesn't exist yet (see
Issue #73); baking in an assumed `parameters["preprocessing"]` shape now
would invent GUI state ahead of the GUI that actually produces it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import ClassVar

from gnovi_plot.analysis.results import ENGINE_GNOVI, AnalysisResult, register_result_kind
from gnovi_plot.core.app_info import __version__ as _APP_VERSION
from gnovi_plot.modules.raman.peaks import RamanPeakSeed

OPERATION_PEAK_DETECTION = "raman_peak_detection"

# The authoritative detailed peak-table columns -- deliberately excludes
# anything implying profile fitting (fitted center, linewidth/FWHM, area,
# model/quality) -- this milestone has no fitting yet (see this module's
# own docstring). Shared by `RamanAnalysisResult.detail_table()` (the
# bottom Results-tab table) and any future CSV export, mirroring
# `modules.xrd.results.PEAK_TABLE_COLUMNS`'s own convention so a
# researcher sees identical headers on screen and in an exported CSV.
PEAK_TABLE_COLUMNS = [
    "Peak #",
    "Raman shift (cm⁻¹)",
    "Observed intensity",
    "Prominence",
    "Origin",
    "Enabled",
]


def _format_optional_number(value: float | int | None) -> str:
    """`"—"` for `None` (the setting wasn't used), otherwise a compact
    numeric string -- shared by `RamanAnalysisResult.details()`'s
    Prominence/Minimum-separation rows, mirroring `modules.xrd.results`'s
    own helper of the same name."""
    if value is None:
        return "—"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


@register_result_kind
@dataclass
class RamanAnalysisResult(AnalysisResult):
    """The result of one Raman peak-detection pass against a panel's
    data: the peak candidates found (see `modules.raman.peaks.
    detect_raman_peaks`) -- NOT a fitted-peak result (no linewidth/FWHM/
    area/uncertainty here; peak-profile fitting is explicitly outside
    this milestone, see Issue #72). Deliberately composable rather than a
    single monolithic object with placeholder `None` fields for
    not-yet-built features: a later milestone adding peak fitting extends
    `RamanPeakSeed`'s successor / adds its own `AnalysisResult` subclass,
    not this one -- the same convention `XRDAnalysisResult`'s own
    docstring already established for XRD.
    """

    kind: ClassVar[str] = "raman_peaks"

    peaks: list[RamanPeakSeed]

    def summary(self) -> str:
        enabled = sum(1 for p in self.peaks if p.enabled)
        return f"Raman peak detection: {enabled}/{len(self.peaks)} peak candidate(s)"

    def details(self) -> list[tuple[str, str]]:
        """A BOUNDED summary -- deliberately never one row per peak (see
        `XRDAnalysisResult.details()`'s own docstring for the layout bug
        that motivated this convention: the full per-peak view belongs in
        `detail_table()` below, rendered in a bounded, internally-
        scrolling table, never here).

        Deliberately reports only detection settings (`parameters
        ["detection"]`) -- this milestone has no preprocessing GUI yet
        (see the module docstring), so no background/smoothing rows are
        assumed here the way XRD's own `details()` does."""
        enabled = sum(1 for p in self.peaks if p.enabled)
        detection = self.parameters.get("detection") or {}

        return [
            ("Peak candidates", str(len(self.peaks))),
            ("Enabled", str(enabled)),
            ("Prominence", _format_optional_number(detection.get("prominence"))),
            ("Minimum separation", _format_optional_number(detection.get("distance"))),
        ]

    def detail_table(self) -> tuple[list[str], list[list[str]]]:
        """One row per peak candidate -- the authoritative detailed peak
        view (see `PEAK_TABLE_COLUMNS`), mirroring `XRDAnalysisResult.
        detail_table()`'s own convention. No d-spacing column: there is
        no Raman equivalent."""
        rows: list[list[str]] = []
        for position, peak in enumerate(self.peaks, start=1):
            rows.append(
                [
                    str(position),
                    f"{peak.raman_shift:.4f}",
                    f"{peak.intensity:.6g}",
                    f"{peak.prominence:.4g}" if peak.prominence is not None else "—",
                    peak.origin,
                    "Yes" if peak.enabled else "No",
                ]
            )
        return list(PEAK_TABLE_COLUMNS), rows

    def detail_table_title(self) -> str:
        return "Detected peaks"

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["peaks"] = [peak.to_dict() for peak in self.peaks]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RamanAnalysisResult":
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
            peaks=[RamanPeakSeed.from_dict(peak) for peak in data.get("peaks", [])],
        )


def build_raman_analysis_result(
    *,
    source_dataset_id: str,
    x_column: str,
    y_column: str,
    peaks: list[RamanPeakSeed],
    source_dataset_name: str | None = None,
    source_series_id: str | None = None,
    source_series_label: str | None = None,
    row_range: tuple[int, int] | None = None,
    source_panel_id: str | None = None,
    parameters: dict | None = None,
) -> RamanAnalysisResult:
    """Construct a fresh `RamanAnalysisResult` -- mirrors
    `modules.xrd.results.build_xrd_analysis_result`'s own construction
    convention: `result_id` is always freshly generated here (never
    caller-supplied), `engine`/`engine_version`/`operation` are always
    the native-GNOVI values (see `AnalysisResult.engine`'s own
    docstring), and every provenance argument is threaded straight
    through, opaque to this function -- it never imports `Dataset`, Qt,
    or anything from `gnovi_plot.plotting`.

    Pure construction only -- this does not run detection itself (see
    `modules.raman.peaks.detect_raman_peaks`); the caller supplies
    already-computed `peaks` and whatever `parameters` it wants
    recorded.
    """
    return RamanAnalysisResult(
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
        peaks=peaks,
    )
