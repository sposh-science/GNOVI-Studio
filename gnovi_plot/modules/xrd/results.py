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
from gnovi_plot.modules.xrd.bragg import InvalidBraggInputError, d_spacing
from gnovi_plot.modules.xrd.peaks import XRDPeakSeed
from gnovi_plot.modules.xrd.radiation import Radiation

OPERATION_PEAK_DETECTION = "xrd_peak_detection"

# The authoritative detailed peak-table columns -- deliberately excludes
# anything implying profile fitting (fitted center, FWHM, area, model/
# quality) -- XRD-2 has no fitting yet (see this class's own docstring).
# Shared by `XRDAnalysisResult.detail_table()` (the bottom Results-tab
# table, which is now the one authoritative detailed peak view) and
# `gui.widgets.xrd_analysis_section.XRDAnalysisSection.export_peak_table_csv`
# so a researcher sees identical headers on screen and in an exported CSV.
PEAK_TABLE_COLUMNS = [
    "Peak #",
    "Seed 2θ (°)",
    "Observed intensity",
    "Prominence",
    "d-spacing (Å)",
    "Origin",
    "Enabled",
]


# Maps `modules.xrd.preprocessing`'s internal `BaselineResult.method`
# values ("polynomial"/"arpls") to the exact display strings
# `XRDAnalysisSection`'s own Background dropdown already uses -- so a
# researcher sees the SAME label ("arPLS", not "arpls") in the Results tab
# as they picked in the workflow controls.
_BACKGROUND_METHOD_LABELS = {"polynomial": "Polynomial", "arpls": "arPLS"}


def _format_optional_number(value: float | int | None) -> str:
    """`"—"` for `None` (the setting wasn't used), otherwise a compact
    numeric string -- shared by `XRDAnalysisResult.details()`'s
    Prominence/Minimum-separation rows."""
    if value is None:
        return "—"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


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
        """A BOUNDED summary -- deliberately never one row per peak.

        An earlier version of this method appended one row per
        `XRDPeakSeed`, which is fine for a handful of peaks but not for
        the hundreds/thousands `scipy.signal.find_peaks` can return on
        real noisy data with a permissive prominence: `AnalysisResultView`
        renders `details()` into a plain `QFormLayout` with no bound on
        its own size, so that many rows gave the containing widget a
        `minimumSizeHint` of literally tens of thousands of pixels --
        which the Results tab (and therefore GNOVI's central vertical
        splitter, see `gui.widgets.bottom_panel.BottomPanel`) has no way
        to display within, permanently starving the plot canvas of space
        with no way to drag it back. The full, row-per-peak view belongs
        in `detail_table()` (below), rendered by `AnalysisResultView` in
        the bottom Results tab as a bounded, internally-scrolling table
        built for arbitrary row counts that never dictates its parent's
        size -- this method must stay a small, FIXED number of rows
        regardless of how many peaks were found, so it can never do that
        again to any future analysis tool that reuses `AnalysisResultView`
        either."""
        enabled = sum(1 for p in self.peaks if p.enabled)
        preprocessing = self.parameters.get("preprocessing") or {}
        background = preprocessing.get("background")
        smoothing = preprocessing.get("smoothing")
        detection = self.parameters.get("detection") or {}

        rows: list[tuple[str, str]] = [
            ("Radiation", f"{self.radiation.label} (λ = {self.radiation.wavelength_angstrom:.6g} Å)"),
            ("Peak candidates", str(len(self.peaks))),
            ("Enabled", str(enabled)),
            ("Background", _BACKGROUND_METHOD_LABELS.get(background.get("method"), "None") if background else "None"),
            ("Smoothing", "On" if smoothing else "Off"),
            ("Detection input", str(self.parameters.get("detection_input", "raw")).replace("_", " ").capitalize()),
            ("Prominence", _format_optional_number(detection.get("prominence"))),
            ("Minimum separation", _format_optional_number(detection.get("distance"))),
        ]
        return rows

    def _peak_d_spacing(self, peak: XRDPeakSeed) -> float | None:
        """First-order Bragg d-spacing for `peak`'s seed 2θ at this
        result's own radiation -- `None` if the angle is outside the
        physically valid range (see `modules.xrd.bragg.d_spacing`). Uses
        `self.radiation`, so it always reflects whatever radiation the
        result currently carries (a later radiation change re-displays the
        result and this recomputes)."""
        try:
            return float(d_spacing(peak.two_theta, self.radiation.wavelength_angstrom))
        except InvalidBraggInputError:
            return None

    def detail_table(self) -> tuple[list[str], list[list[str]]]:
        """One row per peak candidate -- the authoritative detailed peak
        view (see `PEAK_TABLE_COLUMNS`). Rendered by `gui.widgets.
        analysis_result_view.AnalysisResultView` in the bottom Results tab,
        which bounds its own height and scrolls internally regardless of
        how many peaks this returns (`details()` above stays the small,
        fixed summary alongside it)."""
        rows: list[list[str]] = []
        for position, peak in enumerate(self.peaks, start=1):
            d = self._peak_d_spacing(peak)
            rows.append(
                [
                    str(position),
                    f"{peak.two_theta:.4f}",
                    f"{peak.intensity:.6g}",
                    f"{peak.prominence:.4g}" if peak.prominence is not None else "—",
                    f"{d:.4f}" if d is not None else "—",
                    peak.origin,
                    "Yes" if peak.enabled else "No",
                ]
            )
        return list(PEAK_TABLE_COLUMNS), rows

    def detail_table_title(self) -> str:
        return "Detected peaks"

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
