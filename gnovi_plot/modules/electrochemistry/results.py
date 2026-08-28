"""Persisted Cyclic Voltammetry analysis result types.

``CVCycleAnalysisResult`` is an :class:`~gnovi_plot.analysis.results.
AnalysisResult` subclass (kind ``"cv_peaks"``) registered through the
existing ``@register_result_kind`` polymorphic mechanism -- so project
save/load, ``PanelResultHistory``, Extract/Focus, and the bottom Results
tab all work with no format-version bump and no XRD-style special-casing,
exactly like ``modules.xrd.results.XRDAnalysisResult``.

Generic provenance (``source_dataset_id``/``source_series_id``/
``source_panel_id``/``result_id``/``engine``/``operation``/``parameters``/
...) lives on ``AnalysisResult`` and is NOT duplicated here. This module
adds only what is genuinely CV-specific and available now: the sign
convention, the cycle/sweep segmentation, the measured peaks, and the
couple metrics.

Deliberately NO speculative fields for multi-scan-rate / Randles-Sevcik /
reversibility classification -- those are CV-3 and get their own result
types then.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import ClassVar

from gnovi_plot.analysis.results import ENGINE_GNOVI, AnalysisResult, register_result_kind
from gnovi_plot.core.app_info import __version__ as _APP_VERSION
from gnovi_plot.modules.electrochemistry.common import SweepSegment
from gnovi_plot.modules.electrochemistry.cv import (
    ORIGIN_AUTOMATIC,
    PROCESS_ANODIC,
    PROCESS_CATHODIC,
    RATIO_BASIS_CORRECTED,
    RATIO_BASIS_RAW,
    CVCoupleMetrics,
    CVPeakMeasurement,
    CVPeakSeed,
    couple_metrics,
)

CV_OPERATION_PEAK_ANALYSIS = "cv_peak_analysis"

# How the cycle used for analysis was determined.
CYCLE_CONFIDENCE_EXPLICIT = "explicit"  # a metadata / segment column said so
CYCLE_CONFIDENCE_DETECTED = "detected"  # segment_sweeps + pair_cycles
CYCLE_CONFIDENCE_MANUAL = "manual"  # the researcher picked the rows

#: Authoritative per-peak detail-table columns. Shared by
#: ``CVCycleAnalysisResult.detail_table()`` (the bottom Results tab) and
#: any future CSV export so a researcher sees identical headers.
PEAK_TABLE_COLUMNS = [
    "Peak #",
    "Sweep",
    "Process",
    "Origin",
    "Enabled",
    "E_peak (V)",
    "I_peak raw (A)",
    "Baseline",
    "I_peak corrected (A)",
    "Prominence",
]

_VALID_RATIO_BASIS = {RATIO_BASIS_CORRECTED, RATIO_BASIS_RAW}


def _fmt_opt(value: float | None, spec: str = ".6g") -> str:
    return "—" if value is None else format(value, spec)


@dataclass
class CVSweepInfo:
    """Serializable snapshot of one sweep segment (geometry only)."""

    start: int
    end: int
    direction: str
    e_start: float
    e_end: float

    @classmethod
    def from_segment(cls, seg: SweepSegment) -> "CVSweepInfo":
        return cls(
            start=int(seg.start),
            end=int(seg.end),
            direction=seg.direction,
            e_start=float(seg.e_start),
            e_end=float(seg.e_end),
        )

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "direction": self.direction,
            "e_start": self.e_start,
            "e_end": self.e_end,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CVSweepInfo":
        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            direction=data["direction"],
            e_start=float(data["e_start"]),
            e_end=float(data["e_end"]),
        )


@dataclass
class CVBaselineInfo:
    """Serializable record of the baseline used for one measured peak.

    ``method`` is ``"linear"`` or ``"none"``. ``anchor_ranges`` are the
    positional row ranges the caller marked as background.
    ``baseline_current_a`` is the baseline evaluated at the peak potential
    (``None`` when ``method == "none"``). The baseline curve itself is not
    stored -- it is cheap to recompute from the anchors and the source
    data.
    """

    method: str
    anchor_ranges: list[tuple[int, int]]
    baseline_current_a: float | None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "anchor_ranges": [list(r) for r in self.anchor_ranges],
            "baseline_current_a": self.baseline_current_a,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CVBaselineInfo":
        return cls(
            method=data.get("method", "none"),
            anchor_ranges=[tuple(r) for r in data.get("anchor_ranges", [])],
            baseline_current_a=data.get("baseline_current_a"),
        )

    @classmethod
    def none(cls) -> "CVBaselineInfo":
        return cls(method="none", anchor_ranges=[], baseline_current_a=None)


@dataclass
class CVPeakResult:
    """One measured CV peak, ready to persist.

    ``i_peak_raw_a`` is the current at the extremum exactly as imported.
    ``i_peak_corrected_a`` is ``i_peak_raw_a - baseline(Ep)`` and is
    ``None`` when no baseline was drawn -- in which case ``i_peak_raw_a`` is
    a RAW EXTREMUM and the detail table shows it as such, never as a
    baseline-corrected Ipa/Ipc.
    """

    peak_id: str
    sweep: str
    process: str
    origin: str
    enabled: bool
    e_peak_v: float
    i_peak_raw_a: float
    i_peak_corrected_a: float | None
    baseline: CVBaselineInfo | None
    prominence: float | None

    def to_dict(self) -> dict:
        return {
            "peak_id": self.peak_id,
            "sweep": self.sweep,
            "process": self.process,
            "origin": self.origin,
            "enabled": self.enabled,
            "e_peak_v": self.e_peak_v,
            "i_peak_raw_a": self.i_peak_raw_a,
            "i_peak_corrected_a": self.i_peak_corrected_a,
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "prominence": self.prominence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CVPeakResult":
        baseline = data.get("baseline")
        return cls(
            peak_id=data.get("peak_id") or uuid.uuid4().hex,
            sweep=data["sweep"],
            process=data["process"],
            origin=data.get("origin", ORIGIN_AUTOMATIC),
            enabled=data.get("enabled", True),
            e_peak_v=float(data["e_peak_v"]),
            i_peak_raw_a=float(data["i_peak_raw_a"]),
            i_peak_corrected_a=data.get("i_peak_corrected_a"),
            baseline=CVBaselineInfo.from_dict(baseline) if baseline is not None else None,
            prominence=data.get("prominence"),
        )


@register_result_kind
@dataclass
class CVCycleAnalysisResult(AnalysisResult):
    """Result of one CV peak-analysis pass against a single cycle.

    NOT a multi-scan-rate result and NOT a reversibility verdict -- it
    carries the sign convention, the sweep segmentation, the measured
    peaks, and (when an anodic/cathodic couple was measured) ΔEp, the
    midpoint potential E½, and the explicitly-labelled ``|Ipa|/|Ipc|`` /
    ``|Ipc|/|Ipa|`` ratios with the basis they were computed on.
    """

    kind: ClassVar[str] = "cv_peaks"

    sign_convention: str
    cycle_index: int | None
    cycle_confidence: str
    cycle_complete: bool
    sweeps: list[CVSweepInfo]
    peaks: list[CVPeakResult]
    delta_ep_v: float | None
    e_half_v: float | None
    peak_current_ratio_ipa_over_ipc: float | None
    peak_current_ratio_ipc_over_ipa: float | None
    peak_current_ratio_basis: str | None
    # `peak_id` of the two peaks that currently form the anodic/cathodic
    # couple the ΔEp / E½ / ratio numbers above were computed from -- so a
    # reader can see WHICH peak numbers those metrics belong to (see
    # `details()`). Defaulted (not required) so older `from_dict` calls and
    # every CV-1 test keep working unchanged. `None` when no clean couple
    # exists (0 or >1 enabled peak of a process, or a single-sweep result).
    couple_anodic_peak_id: str | None = None
    couple_cathodic_peak_id: str | None = None

    # --- display contract -------------------------------------------------

    def summary(self) -> str:
        enabled = sum(1 for p in self.peaks if p.enabled)
        cycle = "—" if self.cycle_index is None else str(self.cycle_index)
        parts = [f"CV peak analysis: cycle {cycle}, {enabled}/{len(self.peaks)} candidate(s)"]
        if self.delta_ep_v is not None:
            parts.append(f"ΔEp {self.delta_ep_v * 1e3:.0f} mV")
        if self.e_half_v is not None:
            parts.append(f"E½ {self.e_half_v:.3f} V")
        return ", ".join(parts)

    def _peak_position(self, peak_id: str | None) -> int | None:
        """1-based position of ``peak_id`` in :attr:`peaks`, or ``None``."""
        if peak_id is None:
            return None
        for position, peak in enumerate(self.peaks, start=1):
            if peak.peak_id == peak_id:
                return position
        return None

    def details(self) -> list[tuple[str, str]]:
        """A BOUNDED summary -- a fixed set of rows regardless of peak count
        (the per-peak view is :meth:`detail_table`, rendered in the wide
        bottom Results tab; see ``modules.xrd.results.XRDAnalysisResult.
        details`` for the layout bug a per-peak ``details()`` caused)."""
        enabled = sum(1 for p in self.peaks if p.enabled)
        directions = "/".join(s.direction for s in self.sweeps) or "—"
        rows: list[tuple[str, str]] = [
            ("Sign convention", str(self.sign_convention)),
            ("Cycle", "—" if self.cycle_index is None else str(self.cycle_index)),
            ("Cycle source", self.cycle_confidence),
            ("Cycle complete", "Yes" if self.cycle_complete else "No"),
            ("Sweeps", f"{len(self.sweeps)} ({directions})"),
            ("Peak candidates", str(len(self.peaks))),
            ("Enabled candidates", str(enabled)),
        ]

        pos_a = self._peak_position(self.couple_anodic_peak_id)
        pos_c = self._peak_position(self.couple_cathodic_peak_id)
        if pos_a is not None and pos_c is not None:
            rows.append(("Couple", f"peak #{pos_a} (anodic) + peak #{pos_c} (cathodic)"))
        elif self.delta_ep_v is None and self.e_half_v is None:
            rows.append(("Couple", "no anodic–cathodic couple in this cycle"))

        if self.delta_ep_v is not None:
            rows.append(("ΔEp", f"{self.delta_ep_v:.6g} V ({self.delta_ep_v * 1e3:.1f} mV)"))
        if self.e_half_v is not None:
            rows.append(("E½ (midpoint)", f"{self.e_half_v:.6g} V"))
        basis = self.peak_current_ratio_basis
        basis_suffix = f" ({basis.replace('_', ' ')})" if basis is not None else ""
        if self.peak_current_ratio_ipa_over_ipc is not None:
            rows.append(("|Ipa| / |Ipc|", f"{self.peak_current_ratio_ipa_over_ipc:.4g}{basis_suffix}"))
        if self.peak_current_ratio_ipc_over_ipa is not None:
            rows.append(("|Ipc| / |Ipa|", f"{self.peak_current_ratio_ipc_over_ipa:.4g}{basis_suffix}"))
        if basis == "raw_extremum" and self.peak_current_ratio_ipa_over_ipc is not None:
            rows.append(
                ("Note", "Ratio from raw extrema — draw peak baselines for a defensible value.")
            )
        return rows

    def detail_table(self) -> tuple[list[str], list[list[str]]]:
        rows: list[list[str]] = []
        for position, peak in enumerate(self.peaks, start=1):
            if peak.baseline is None or peak.baseline.method == "none":
                baseline_label = "none (raw extremum)"
            else:
                baseline_label = peak.baseline.method
            rows.append(
                [
                    str(position),
                    peak.sweep,
                    peak.process,
                    peak.origin,
                    "Yes" if peak.enabled else "No",
                    f"{peak.e_peak_v:.6g}",
                    f"{peak.i_peak_raw_a:.6g}",
                    baseline_label,
                    "—" if peak.i_peak_corrected_a is None else f"{peak.i_peak_corrected_a:.6g}",
                    _fmt_opt(peak.prominence, ".4g"),
                ]
            )
        return list(PEAK_TABLE_COLUMNS), rows

    def detail_table_title(self) -> str:
        return "CV peaks"

    # --- serialization --------------------------------------------------

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "sign_convention": self.sign_convention,
                "cycle_index": self.cycle_index,
                "cycle_confidence": self.cycle_confidence,
                "cycle_complete": self.cycle_complete,
                "sweeps": [s.to_dict() for s in self.sweeps],
                "peaks": [p.to_dict() for p in self.peaks],
                "delta_ep_v": self.delta_ep_v,
                "e_half_v": self.e_half_v,
                "peak_current_ratio_ipa_over_ipc": self.peak_current_ratio_ipa_over_ipc,
                "peak_current_ratio_ipc_over_ipa": self.peak_current_ratio_ipc_over_ipa,
                "peak_current_ratio_basis": self.peak_current_ratio_basis,
                "couple_anodic_peak_id": self.couple_anodic_peak_id,
                "couple_cathodic_peak_id": self.couple_cathodic_peak_id,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CVCycleAnalysisResult":
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
            operation=data.get("operation", CV_OPERATION_PEAK_ANALYSIS),
            parameters=dict(data.get("parameters", {})),
            sign_convention=data["sign_convention"],
            cycle_index=data.get("cycle_index"),
            cycle_confidence=data.get("cycle_confidence", CYCLE_CONFIDENCE_DETECTED),
            cycle_complete=data.get("cycle_complete", True),
            sweeps=[CVSweepInfo.from_dict(s) for s in data.get("sweeps", [])],
            peaks=[CVPeakResult.from_dict(p) for p in data.get("peaks", [])],
            delta_ep_v=data.get("delta_ep_v"),
            e_half_v=data.get("e_half_v"),
            peak_current_ratio_ipa_over_ipc=data.get("peak_current_ratio_ipa_over_ipc"),
            peak_current_ratio_ipc_over_ipa=data.get("peak_current_ratio_ipc_over_ipa"),
            peak_current_ratio_basis=data.get("peak_current_ratio_basis"),
            couple_anodic_peak_id=data.get("couple_anodic_peak_id"),
            couple_cathodic_peak_id=data.get("couple_cathodic_peak_id"),
        )


def peak_result_from_seed(
    seed: CVPeakSeed, baseline: CVBaselineInfo | None = None
) -> CVPeakResult:
    """Turn a detected/manual :class:`CVPeakSeed` into a persistable
    :class:`CVPeakResult`, carrying its raw seed current across. Baseline
    correction (setting ``i_peak_corrected_a``) is a measurement step a
    caller does separately via ``cv.measure_peak``; this only bridges the
    candidate model to the result model."""
    return CVPeakResult(
        peak_id=seed.id,
        sweep=seed.sweep,
        process=seed.process,
        origin=seed.origin,
        enabled=seed.enabled,
        e_peak_v=seed.potential_v,
        i_peak_raw_a=seed.current_a,
        i_peak_corrected_a=None,
        baseline=baseline,
        prominence=seed.prominence,
    )


def _pick_couple_member(indexed: list[tuple[int, CVPeakResult]]) -> CVPeakResult | None:
    """Choose one process's couple member from its enabled candidates,
    each paired with its position in the result's ``peaks`` list.

    Deterministic ordering, and NEVER a raw-current-magnitude ranking
    (raw current at the extremum is dominated by the charging background
    and is not comparable to a SciPy prominence):

    1. If any candidate carries a ``prominence`` (every automatic candidate
       from a normal Find Peaks does), the one with the LARGEST prominence
       wins -- ties broken toward the EARLIEST position for stability.
       Prominence is the "how far this stands above its surroundings"
       measure, so a genuine wave always outranks a small bump, and a
       stray manual click (no prominence) can never silently displace a
       real automatic couple member.
    2. Otherwise -- only manual candidates, or an automatic pass run with
       no prominence threshold at all, so nothing has a prominence -- the
       fallback is the candidate added LAST (highest position): a manual
       candidate's most recent deliberate placement, or the last automatic
       peak in detection order.

    A researcher who adds a manual candidate to REPLACE a spurious
    automatic one simply disables that automatic candidate (it is right
    there in the peak table), after which rule 2 (or rule 1 among the
    remaining automatics) picks the manual one.
    """
    if not indexed:
        return None
    with_prominence = [(idx, p) for idx, p in indexed if p.prominence is not None]
    if with_prominence:
        return max(with_prominence, key=lambda ip: (ip[1].prominence, -ip[0]))[1]
    return max(indexed, key=lambda ip: ip[0])[1]


def assign_couple(
    peaks: list[CVPeakResult],
) -> tuple[CVPeakResult | None, CVPeakResult | None]:
    """The anodic/cathodic couple for a cycle -- one ENABLED anodic
    candidate + one enabled cathodic candidate (see
    :func:`_pick_couple_member` for the exact, deterministic selection
    rule). ``unassigned`` and disabled peaks are never couple members.
    Either side is ``None`` when that process has no enabled candidate.
    """
    anodic = [(idx, p) for idx, p in enumerate(peaks) if p.enabled and p.process == PROCESS_ANODIC]
    cathodic = [(idx, p) for idx, p in enumerate(peaks) if p.enabled and p.process == PROCESS_CATHODIC]
    return _pick_couple_member(anodic), _pick_couple_member(cathodic)


def _as_measurement(peak: CVPeakResult) -> CVPeakMeasurement:
    return CVPeakMeasurement(
        potential_v=peak.e_peak_v,
        i_peak_raw_a=peak.i_peak_raw_a,
        i_peak_corrected_a=peak.i_peak_corrected_a,
        baseline_current_a=(peak.baseline.baseline_current_a if peak.baseline is not None else None),
        process=peak.process,
        sweep=peak.sweep,
        index=-1,
    )


def couple_from_peak_results(
    peaks: list[CVPeakResult],
) -> tuple[CVCoupleMetrics | None, str | None, str | None]:
    """``(metrics, anodic_peak_id, cathodic_peak_id)`` for the couple
    :func:`assign_couple` picks from ``peaks``. ``metrics`` (and both ids)
    are ``None`` when a full anodic+cathodic couple is not available. Pure:
    reuses the CV-1 :func:`~gnovi_plot.modules.electrochemistry.cv.
    couple_metrics` on measurements reconstructed from the stored peak
    results, so ΔEp / E½ / ratio here always agree with a fresh measurement.
    """
    anodic, cathodic = assign_couple(peaks)
    if anodic is None or cathodic is None:
        return None, None, None
    metrics = couple_metrics(_as_measurement(anodic), _as_measurement(cathodic))
    return metrics, anodic.peak_id, cathodic.peak_id


def build_cv_cycle_analysis_result(
    *,
    source_dataset_id: str,
    x_column: str,
    y_column: str,
    sign_convention: str,
    sweeps: list[SweepSegment] | list[CVSweepInfo],
    peaks: list[CVPeakResult],
    cycle_index: int | None = None,
    cycle_confidence: str = CYCLE_CONFIDENCE_DETECTED,
    cycle_complete: bool = True,
    couple: CVCoupleMetrics | None = None,
    couple_anodic_peak_id: str | None = None,
    couple_cathodic_peak_id: str | None = None,
    source_dataset_name: str | None = None,
    source_series_id: str | None = None,
    source_series_label: str | None = None,
    row_range: tuple[int, int] | None = None,
    source_panel_id: str | None = None,
    parameters: dict | None = None,
) -> CVCycleAnalysisResult:
    """Construct a fresh :class:`CVCycleAnalysisResult` -- the CV
    counterpart of ``modules.xrd.results.build_xrd_analysis_result``:
    ``result_id`` is always freshly generated, ``engine``/``engine_version``/
    ``operation`` are the native-GNOVI values, and every provenance argument
    is threaded straight through (this function never imports ``Dataset``,
    Qt, or anything from ``gnovi_plot.plotting``).

    When ``couple`` is supplied its ΔEp / E½ / ratios are copied onto the
    result; otherwise those fields are ``None``.
    """
    sweep_infos = [
        s if isinstance(s, CVSweepInfo) else CVSweepInfo.from_segment(s) for s in sweeps
    ]
    if couple is not None:
        delta_ep = couple.delta_ep_v
        e_half = couple.e_half_v
        r_ac = couple.ratio_ipa_over_ipc
        r_ca = couple.ratio_ipc_over_ipa
        basis = couple.ratio_basis
    else:
        delta_ep = e_half = r_ac = r_ca = basis = None

    return CVCycleAnalysisResult(
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
        operation=CV_OPERATION_PEAK_ANALYSIS,
        parameters=dict(parameters) if parameters is not None else {},
        sign_convention=sign_convention,
        cycle_index=cycle_index,
        cycle_confidence=cycle_confidence,
        cycle_complete=cycle_complete,
        sweeps=sweep_infos,
        peaks=list(peaks),
        delta_ep_v=delta_ep,
        e_half_v=e_half,
        peak_current_ratio_ipa_over_ipc=r_ac,
        peak_current_ratio_ipc_over_ipa=r_ca,
        peak_current_ratio_basis=basis,
        couple_anodic_peak_id=couple_anodic_peak_id,
        couple_cathodic_peak_id=couple_cathodic_peak_id,
    )
