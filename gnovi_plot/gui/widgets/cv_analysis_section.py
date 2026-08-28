"""Cyclic Voltammetry analysis workspace (CV-2A).

The narrow left-drawer *controls* for single-run CV: source series + current
sign convention + optional physical context, cycle selection, sweep
selection, and candidate peak detection with manual curation. The wide
per-peak table and the anodic/cathodic couple summary live in the bottom
Results tab (``gui.widgets.analysis_result_view.AnalysisResultView``, fed by
``CVCycleAnalysisResult.detail_table()`` / ``.details()``). Graph aids
(selected-cycle tint, sweep tint, switching-potential line, candidate /
enabled markers) are a LIVE-ONLY overlay, reconstructed each render from the
current result -- never a persisted ``PlotSeries``/annotation, never in a
publication export.

CV-2A scope: no interactive baseline (peaks are raw-extremum only; the
Baseline section is CV-2B), no smoothing, no multi-scan-rate, no
Randles-Sevcik, no derived curves, no CSV export. See the CV-2 Workflow
Audit and PROJECT_GUIDE.md.

"Find Peaks" always produces a BRAND NEW ``CVCycleAnalysisResult`` (a new
Analysis History entry) -- the "Run Fit" / XRD "Find Peaks" convention.
Manual add/remove/enable/disable, process reassignment, and a sign-
convention change all edit the current result IN PLACE (dirty + redisplay,
no new history entry). Changing the selected cycle or sweep does NOT create
an entry -- it re-arms detection; the next Find Peaks creates the entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.analysis.segments import (
    InvalidRowRangeError,
    OverlappingRowRangeError,
    RowRangeCollection,
)
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.modules.electrochemistry.common import (
    SWEEP_FALLING,
    SWEEP_RISING,
    CurrentSignConvention,
    ElectrodeContext,
    InvalidElectrodeContextError,
    SweepSegment,
    SweepSegmentationError,
    oxidative_sign,
    scan_rate_to_v_per_s,
    segment_sweeps,
)
from gnovi_plot.modules.electrochemistry.cv import (
    PROCESS_ANODIC,
    PROCESS_CATHODIC,
    PROCESS_UNASSIGNED,
    CVPeakSeed,
    Cycle,
    InvalidCVInputError,
    ambiguous_segmentation,
    default_prominence,
    detect_cv_peaks,
    mv_to_sample_distance,
    pair_cycles,
)
from gnovi_plot.modules.electrochemistry.results import (
    CYCLE_CONFIDENCE_DETECTED,
    CYCLE_CONFIDENCE_EXPLICIT,
    CYCLE_CONFIDENCE_MANUAL,
    CVCycleAnalysisResult,
    build_cv_cycle_analysis_result,
    couple_from_peak_results,
    peak_result_from_seed,
)
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries

_NO_SOURCE_TEXT = (
    "No plotted line/scatter series in the active panel yet -- add one from "
    "the 2D page first."
)
_PANEL3D_TEXT = (
    "Cyclic Voltammetry works on a 2D panel's plotted series -- switch to "
    "(or add) a 2D panel first."
)

# Short combo labels -- the full explanation is the combo tooltip + the
# help line under it (see __init__). Kept short so the combo's
# minimumSizeHint does not push the whole Analysis drawer wider than the
# width Curve Fitting / XRD already use.
_SIGN_LABELS = {
    CurrentSignConvention.ANODIC_POSITIVE: "Anodic-positive",
    CurrentSignConvention.CATHODIC_POSITIVE: "Cathodic-positive",
}
_SIGN_HELP = (
    "Which direction of your recorded current is oxidation. GNOVI never "
    "changes the data — this only sets the interpretation."
)

_CYCLE_SOURCE_AUTO = "auto"
_CYCLE_SOURCE_METADATA = "metadata"
_CYCLE_SOURCE_MANUAL = "manual"

_SWEEP_BOTH = "both"
_SWEEP_RISING_ONLY = "rising"
_SWEEP_FALLING_ONLY = "falling"

_SCAN_RATE_UNITS = ["mV/s", "V/s"]

# `process` value -> the OPPOSITE, used when the current sign convention is
# flipped (a peak that was an oxidation current is a reduction current under
# the other convention). `unassigned` is left alone.
_PROCESS_FLIP = {PROCESS_ANODIC: PROCESS_CATHODIC, PROCESS_CATHODIC: PROCESS_ANODIC}


def eligible_cv_series(figure: GnoviFigure) -> list[PlotSeries]:
    """Line/scatter series in the active panel usable as a CV source --
    excludes histograms (no ``y_column``), stale series, and anything that
    isn't a ``PlotSeries`` (a ``Panel3D``'s ``Series3D`` items). Empty when
    the active panel is a ``Panel3D``. Mirrors
    ``gui.widgets.xrd_analysis_section._eligible_series``.
    """
    if isinstance(figure.active_panel, Panel3D):
        return []
    return [
        s
        for s in figure.series
        if isinstance(s, PlotSeries) and s.y_column is not None and not s.stale
    ]


class CVAnalysisSection(QWidget):
    """See the module docstring. Emits the generic ``analysis_result_ready``
    for a fresh Find Peaks (a new history entry) and ``result_updated`` for
    an in-place edit of the current result."""

    analysis_result_ready = Signal(AnalysisResult)
    result_updated = Signal(AnalysisResult)
    overlay_changed = Signal()
    manual_peak_mode_changed = Signal(bool)
    status_message = Signal(str)
    add_to_plot_requested = Signal(list)  # plumbed for CV-2B; unused in CV-2A

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._manager = dataset_manager
        self._current_result: CVCycleAnalysisResult | None = None
        self._manual_peak_mode = False
        self._results_selected_rows: list[int] = []
        self._detection_defaults_touched = False
        # Cached per-refresh; recomputed from the source series.
        self._cycles: list[Cycle] = []

        # --- Source ----------------------------------------------------------
        self.source_combo = QComboBox()
        self.columns_label = QLabel("")
        self.columns_label.setWordWrap(True)
        self.status_label = QLabel(_NO_SOURCE_TEXT)
        self.status_label.setWordWrap(True)

        self.sign_combo = QComboBox()
        for convention, text in _SIGN_LABELS.items():
            self.sign_combo.addItem(text, convention.value)
        self.sign_combo.setToolTip(_SIGN_HELP)
        # Do not let the combo's content width drive the drawer's minimum
        # width (the XRD sidebar-overflow lesson).
        self.sign_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.sign_combo.setMinimumContentsLength(10)
        self.sign_help_label = QLabel(_SIGN_HELP)
        self.sign_help_label.setWordWrap(True)
        self.sign_help_label.setStyleSheet("color: palette(mid);")

        self.scan_rate_spin = QDoubleSpinBox()
        self.scan_rate_spin.setDecimals(4)
        self.scan_rate_spin.setRange(0.0, 1e6)
        self.scan_rate_spin.setSpecialValueText("(not set)")
        self.scan_rate_unit_combo = QComboBox()
        self.scan_rate_unit_combo.addItems(_SCAN_RATE_UNITS)
        scan_rate_row = QHBoxLayout()
        scan_rate_row.addWidget(self.scan_rate_spin)
        scan_rate_row.addWidget(self.scan_rate_unit_combo)

        # optional physical context (collapsed)
        self.area_spin = self._optional_spin(" cm²", decimals=5)
        self.n_spin = self._optional_spin("", decimals=2)
        self.conc_spin = self._optional_spin(" mM", decimals=4)
        self.temp_spin = self._optional_spin(" K", decimals=2, maximum=5000.0)
        self.ref_edit = QLineEdit()
        self.we_edit = QLineEdit()
        self.ce_edit = QLineEdit()
        self.electrolyte_edit = QLineEdit()
        phys_body = QWidget()
        phys_layout = QVBoxLayout(phys_body)
        phys_layout.setContentsMargins(0, 0, 0, 0)
        _phys_hint = QLabel("Optional — not needed for peak analysis.")
        _phys_hint.setWordWrap(True)
        phys_layout.addWidget(_phys_hint)
        for text, w in (
            ("Electrode area", self.area_spin),
            ("Number of electrons n", self.n_spin),
            ("Analyte concentration", self.conc_spin),
            ("Temperature", self.temp_spin),
            ("Reference electrode", self.ref_edit),
            ("Working electrode", self.we_edit),
            ("Counter electrode", self.ce_edit),
            ("Supporting electrolyte", self.electrolyte_edit),
        ):
            phys_layout.addWidget(QLabel(text))
            phys_layout.addWidget(w)
        self.physical_section = CollapsibleSection("Physical context", phys_body, expanded=False)

        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.addWidget(QLabel("Source series"))
        source_layout.addWidget(self.source_combo)
        source_layout.addWidget(self.columns_label)
        source_layout.addWidget(self.status_label)
        source_layout.addWidget(QLabel("Current sign convention"))
        source_layout.addWidget(self.sign_combo)
        source_layout.addWidget(self.sign_help_label)
        source_layout.addWidget(QLabel("Scan rate (optional)"))
        source_layout.addLayout(scan_rate_row)
        source_layout.addWidget(self.physical_section)

        # --- Cycle Selection ----------------------------------------------
        self.cycle_source_combo = QComboBox()
        self.cycle_source_combo.addItem("Auto-detect", _CYCLE_SOURCE_AUTO)
        self.cycle_source_combo.addItem("Metadata column…", _CYCLE_SOURCE_METADATA)
        self.cycle_source_combo.addItem("Manual (row ranges)", _CYCLE_SOURCE_MANUAL)
        self.metadata_column_combo = QComboBox()
        self.manual_ranges_edit = QLineEdit()
        self.manual_ranges_edit.setPlaceholderText("e.g. 0-1600, 1600-3200")
        self.cycle_status_label = QLabel("")
        self.cycle_status_label.setWordWrap(True)
        self.cycle_combo = QComboBox()
        self.cycle_prev_button = QPushButton("◀")
        self.cycle_next_button = QPushButton("▶")
        self.cycle_prev_button.setMaximumWidth(32)
        self.cycle_next_button.setMaximumWidth(32)
        cycle_pick_row = QHBoxLayout()
        cycle_pick_row.addWidget(self.cycle_combo, 1)
        cycle_pick_row.addWidget(self.cycle_prev_button)
        cycle_pick_row.addWidget(self.cycle_next_button)
        self.cycle_confidence_label = QLabel("")

        cycle_group = QGroupBox("Cycle Selection")
        cycle_layout = QVBoxLayout(cycle_group)
        cycle_layout.addWidget(QLabel("Cycle source"))
        cycle_layout.addWidget(self.cycle_source_combo)
        cycle_layout.addWidget(self.metadata_column_combo)
        cycle_layout.addWidget(self.manual_ranges_edit)
        cycle_layout.addWidget(self.cycle_status_label)
        cycle_layout.addLayout(cycle_pick_row)
        cycle_layout.addWidget(self.cycle_confidence_label)

        # --- Sweep Selection --------------------------------------------
        self.sweep_combo = QComboBox()
        self.sweep_combo.addItem("Both sweeps", _SWEEP_BOTH)
        self.sweep_combo.addItem("Rising sweep only", _SWEEP_RISING_ONLY)
        self.sweep_combo.addItem("Falling sweep only", _SWEEP_FALLING_ONLY)
        self.sweep_readout_label = QLabel("")
        self.sweep_readout_label.setWordWrap(True)

        sweep_group = QGroupBox("Sweep Selection")
        sweep_layout = QVBoxLayout(sweep_group)
        sweep_layout.addWidget(QLabel("Analyse"))
        sweep_layout.addWidget(self.sweep_combo)
        sweep_layout.addWidget(self.sweep_readout_label)

        # --- Peak Detection -------------------------------------------
        # CV currents span a wide dynamic range (mA to nA), so prominence
        # is a free-text float field (accepts "5e-7"), not a fixed-decimal
        # spinbox. Blank / unparseable means "no prominence threshold".
        self.prominence_edit = QLineEdit()
        self.prominence_edit.setPlaceholderText("(auto)")
        self.min_sep_spin = QDoubleSpinBox()
        self.min_sep_spin.setDecimals(1)
        self.min_sep_spin.setRange(0.0, 1e6)
        self.min_sep_spin.setSuffix(" mV")
        self.width_check = QCheckBox("Use minimum width")
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setDecimals(1)
        self.width_spin.setRange(0.0, 1e6)
        self.width_spin.setSuffix(" mV")
        adv_body = QWidget()
        adv_layout = QVBoxLayout(adv_body)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.addWidget(self.width_check)
        adv_layout.addWidget(self.width_spin)
        self.advanced_section = CollapsibleSection("Advanced", adv_body, expanded=False)

        self.find_peaks_button = QPushButton("Find Peaks")
        self.find_peaks_button.setProperty("primary", True)
        self.detection_status_label = QLabel("")
        self.detection_status_label.setWordWrap(True)

        self.add_peak_button = QPushButton("Add Peak (click graph)")
        self.add_peak_button.setCheckable(True)
        self.remove_peak_button = QPushButton("Remove Selected")
        self.toggle_enabled_button = QPushButton("Enable/Disable Selected")
        self.set_process_button = QPushButton("Set Process ▾")
        process_menu = QMenu(self.set_process_button)
        for label, value in (
            ("Anodic", PROCESS_ANODIC),
            ("Cathodic", PROCESS_CATHODIC),
            ("Unassigned", PROCESS_UNASSIGNED),
        ):
            process_menu.addAction(label, lambda v=value: self._on_set_process(v))
        self.set_process_button.setMenu(process_menu)
        self.peak_actions_hint = QLabel(
            "Select candidate rows in the Results tab below, then use these actions."
        )
        self.peak_actions_hint.setWordWrap(True)

        detection_group = QGroupBox("Peak Detection")
        detection_layout = QVBoxLayout(detection_group)
        _raw_hint = QLabel("Peaks are detected on the raw current.")
        _raw_hint.setWordWrap(True)
        detection_layout.addWidget(_raw_hint)
        detection_layout.addWidget(QLabel("Prominence (current units)"))
        detection_layout.addWidget(self.prominence_edit)
        detection_layout.addWidget(QLabel("Minimum separation"))
        detection_layout.addWidget(self.min_sep_spin)
        detection_layout.addWidget(self.advanced_section)
        detection_layout.addWidget(self.find_peaks_button)
        detection_layout.addWidget(self.detection_status_label)
        detection_layout.addWidget(self.add_peak_button)
        detection_layout.addWidget(self.remove_peak_button)
        detection_layout.addWidget(self.toggle_enabled_button)
        detection_layout.addWidget(self.set_process_button)
        detection_layout.addWidget(self.peak_actions_hint)

        layout = QVBoxLayout(self)
        layout.addWidget(source_group)
        layout.addWidget(cycle_group)
        layout.addWidget(sweep_group)
        layout.addWidget(detection_group)
        layout.addStretch(1)

        # --- wiring -------------------------------------------------------
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.sign_combo.currentIndexChanged.connect(self._on_sign_convention_changed)
        self.cycle_source_combo.currentIndexChanged.connect(self._on_cycle_source_changed)
        self.metadata_column_combo.currentIndexChanged.connect(self._on_cycle_inputs_changed)
        self.manual_ranges_edit.editingFinished.connect(self._on_cycle_inputs_changed)
        self.cycle_combo.currentIndexChanged.connect(self._on_cycle_selection_changed)
        self.cycle_prev_button.clicked.connect(lambda: self._step_cycle(-1))
        self.cycle_next_button.clicked.connect(lambda: self._step_cycle(1))
        self.sweep_combo.currentIndexChanged.connect(self._on_sweep_changed)
        self.prominence_edit.textEdited.connect(self._on_detection_param_edited)
        self.min_sep_spin.valueChanged.connect(self._on_detection_param_edited)
        self.find_peaks_button.clicked.connect(self._on_find_peaks_clicked)
        self.add_peak_button.toggled.connect(self._set_manual_peak_mode)
        self.remove_peak_button.clicked.connect(self._on_remove_selected)
        self.toggle_enabled_button.clicked.connect(self._on_toggle_enabled)

        self._on_cycle_source_changed()
        self.refresh()

    # --- construction helper ---------------------------------------------

    @staticmethod
    def _optional_spin(suffix: str, *, decimals: int, maximum: float = 1e9) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(0.0, maximum)
        spin.setSuffix(suffix)
        spin.setSpecialValueText("(not set)")
        return spin

    # --- wiring from AnalysisPanel / MainWindow -------------------------

    def set_figure(self, figure: GnoviFigure) -> None:
        self._figure = figure
        self._invalidate_transient()
        self.refresh()

    def set_manager(self, dataset_manager: DatasetManager) -> None:
        self._manager = dataset_manager

    def _invalidate_transient(self) -> None:
        """Drop every piece of state that is only meaningful against the
        source series / cycle it was computed for: the armed Add-Peak mode,
        the working ``_current_result`` pointer (the persisted Analysis
        History entry it references is NEVER touched -- it stays selectable
        from the History list), the Results-table row selection carried
        into this sidebar, and -- via the ``overlay_changed`` its callers
        emit -- the live graph overlay. Called on a source-series change,
        a Workbench/figure switch, and a project open."""
        self._set_manual_peak_mode(False)
        self._current_result = None
        self._results_selected_rows = []

    def refresh(self) -> None:
        """Rebuild the source list, recompute cycles, and enable/disable the
        whole section (a clear message for a ``Panel3D`` active panel or no
        eligible 2D series). Called by ``MainWindow`` after any figure-
        content change or panel switch."""
        is_panel3d = isinstance(self._figure.active_panel, Panel3D)
        eligible = eligible_cv_series(self._figure)

        previous_id = self.source_combo.currentData()
        # Rebuild AND re-select under blocked signals -- `setCurrentIndex`
        # here must not fire `_on_source_changed` (that handler is for a
        # genuine USER combo pick only); a silent resolution change is
        # handled explicitly just below.
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        target = -1
        for i, series in enumerate(eligible):
            self.source_combo.addItem(series.label, series.id)
            if series.id == previous_id:
                target = i
        if target >= 0:
            self.source_combo.setCurrentIndex(target)
        elif eligible:
            self.source_combo.setCurrentIndex(0)
        self.source_combo.blockSignals(False)

        # A silently-resolved source change (the previously-selected series
        # was removed, so `refresh()` landed on a different one) must drop
        # the same transient state an explicit `_on_source_changed` does --
        # otherwise the old result's markers could linger over the new
        # series. `previous_id is not None` so the very first population
        # (empty combo -> first series) is not treated as a "change".
        if previous_id is not None and self.source_combo.currentData() != previous_id:
            self._detection_defaults_touched = False
            self._invalidate_transient()

        has_eligible = bool(eligible)
        enabled = has_eligible and not is_panel3d
        for w in (
            self.source_combo, self.sign_combo, self.scan_rate_spin, self.scan_rate_unit_combo,
            self.cycle_source_combo, self.cycle_combo, self.cycle_prev_button, self.cycle_next_button,
            self.sweep_combo, self.prominence_edit, self.min_sep_spin,
            self.find_peaks_button, self.add_peak_button, self.remove_peak_button,
            self.toggle_enabled_button, self.set_process_button,
        ):
            w.setEnabled(enabled)

        if is_panel3d:
            self.status_label.setText(_PANEL3D_TEXT)
            self.status_label.setVisible(True)
        elif not has_eligible:
            self.status_label.setText(_NO_SOURCE_TEXT)
            self.status_label.setVisible(True)
        else:
            self.status_label.setVisible(False)

        self._refresh_columns_label()
        self._refresh_metadata_columns()
        self._rebuild_cycles()
        self._maybe_apply_default_prominence()

    def load_result(self, result: AnalysisResult | None) -> None:
        """Restore ``result`` (if a ``CVCycleAnalysisResult``) as the working
        source series / sign convention / cycle / sweep / detection
        settings, WITHOUT rerunning detection. Called on an Analysis
        History selection."""
        self._current_result = result if isinstance(result, CVCycleAnalysisResult) else None
        self._results_selected_rows = []
        if self._current_result is not None:
            r = self._current_result
            # Point the source combo back at the series this result was
            # computed on (if it is still an eligible series in the active
            # panel) -- so the overlay's series-id gate matches and the
            # sweep/cycle readouts describe the right data. Blocked so it
            # does not re-enter `_on_source_changed` and wipe the result we
            # are in the middle of restoring.
            if r.source_series_id is not None:
                s_idx = self.source_combo.findData(r.source_series_id)
                if s_idx >= 0 and s_idx != self.source_combo.currentIndex():
                    self.source_combo.blockSignals(True)
                    self.source_combo.setCurrentIndex(s_idx)
                    self.source_combo.blockSignals(False)
                    self._refresh_columns_label()
                    self._refresh_metadata_columns()
                    self._rebuild_cycles()
            idx = self.sign_combo.findData(r.sign_convention)
            if idx >= 0:
                self.sign_combo.blockSignals(True)
                self.sign_combo.setCurrentIndex(idx)
                self.sign_combo.blockSignals(False)
            params = r.parameters or {}
            sweep_idx = self.sweep_combo.findData(params.get("sweep_selection", _SWEEP_BOTH))
            if sweep_idx >= 0:
                self.sweep_combo.blockSignals(True)
                self.sweep_combo.setCurrentIndex(sweep_idx)
                self.sweep_combo.blockSignals(False)
            self._restore_detection_params(params.get("detection", {}) or {})
            if r.cycle_index is not None:
                pos = self.cycle_combo.findData(r.cycle_index)
                if pos >= 0:
                    self.cycle_combo.blockSignals(True)
                    self.cycle_combo.setCurrentIndex(pos)
                    self.cycle_combo.blockSignals(False)
        self._refresh_sweep_readout()
        self.overlay_changed.emit()

    def _restore_detection_params(self, detection: dict) -> None:
        prominence = detection.get("prominence")
        self.prominence_edit.setText("" if prominence is None else f"{prominence:.4g}")
        self.min_sep_spin.blockSignals(True)
        try:
            if detection.get("min_separation_mv") is not None:
                self.min_sep_spin.setValue(detection["min_separation_mv"])
        finally:
            self.min_sep_spin.blockSignals(False)
        width = detection.get("width_mv")
        self.width_check.setChecked(width is not None)
        if width is not None:
            self.width_spin.setValue(width)

    # --- state accessors used by AnalysisPanel/MainWindow -----------------

    def current_result(self) -> CVCycleAnalysisResult | None:
        return self._current_result

    def is_manual_peak_mode(self) -> bool:
        return self._manual_peak_mode

    def set_selected_peak_rows(self, rows: list[int]) -> None:
        self._results_selected_rows = sorted({int(r) for r in rows})

    def disarm_manual_peak_mode(self) -> None:
        self._set_manual_peak_mode(False)

    def overlay_payload(self) -> dict | None:
        """The transient CV overlay for the active panel, or ``None`` when
        there is nothing to draw / the current result belongs to another
        panel. Rebuilt fresh each call from the live source data + the
        current result -- never cached, never a ``PlotSeries``."""
        series = self._source_series()
        if series is None:
            return None
        xy = self._raw_xy()
        if xy is None:
            return None
        e, i = xy
        payload: dict = {}

        cycle = self._selected_cycle()
        if cycle is not None:
            rising = next((s for s in cycle.sweeps if s.direction == SWEEP_RISING), None)
            falling = next((s for s in cycle.sweeps if s.direction == SWEEP_FALLING), None)
            if rising is not None:
                payload["cycle_rising_xy"] = (e[rising.start:rising.end], i[rising.start:rising.end])
            if falling is not None:
                payload["cycle_falling_xy"] = (e[falling.start:falling.end], i[falling.start:falling.end])
            payload["switching_potential_v"] = self._switching_potential(cycle)

        result = self._current_result
        # Peak markers only when the current result actually belongs to
        # what is on screen now: the active panel, the SELECTED SOURCE
        # SERIES (not just the panel -- a panel can hold several series),
        # and the selected cycle. Any mismatch -> cycle tint only, so a
        # result computed on series A never floats its markers over
        # series B.
        if (
            result is not None
            and result.source_panel_id == self._figure.active_panel.id
            and (result.source_series_id is None or result.source_series_id == series.id)
            and (cycle is None or result.cycle_index == cycle.index)
        ):
            cand_x = [p.e_peak_v for p in result.peaks]
            cand_y = [p.i_peak_raw_a for p in result.peaks]
            payload["candidate_xy"] = (cand_x, cand_y)
            payload["anodic_xy"] = self._process_points(result, PROCESS_ANODIC)
            payload["cathodic_xy"] = self._process_points(result, PROCESS_CATHODIC)
        return payload or None

    @staticmethod
    def _process_points(result: CVCycleAnalysisResult, process: str) -> tuple[list[float], list[float]]:
        pts = [(p.e_peak_v, p.i_peak_raw_a) for p in result.peaks if p.enabled and p.process == process]
        return [p[0] for p in pts], [p[1] for p in pts]

    @staticmethod
    def _switching_potential(cycle: Cycle) -> float | None:
        rising = next((s for s in cycle.sweeps if s.direction == SWEEP_RISING), None)
        falling = next((s for s in cycle.sweeps if s.direction == SWEEP_FALLING), None)
        if rising is not None and falling is not None:
            # the shared vertex -- the end of whichever sweep comes first
            first = rising if rising.start <= falling.start else falling
            return first.e_end
        return None

    # --- source / sign convention -------------------------------------

    def _source_series(self) -> PlotSeries | None:
        series_id = self.source_combo.currentData()
        if series_id is None:
            return None
        return self._figure.get_series(series_id)

    def _raw_xy(self) -> tuple[np.ndarray, np.ndarray] | None:
        series = self._source_series()
        if series is None:
            return None
        try:
            x, y = numeric_xy(series.dataframe, series.x_column, series.y_column)
        except (KeyError, InsufficientNumericDataError):
            return None
        return x.to_numpy(), y.to_numpy()

    def _sign_convention(self) -> CurrentSignConvention:
        return CurrentSignConvention(self.sign_combo.currentData())

    def _refresh_columns_label(self) -> None:
        series = self._source_series()
        if series is None:
            self.columns_label.setText("")
            self.columns_label.setVisible(False)
            return
        self.columns_label.setText(
            f"Potential: {series.x_column} · Current: {series.y_column}"
        )
        self.columns_label.setVisible(True)

    def _on_source_changed(self) -> None:
        """A genuine USER pick of a different source series. Detach the
        working result (its History entry is untouched and stays
        selectable), drop all transient state, reset the data-dependent
        detection default for the new series, and clear the overlay."""
        self._invalidate_transient()
        self._detection_defaults_touched = False
        self._refresh_columns_label()
        self._refresh_metadata_columns()
        self._rebuild_cycles()
        self._maybe_apply_default_prominence()
        self.overlay_changed.emit()

    def _on_sign_convention_changed(self) -> None:
        """Reinterpret the CURRENT result under the new convention -- flip
        every definite anodic/cathodic tag, recompute the couple, and emit
        an in-place update. Never a new history entry, never touches the
        recorded current."""
        result = self._current_result
        series = self._source_series()
        if result is None or series is None or (
            result.source_series_id is not None and result.source_series_id != series.id
        ):
            # No result, or one that no longer belongs to the shown series
            # (see `_invalidate_transient`) -- just refresh the overlay.
            self.overlay_changed.emit()
            return
        result.sign_convention = self._sign_convention().value
        for peak in result.peaks:
            peak.process = _PROCESS_FLIP.get(peak.process, peak.process)
        self._recompute_couple(result)
        self.result_updated.emit(result)
        self.overlay_changed.emit()

    def _electrode_context(self) -> ElectrodeContext:
        def opt(spin: QDoubleSpinBox) -> float | None:
            return spin.value() if spin.value() > 0 else None

        conc_mm = opt(self.conc_spin)
        try:
            return ElectrodeContext(
                area_cm2=opt(self.area_spin),
                n=opt(self.n_spin),
                # entered in mM (mmol/L); 1 mM = 1e-6 mol/cm³
                concentration_mol_cm3=(conc_mm * 1e-6 if conc_mm is not None else None),
                temperature_k=opt(self.temp_spin),
                reference_electrode=self.ref_edit.text().strip() or None,
                working_electrode=self.we_edit.text().strip() or None,
                counter_electrode=self.ce_edit.text().strip() or None,
                electrolyte=self.electrolyte_edit.text().strip() or None,
            )
        except InvalidElectrodeContextError:
            return ElectrodeContext()

    def _scan_rate_v_per_s(self) -> float | None:
        if self.scan_rate_spin.value() <= 0:
            return None
        return scan_rate_to_v_per_s(self.scan_rate_spin.value(), self.scan_rate_unit_combo.currentText())

    # --- cycle selection ---------------------------------------------

    def _cycle_source(self) -> str:
        return self.cycle_source_combo.currentData()

    def _on_cycle_source_changed(self) -> None:
        source = self._cycle_source()
        self.metadata_column_combo.setVisible(source == _CYCLE_SOURCE_METADATA)
        self.manual_ranges_edit.setVisible(source == _CYCLE_SOURCE_MANUAL)
        self._on_cycle_inputs_changed()

    def _on_cycle_inputs_changed(self) -> None:
        self._rebuild_cycles()
        self.overlay_changed.emit()

    def _refresh_metadata_columns(self) -> None:
        series = self._source_series()
        self.metadata_column_combo.blockSignals(True)
        self.metadata_column_combo.clear()
        if series is not None:
            for col in series.dataframe.columns:
                if col not in (series.x_column, series.y_column):
                    self.metadata_column_combo.addItem(str(col))
        self.metadata_column_combo.blockSignals(False)

    def _sweeps_for_range(self, e: np.ndarray, start: int, end: int) -> tuple[SweepSegment, ...]:
        """Segment the slice ``e[start:end]`` and offset the segment
        positions back into the full array. A slice that does not reverse
        (LSV-like) yields a single sweep; a slice too short/flat yields an
        empty tuple."""
        try:
            local = segment_sweeps(e[start:end])
        except SweepSegmentationError:
            return ()
        return tuple(
            SweepSegment(
                start=start + s.start,
                end=start + s.end,
                direction=s.direction,
                e_start=s.e_start,
                e_end=s.e_end,
            )
            for s in local
        )

    def _rebuild_cycles(self) -> None:
        """Recompute ``self._cycles`` and repopulate the cycle picker from
        whichever source the researcher selected. Never mutates any result;
        purely the transient "what cycles does this data have" state."""
        self._cycles = []
        confidence = CYCLE_CONFIDENCE_DETECTED
        status = ""
        xy = self._raw_xy()
        if xy is None:
            self.cycle_status_label.setText("")
            self._repopulate_cycle_combo()
            self.cycle_confidence_label.setText("")
            self._refresh_sweep_readout()
            return
        e, _i = xy
        source = self._cycle_source()

        if source == _CYCLE_SOURCE_MANUAL:
            confidence = CYCLE_CONFIDENCE_MANUAL
            ranges = self._parse_manual_ranges(len(e))
            for k, (start, end) in enumerate(ranges, start=1):
                sweeps = self._sweeps_for_range(e, start, end)
                self._cycles.append(
                    Cycle(index=k, start=start, end=end, sweeps=sweeps,
                          complete={SWEEP_RISING, SWEEP_FALLING} <= {s.direction for s in sweeps})
                )
            status = f"{len(self._cycles)} manual range(s)." if self._cycles else \
                "Enter one or more row ranges, e.g. 0-1600, 1600-3200."
        elif source == _CYCLE_SOURCE_METADATA:
            confidence = CYCLE_CONFIDENCE_EXPLICIT
            col = self.metadata_column_combo.currentText()
            ranges = self._ranges_from_metadata(col)
            for k, (start, end) in enumerate(ranges, start=1):
                sweeps = self._sweeps_for_range(e, start, end)
                self._cycles.append(
                    Cycle(index=k, start=start, end=end, sweeps=sweeps,
                          complete={SWEEP_RISING, SWEEP_FALLING} <= {s.direction for s in sweeps})
                )
            status = (
                f"{len(self._cycles)} cycle(s) from column '{col}'."
                if self._cycles else f"Column '{col}' did not yield usable cycle groups."
            )
        else:  # auto-detect
            try:
                sweeps = segment_sweeps(e)
            except SweepSegmentationError as exc:
                self.cycle_status_label.setText(str(exc))
                self._repopulate_cycle_combo()
                self.cycle_confidence_label.setText("Cycle source: detected")
                self._refresh_sweep_readout()
                return
            self._cycles = list(pair_cycles(sweeps))
            n_complete = sum(1 for c in self._cycles if c.complete)
            n_sweeps = len(sweeps)
            if len(self._cycles) == 1 and not self._cycles[0].complete and n_sweeps == 1:
                status = (
                    "Single sweep (no reversal). Per-peak Epa/Ipa is available; "
                    "ΔEp / E½ / ratio need a full couple and are not computed."
                )
            elif ambiguous_segmentation(sweeps):
                status = (
                    f"{n_sweeps} sweeps → {len(self._cycles)} cycle(s) — segmentation looks "
                    "ambiguous. Check the overlay or switch to Manual."
                )
            else:
                status = f"{n_sweeps} sweeps → {len(self._cycles)} cycle(s), {n_complete} complete."

        self.cycle_status_label.setText(status)
        self.cycle_confidence_label.setText(f"Cycle source: {confidence}")
        self._repopulate_cycle_combo()
        self._refresh_sweep_readout()

    def _parse_manual_ranges(self, n_rows: int) -> list[tuple[int, int]]:
        text = self.manual_ranges_edit.text().strip()
        if not text:
            return []
        collection = RowRangeCollection(n_rows)
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk or "-" not in chunk:
                continue
            a, _, b = chunk.partition("-")
            try:
                start, end = int(a), int(b)
                collection.add(start, end)
            except (ValueError, InvalidRowRangeError, OverlappingRowRangeError):
                continue
        return collection.ranges

    def _valid_row_mask(self) -> np.ndarray | None:
        """Boolean mask over ``source_series.dataframe`` rows that survive
        ``numeric_xy``'s NaN filter -- i.e. the rows present in ``_raw_xy``,
        in the same order. Everything downstream (sweep segmentation, peak
        indices, the overlay) works in this CLEANED positional space, so a
        metadata cycle column must be mapped through this mask too."""
        series = self._source_series()
        if series is None:
            return None
        df = series.dataframe
        try:
            x = pd.to_numeric(df[series.x_column], errors="coerce")
            y = pd.to_numeric(df[series.y_column], errors="coerce")
        except KeyError:
            return None
        return (x.notna() & y.notna()).to_numpy()

    def _ranges_from_metadata(self, column: str) -> list[tuple[int, int]]:
        """Contiguous ``(start, end)`` runs of equal value in ``column``,
        in the CLEANED-row positional space (aligned with ``_raw_xy`` --
        NaN rows in the potential/current columns are removed first, so the
        ranges index the same array ``_sweeps_for_range`` slices)."""
        series = self._source_series()
        mask = self._valid_row_mask()
        if series is None or not column or mask is None:
            return []
        df = series.dataframe
        if column not in df.columns:
            return []
        values = df[column].to_numpy()[mask]
        ranges: list[tuple[int, int]] = []
        if len(values) == 0:
            return ranges
        start = 0
        for pos in range(1, len(values)):
            if values[pos] != values[pos - 1]:
                ranges.append((start, pos))
                start = pos
        ranges.append((start, len(values)))
        return [r for r in ranges if r[1] - r[0] >= 2]

    def _repopulate_cycle_combo(self) -> None:
        previous = self.cycle_combo.currentData()
        self.cycle_combo.blockSignals(True)
        self.cycle_combo.clear()
        for cycle in self._cycles:
            suffix = "" if cycle.complete else " (incomplete)"
            self.cycle_combo.addItem(f"Cycle {cycle.index}{suffix}", cycle.index)
        # default = last complete, else last
        target = -1
        if previous is not None:
            target = self.cycle_combo.findData(previous)
        if target < 0:
            complete = [i for i, c in enumerate(self._cycles) if c.complete]
            target = complete[-1] if complete else (len(self._cycles) - 1)
        if 0 <= target < self.cycle_combo.count():
            self.cycle_combo.setCurrentIndex(target)
        self.cycle_combo.blockSignals(False)
        self._update_cycle_stepper_state()

    def _update_cycle_stepper_state(self) -> None:
        idx = self.cycle_combo.currentIndex()
        count = self.cycle_combo.count()
        self.cycle_prev_button.setEnabled(idx > 0)
        self.cycle_next_button.setEnabled(0 <= idx < count - 1)

    def _step_cycle(self, delta: int) -> None:
        self.cycle_combo.setCurrentIndex(
            max(0, min(self.cycle_combo.count() - 1, self.cycle_combo.currentIndex() + delta))
        )

    def _selected_cycle(self) -> Cycle | None:
        index = self.cycle_combo.currentData()
        return next((c for c in self._cycles if c.index == index), None)

    def _on_cycle_selection_changed(self) -> None:
        # re-arm detection: no new history entry, just refresh context.
        # Different cycles of one run share the same noise scale, so a
        # data-dependent prominence default is only re-applied when the
        # researcher hasn't set their own (the `_maybe_apply_*` guard) --
        # a deliberate value is never wiped by a cycle-picker click.
        self.disarm_manual_peak_mode()
        self._update_cycle_stepper_state()
        self._maybe_apply_default_prominence()
        self._refresh_sweep_readout()
        self.overlay_changed.emit()

    # --- sweep selection --------------------------------------------

    def _selected_sweeps(self) -> list[SweepSegment]:
        cycle = self._selected_cycle()
        if cycle is None:
            return []
        choice = self.sweep_combo.currentData()
        if choice == _SWEEP_RISING_ONLY:
            return [s for s in cycle.sweeps if s.direction == SWEEP_RISING]
        if choice == _SWEEP_FALLING_ONLY:
            return [s for s in cycle.sweeps if s.direction == SWEEP_FALLING]
        return list(cycle.sweeps)

    def _on_sweep_changed(self) -> None:
        self.disarm_manual_peak_mode()
        self._refresh_sweep_readout()
        self.overlay_changed.emit()

    def _refresh_sweep_readout(self) -> None:
        cycle = self._selected_cycle()
        if cycle is None or not cycle.sweeps:
            self.sweep_readout_label.setText("")
            return
        parts = []
        for s in cycle.sweeps:
            parts.append(
                f"{s.direction.capitalize()}: rows {s.start}–{s.end - 1} "
                f"({s.e_start:.3g} → {s.e_end:.3g} V)"
            )
        self.sweep_readout_label.setText(" · ".join(parts))

    # --- detection defaults ----------------------------------------

    def _on_detection_param_edited(self, *_args) -> None:
        self._detection_defaults_touched = True

    def _maybe_apply_default_prominence(self) -> None:
        if self._detection_defaults_touched:
            return
        xy = self._raw_xy()
        sweeps = self._selected_sweeps()
        if xy is None or not sweeps:
            return
        _e, i = xy
        # from the selected sweep(s) only -- concatenated per sweep
        segments = [i[s.start:s.end] for s in sweeps]
        current = np.concatenate(segments) if segments else i
        value = default_prominence(current)
        self.prominence_edit.setText(f"{value:.4g}" if value > 0 else "")

    # --- Find Peaks -----------------------------------------------

    def _prominence_value(self) -> float | None:
        text = self.prominence_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _detection_kwargs(self, potential: np.ndarray) -> dict:
        distance = mv_to_sample_distance(potential, self.min_sep_spin.value())
        width = None
        if self.width_check.isChecked() and self.width_spin.value() > 0:
            samples = mv_to_sample_distance(potential, self.width_spin.value())
            width = float(samples) if samples is not None else None
        # `prominence=0.0` (not None) when the field is blank: it filters
        # nothing (every local maximum has prominence >= 0) but makes SciPy
        # COMPUTE a prominence for every automatic candidate, so couple
        # selection can always rank them by prominence rather than falling
        # back to detection order (see `results.assign_couple`).
        return {"prominence": self._prominence_value() or 0.0, "distance": distance, "width": width}

    def _on_find_peaks_clicked(self) -> None:
        series = self._source_series()
        xy = self._raw_xy()
        if series is None or xy is None:
            QMessageBox.warning(self, "Cyclic Voltammetry", "Select a plotted 2D series first.")
            return
        e, i = xy
        sweeps = self._selected_sweeps()
        cycle = self._selected_cycle()
        if not sweeps:
            QMessageBox.warning(
                self, "Cyclic Voltammetry",
                "No sweeps to analyse in the selected cycle / sweep choice.",
            )
            return

        convention = self._sign_convention()
        detection = self._detection_kwargs(e)
        seeds: list[CVPeakSeed] = []
        try:
            for sweep in sweeps:
                seeds.extend(detect_cv_peaks(e, i, sweep, convention=convention, **detection))
        except InvalidCVInputError as exc:
            QMessageBox.critical(self, "Cyclic Voltammetry", str(exc))
            return

        peaks = [peak_result_from_seed(s) for s in seeds]
        n_an = sum(1 for p in peaks if p.process == PROCESS_ANODIC)
        n_ca = sum(1 for p in peaks if p.process == PROCESS_CATHODIC)
        self.detection_status_label.setText(
            f"{len(peaks)} candidate(s) ({n_an} anodic, {n_ca} cathodic)."
        )

        metrics, anodic_id, cathodic_id = couple_from_peak_results(peaks)
        electrode = self._electrode_context()
        parameters = {
            "sweep_selection": self.sweep_combo.currentData(),
            "detection": {
                # the semantic value (None when the researcher left the
                # field blank), not the 0.0 passed to find_peaks
                "prominence": self._prominence_value(),
                "min_separation_mv": self.min_sep_spin.value() or None,
                "width_mv": self.width_spin.value() if self.width_check.isChecked() else None,
            },
            "scan_rate_v_per_s": self._scan_rate_v_per_s(),
            "scan_rate_unit": self.scan_rate_unit_combo.currentText(),
            "electrode_context": None if electrode.is_empty() else electrode.to_dict(),
        }
        if self._cycle_source() == _CYCLE_SOURCE_MANUAL:
            parameters["manual_cycle_ranges"] = [list(r) for r in self._parse_manual_ranges(len(e))]

        result = build_cv_cycle_analysis_result(
            source_dataset_id=series.dataset.id,
            x_column=series.x_column,
            y_column=series.y_column,
            sign_convention=convention.value,
            sweeps=list(cycle.sweeps) if cycle is not None else sweeps,
            peaks=peaks,
            cycle_index=cycle.index if cycle is not None else None,
            cycle_confidence=self._current_confidence(),
            cycle_complete=cycle.complete if cycle is not None else True,
            couple=metrics,
            couple_anodic_peak_id=anodic_id,
            couple_cathodic_peak_id=cathodic_id,
            source_dataset_name=series.dataset.name,
            source_series_id=series.id,
            source_series_label=series.label,
            row_range=series.row_range,
            source_panel_id=self._figure.active_panel.id,
            parameters=parameters,
        )
        self._current_result = result
        self._results_selected_rows = []
        self.overlay_changed.emit()
        self.analysis_result_ready.emit(result)

    def _current_confidence(self) -> str:
        source = self._cycle_source()
        if source == _CYCLE_SOURCE_MANUAL:
            return CYCLE_CONFIDENCE_MANUAL
        if source == _CYCLE_SOURCE_METADATA:
            return CYCLE_CONFIDENCE_EXPLICIT
        return CYCLE_CONFIDENCE_DETECTED

    # --- manual peak editing --------------------------------------

    def _set_manual_peak_mode(self, active: bool) -> None:
        active = bool(active)
        if active == self._manual_peak_mode and self.add_peak_button.isChecked() == active:
            return
        self._manual_peak_mode = active
        self.add_peak_button.blockSignals(True)
        self.add_peak_button.setChecked(active)
        self.add_peak_button.blockSignals(False)
        if active:
            self.status_message.emit("Click a peak location inside the selected cycle.")
        self.manual_peak_mode_changed.emit(active)

    #: A click whose nearest point on the selected cycle's trace is farther
    #: than this (as a fraction of the cycle's own E×I bounding-box
    #: diagonal, in normalised coordinates) is rejected -- it is not on the
    #: selected cycle's curve. Generous enough to tolerate an imprecise but
    #: clearly-intended click, far below the distance to empty space or to
    #: a wildly wrong current.
    _MANUAL_CLICK_MAX_NORM_DISTANCE = 0.12

    def add_manual_peak(self, potential_v: float, current_a: float) -> None:
        """Called by ``MainWindow`` after a canvas click while Add Peak is
        armed (the wrong-panel / Panel3D / non-finite guards already
        passed in ``MainWindow._handle_cv_manual_peak_click``).

        Finds the closest point on the SELECTED CYCLE's trace to the click,
        in (potential, current) space normalised by the cycle's own
        extent -- so the correct sweep is chosen by proximity to the
        actual curve (never "the other sweep just because it shares this
        potential"), and a click that is not on the selected cycle's curve
        at all (empty space, a different cycle far away, a wildly wrong
        current) is REJECTED with a status message and adds nothing.
        """
        self._set_manual_peak_mode(False)
        series = self._source_series()
        xy = self._raw_xy()
        cycle = self._selected_cycle()
        if series is None or xy is None or cycle is None:
            QMessageBox.warning(self, "Cyclic Voltammetry", "Select a source series and a cycle first.")
            return
        e, i = xy
        if not cycle.sweeps:
            self.status_message.emit("The selected cycle has no sweep to place a peak on.")
            return

        cycle_e = e[cycle.start:cycle.end]
        cycle_i = i[cycle.start:cycle.end]
        e_scale = float(np.ptp(cycle_e)) or 1.0
        i_scale = float(np.ptp(cycle_i)) or 1.0

        best = None
        best_d = np.inf
        for s in cycle.sweeps:
            seg_e = e[s.start:s.end]
            seg_i = i[s.start:s.end]
            d = np.hypot((seg_e - potential_v) / e_scale, (seg_i - current_a) / i_scale)
            local = int(np.argmin(d))
            if d[local] < best_d:
                best_d = float(d[local])
                best = (s, s.start + local)

        if best is None or best_d > self._MANUAL_CLICK_MAX_NORM_DISTANCE:
            self.status_message.emit(
                "Click on the selected cycle's curve to add a candidate peak."
            )
            return

        sweep, idx = best
        snapped_e = float(e[idx])
        snapped_i = float(i[idx])
        median_i = float(np.median(i[sweep.start:sweep.end]))
        offset = snapped_i - median_i
        # ambiguous when the click sits essentially on the sweep's own
        # median line -- no clear oxidative/reductive direction
        if abs(offset) < 0.02 * i_scale:
            process = PROCESS_UNASSIGNED
        elif offset * oxidative_sign(self._sign_convention()) > 0:
            process = PROCESS_ANODIC
        else:
            process = PROCESS_CATHODIC
        seed = CVPeakSeed.manual(snapped_e, snapped_i, sweep=sweep.direction, process=process)

        current = self._current_result
        matches_here = (
            current is not None
            and current.source_panel_id == self._figure.active_panel.id
            and (current.source_series_id is None or current.source_series_id == series.id)
            and (current.cycle_index is None or current.cycle_index == cycle.index)
        )
        if not matches_here:
            self._start_result_with_manual_peak(series, cycle, seed)
            return
        current.peaks.append(peak_result_from_seed(seed))
        self._recompute_couple(current)
        self.overlay_changed.emit()
        self.result_updated.emit(current)

    def _start_result_with_manual_peak(self, series: PlotSeries, cycle: Cycle, seed: CVPeakSeed) -> None:
        peaks = [peak_result_from_seed(seed)]
        metrics, anodic_id, cathodic_id = couple_from_peak_results(peaks)
        electrode = self._electrode_context()
        result = build_cv_cycle_analysis_result(
            source_dataset_id=series.dataset.id,
            x_column=series.x_column,
            y_column=series.y_column,
            sign_convention=self._sign_convention().value,
            sweeps=list(cycle.sweeps),
            peaks=peaks,
            cycle_index=cycle.index,
            cycle_confidence=self._current_confidence(),
            cycle_complete=cycle.complete,
            couple=metrics,
            couple_anodic_peak_id=anodic_id,
            couple_cathodic_peak_id=cathodic_id,
            source_dataset_name=series.dataset.name,
            source_series_id=series.id,
            source_series_label=series.label,
            row_range=series.row_range,
            source_panel_id=self._figure.active_panel.id,
            parameters={
                "sweep_selection": self.sweep_combo.currentData(),
                "detection": None,
                "scan_rate_v_per_s": self._scan_rate_v_per_s(),
                "scan_rate_unit": self.scan_rate_unit_combo.currentText(),
                "electrode_context": None if electrode.is_empty() else electrode.to_dict(),
            },
        )
        self._current_result = result
        self._results_selected_rows = []
        self.overlay_changed.emit()
        self.analysis_result_ready.emit(result)

    def _selected_peak_rows(self) -> list[int]:
        if self._current_result is None:
            return []
        count = len(self._current_result.peaks)
        return [r for r in self._results_selected_rows if 0 <= r < count]

    def _on_remove_selected(self) -> None:
        if self._current_result is None:
            return
        rows = self._selected_peak_rows()
        if not rows:
            return
        for row in reversed(rows):
            del self._current_result.peaks[row]
        self._results_selected_rows = []
        self._recompute_couple(self._current_result)
        self.overlay_changed.emit()
        self.result_updated.emit(self._current_result)

    def _on_toggle_enabled(self) -> None:
        if self._current_result is None:
            return
        rows = self._selected_peak_rows()
        if not rows:
            return
        for row in rows:
            self._current_result.peaks[row].enabled = not self._current_result.peaks[row].enabled
        self._recompute_couple(self._current_result)
        self.overlay_changed.emit()
        self.result_updated.emit(self._current_result)

    def _on_set_process(self, process: str) -> None:
        if self._current_result is None:
            return
        rows = self._selected_peak_rows()
        if not rows:
            return
        for row in rows:
            self._current_result.peaks[row].process = process
        self._recompute_couple(self._current_result)
        self.overlay_changed.emit()
        self.result_updated.emit(self._current_result)

    def _recompute_couple(self, result: CVCycleAnalysisResult) -> None:
        metrics, anodic_id, cathodic_id = couple_from_peak_results(result.peaks)
        result.couple_anodic_peak_id = anodic_id
        result.couple_cathodic_peak_id = cathodic_id
        if metrics is not None:
            result.delta_ep_v = metrics.delta_ep_v
            result.e_half_v = metrics.e_half_v
            result.peak_current_ratio_ipa_over_ipc = metrics.ratio_ipa_over_ipc
            result.peak_current_ratio_ipc_over_ipa = metrics.ratio_ipc_over_ipa
            result.peak_current_ratio_basis = metrics.ratio_basis
        else:
            result.delta_ep_v = None
            result.e_half_v = None
            result.peak_current_ratio_ipa_over_ipc = None
            result.peak_current_ratio_ipc_over_ipa = None
            result.peak_current_ratio_basis = None
