from __future__ import annotations

import csv

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.core.app_info import __version__ as _APP_VERSION
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.modules.xrd.bragg import InvalidBraggInputError, d_spacing
from gnovi_plot.modules.xrd.peaks import (
    ORIGIN_AUTOMATIC,
    InvalidPeakDetectionError,
    XRDPeakSeed,
    detect_peaks,
)
from gnovi_plot.modules.xrd.preprocessing import (
    InvalidPreprocessingError,
    PybaselinesNotAvailableError,
    arpls_baseline,
    polynomial_baseline,
    savgol_smooth,
)
from gnovi_plot.modules.xrd.radiation import (
    RADIATION_PRESETS,
    InvalidRadiationError,
    Radiation,
)
from gnovi_plot.modules.xrd.results import (
    PEAK_TABLE_COLUMNS,
    XRDAnalysisResult,
    build_xrd_analysis_result,
)
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries

_NO_SOURCE_TEXT = (
    "No plotted line/scatter series in the active panel yet -- add one "
    "from the 2D page first."
)
_PANEL3D_TEXT = (
    "XRD Peak Analysis works on a 2D Panel's plotted series -- switch to "
    "(or add) a 2D panel first."
)
_NO_RADIATION_TEXT = "Radiation/wavelength is required for d-spacing."

_INPUT_RAW = "raw"
_INPUT_BACKGROUND = "background_corrected"
_INPUT_SMOOTH_RAW = "smoothed_raw"
_INPUT_SMOOTH_BACKGROUND = "smoothed_background_corrected"
_INPUT_LABELS = {
    _INPUT_RAW: "Raw",
    _INPUT_BACKGROUND: "Background-corrected",
    _INPUT_SMOOTH_RAW: "Smoothed raw",
    _INPUT_SMOOTH_BACKGROUND: "Smoothed background-corrected",
}

_BACKGROUND_NONE = "None"
_BACKGROUND_ARPLS = "arPLS"
_BACKGROUND_POLYNOMIAL = "Polynomial"

_LABEL_MODE_OFF = "Off"
_LABEL_MODE_NUMBER = "Peak number"
_LABEL_MODE_TWO_THETA = "2θ"
_LABEL_MODE_D_SPACING = "d-spacing"

# A first-run Prominence of 0 is passed to `detect_peaks` as `None` (no
# threshold at all -- see `_on_find_peaks_clicked`), which on real noisy
# data returns essentially every local maximum, including noise
# fluctuations, as a "peak" (one real run produced 1,118 candidates from
# a raw pattern with about a dozen actual peaks). This multiplier turns a
# robust estimate of the signal's own local noise scale (see
# `_default_prominence_from_signal`) into a conservative first-run
# threshold -- large enough that ordinary sample-to-sample noise rarely
# clears it, small enough that a real diffraction peak (which towers over
# noise by design) still does. A STARTING POINT only: shown in the
# Prominence field, freely editable, and never recomputed once the
# researcher has touched either detection spinbox (see
# `_detection_defaults_touched`).
_PROMINENCE_NOISE_MULTIPLIER = 5.0


def _eligible_series(figure: GnoviFigure) -> list[PlotSeries]:
    """Line/scatter series in the active panel -- empty if the active
    panel is a Panel3D (XRD never operates on 3D data, see the class
    docstring)."""
    if isinstance(figure.active_panel, Panel3D):
        return []
    return [s for s in figure.series if isinstance(s, PlotSeries) and s.y_column is not None and not s.stale]


def _default_prominence_from_signal(y: np.ndarray) -> float:
    """A conservative, transparent, data-dependent STARTING Prominence for
    `scipy.signal.find_peaks` -- see `_PROMINENCE_NOISE_MULTIPLIER`'s own
    docstring for why a first-run default of 0/"no threshold" is unusable
    on real data.

    Uses `1.4826 * median(abs(d - median(d)))` of the signal's first
    differences `d` -- the standard robust estimator of a signal's local
    noise standard deviation (the constant makes it consistent with the
    standard deviation for normally-distributed noise; using the MEDIAN
    absolute deviation rather than the plain standard deviation of the
    differences means a handful of large jumps -- real peak edges, not
    noise -- don't inflate the estimate the way `np.std` would). This is
    NOT an automatic "correct" prominence, an AI/statistical peak
    classifier, or a claim about how many real peaks exist -- it is one
    simple, reproducible number the researcher sees in the Prominence
    field and can freely override before or after running Find Peaks.

    Deliberately computed from the RAW (x, y) regardless of the currently
    selected Detection Input -- background correction/smoothing may
    improve detection quality once used, but the first-run default must
    already be usable directly on raw, unprocessed input (see this
    module's own bug-report notes: the real failure case that motivated
    this function was raw input, no background, no smoothing)."""
    if y.size < 2:
        return 0.0
    diffs = np.diff(y)
    noise_scale = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs))))
    return max(noise_scale * _PROMINENCE_NOISE_MULTIPLIER, 0.0)


def _parse_index_ranges(text: str, max_index: int) -> list[int]:
    """Parse "0-15, 180-200, 250" (row positions, end-inclusive) into a
    sorted, de-duplicated list of valid indices. Raises ValueError with a
    clear message for malformed/out-of-range input -- never silently
    drops or clamps a bad range."""
    indices: set[int] = set()
    text = text.strip()
    if not text:
        raise ValueError("Enter at least one baseline point/range (e.g. 0-15, 180-200).")
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:]:  # allow a leading '-' to not be mistaken for a range dash
            start_s, _, end_s = chunk.partition("-")
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid range '{chunk}': start must be <= end.")
        else:
            start = end = int(chunk)
        if start < 0 or end > max_index:
            raise ValueError(f"Range '{chunk}' is out of bounds for {max_index + 1} data points.")
        indices.update(range(start, end + 1))
    if not indices:
        raise ValueError("Enter at least one baseline point/range (e.g. 0-15, 180-200).")
    return sorted(indices)


class XRDAnalysisSection(QWidget):
    """XRD Peak Analysis -- the second Analysis-page tool, alongside
    Curve Fitting (see `AnalysisPanel`'s own docstring: one Analysis
    destination, one `CollapsibleSection` per tool, sharing the same
    Analysis History). Native GNOVI numerical foundation only
    (`gnovi_plot.modules.xrd`) -- no profile fitting, FWHM, Scherrer,
    phase ID, Rietveld, QPA, or any external engine (see
    PROJECT_GUIDE.md's XRD roadmap notes).

    This narrow left drawer holds only the XRD *controls* (source series,
    radiation, background, smoothing, peak detection, Find Peaks, and the
    manual peak actions: Add Peak, Remove Selected, Enable/Disable, plus
    graph-label mode and CSV export). The detailed peak table itself --
    the one authoritative row-per-candidate view -- lives in the bottom
    Results tab (`gui.widgets.analysis_result_view.AnalysisResultView`,
    fed by `XRDAnalysisResult.detail_table()`), which is wide enough for
    it and bounds/scrolls it correctly. Remove Selected / Enable-Disable
    here act on whichever rows are selected in that Results-tab table,
    pushed back in via `set_selected_peak_rows` -- one table, no hidden
    duplicate.

    Peak markers/labels shown on the graph are a LIVE-ONLY overlay,
    reconstructed each time from whichever `XRDAnalysisResult` is current
    for the active panel (see `overlay_points()`) -- never a persisted
    Panel/figure annotation, never one `PlotSeries` per marker (see this
    milestone's own scope notes: "prefer reconstructing from
    XRDAnalysisResult over duplicating render state"). This means the
    overlay is not part of the saved figure/project and does not appear
    in Export Panel/Complete Figure; only an explicitly-added derived
    corrected/smoothed curve (an ordinary `PlotSeries`) does.

    Background/smoothing previews are equally transient: `_background_
    preview`/`_smooth_preview` are plain local state, never registered as
    a `Dataset` or `PlotSeries` until the scientist explicitly clicks
    "Add Corrected/Smoothed Curve to Plot" (see `_on_add_corrected_
    clicked`/`_on_add_smoothed_clicked`, which follow the exact same
    derived-`Dataset` pattern `AnalysisPanel._on_add_fit_curve_clicked`
    already established for `FitResult`).

    "Find Peaks" always produces a BRAND NEW `XRDAnalysisResult` (a new
    Analysis History entry) -- the same convention "Run Fit" already uses
    for `FitResult`: never overwrites an earlier entry, even for an
    identical rerun. Manual add/remove/enable/disable, by contrast, edit
    `_current_result` IN PLACE (the exact object already in that panel's
    history) and emit `result_updated` -- dirty + redisplay, no new
    history entry -- since these are refinements to the same in-progress
    peak list, not a fresh analysis run.
    """

    analysis_result_ready = Signal(AnalysisResult)
    result_updated = Signal(AnalysisResult)
    add_to_plot_requested = Signal(list)  # list[PlotSeries]
    overlay_changed = Signal()
    manual_peak_mode_changed = Signal(bool)
    status_message = Signal(str)

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._manager = dataset_manager
        self._current_result: XRDAnalysisResult | None = None
        self._radiation: Radiation | None = None
        self._background_preview = None
        self._smooth_preview = None
        self._manual_peak_mode = False
        # Peak rows currently selected in the bottom Results-tab detail
        # table (the authoritative detailed peak table; it moved there out
        # of this narrow sidebar). Pushed in by `MainWindow` via
        # `AnalysisPanel.xrd_set_selected_peak_rows` whenever that table's
        # selection changes, and read by Remove Selected / Enable-Disable
        # here so those actions act on exactly what the researcher selected
        # in the Results table -- see `set_selected_peak_rows`.
        self._results_selected_rows: list[int] = []
        # See `_maybe_apply_default_detection_params`'s own docstring --
        # True once the researcher has edited Prominence/Minimum
        # separation themselves for the currently-selected source series,
        # so a freshly computed data-dependent default never silently
        # overwrites a deliberate choice.
        self._detection_defaults_touched = False

        # --- Source -----------------------------------------------------
        self.source_label = QLabel("Source series")
        self.source_combo = QComboBox()
        self.status_label = QLabel(_NO_SOURCE_TEXT)
        self.status_label.setWordWrap(True)

        # --- Radiation ----------------------------------------------------
        self.radiation_combo = QComboBox()
        self.radiation_combo.addItem("Select radiation…", None)
        for preset_id, radiation in RADIATION_PRESETS.items():
            self.radiation_combo.addItem(radiation.label, preset_id)
        self.radiation_combo.addItem("Custom…", "custom")
        self.custom_wavelength_spin = QDoubleSpinBox()
        self.custom_wavelength_spin.setDecimals(6)
        self.custom_wavelength_spin.setRange(0.000001, 100.0)
        self.custom_wavelength_spin.setValue(1.5406)
        self.custom_wavelength_spin.setSuffix(" Å")
        self.custom_wavelength_spin.setVisible(False)
        self.resolved_wavelength_label = QLabel("Resolved wavelength: —")

        radiation_group = QGroupBox("Radiation")
        radiation_layout = QVBoxLayout(radiation_group)
        radiation_layout.addWidget(self.radiation_combo)
        radiation_layout.addWidget(self.custom_wavelength_spin)
        radiation_layout.addWidget(self.resolved_wavelength_label)

        # --- Background -----------------------------------------------------
        self.background_method_combo = QComboBox()
        self.background_method_combo.addItems([_BACKGROUND_NONE, _BACKGROUND_ARPLS, _BACKGROUND_POLYNOMIAL])
        self.arpls_lam_spin = QDoubleSpinBox()
        self.arpls_lam_spin.setDecimals(0)
        self.arpls_lam_spin.setRange(1.0, 1e12)
        self.arpls_lam_spin.setValue(1e5)
        self.baseline_points_edit = QLineEdit()
        self.baseline_points_edit.setPlaceholderText("e.g. 0-15, 180-200")
        self.polynomial_degree_spin = QSpinBox()
        self.polynomial_degree_spin.setRange(0, 10)
        self.polynomial_degree_spin.setValue(2)
        self.preview_background_button = QPushButton("Preview Background")
        self.background_status_label = QLabel("")
        self.background_status_label.setWordWrap(True)
        self.add_corrected_button = QPushButton("Add Corrected Curve to Plot")
        self.add_corrected_button.setEnabled(False)

        background_group = QGroupBox("Background")
        background_layout = QVBoxLayout(background_group)
        background_layout.addWidget(QLabel("Method"))
        background_layout.addWidget(self.background_method_combo)
        background_layout.addWidget(QLabel("arPLS lambda (smoothness)"))
        background_layout.addWidget(self.arpls_lam_spin)
        background_layout.addWidget(QLabel("Baseline points/ranges (row positions)"))
        background_layout.addWidget(self.baseline_points_edit)
        background_layout.addWidget(QLabel("Polynomial degree"))
        background_layout.addWidget(self.polynomial_degree_spin)
        background_layout.addWidget(self.preview_background_button)
        background_layout.addWidget(self.background_status_label)
        background_layout.addWidget(self.add_corrected_button)

        # --- Smoothing -----------------------------------------------------
        self.smoothing_enabled_check = QCheckBox("Enable Savitzky–Golay")
        self.smoothing_window_spin = QSpinBox()
        self.smoothing_window_spin.setRange(3, 9999)
        self.smoothing_window_spin.setSingleStep(2)
        self.smoothing_window_spin.setValue(11)
        self.smoothing_order_spin = QSpinBox()
        self.smoothing_order_spin.setRange(0, 20)
        self.smoothing_order_spin.setValue(3)
        self.preview_smoothed_button = QPushButton("Preview Smoothed")
        self.smoothing_status_label = QLabel("")
        self.smoothing_status_label.setWordWrap(True)
        self.add_smoothed_button = QPushButton("Add Smoothed Curve to Plot")
        self.add_smoothed_button.setEnabled(False)

        smoothing_group = QGroupBox("Smoothing")
        smoothing_layout = QVBoxLayout(smoothing_group)
        smoothing_layout.addWidget(self.smoothing_enabled_check)
        smoothing_layout.addWidget(QLabel("Window length (odd)"))
        smoothing_layout.addWidget(self.smoothing_window_spin)
        smoothing_layout.addWidget(QLabel("Polynomial order"))
        smoothing_layout.addWidget(self.smoothing_order_spin)
        smoothing_layout.addWidget(self.preview_smoothed_button)
        smoothing_layout.addWidget(self.smoothing_status_label)
        smoothing_layout.addWidget(self.add_smoothed_button)

        # --- Detection input chain -----------------------------------------
        self.detection_input_combo = QComboBox()
        self.detection_input_label = QLabel("Peak detection input: Raw")

        # --- Peak detection -----------------------------------------------
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setDecimals(4)
        self.prominence_spin.setRange(0.0, 1e12)
        self.prominence_spin.setValue(0.0)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(0, 100000)
        self.distance_spin.setValue(0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setDecimals(4)
        self.height_spin.setRange(0.0, 1e12)
        self.height_spin.setValue(0.0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setDecimals(4)
        self.width_spin.setRange(0.0, 1e12)
        self.width_spin.setValue(0.0)
        self.height_check = QCheckBox("Use minimum height")
        self.width_check = QCheckBox("Use minimum width")
        self.find_peaks_button = QPushButton("Find Peaks")
        self.find_peaks_button.setProperty("primary", True)
        self.detection_status_label = QLabel("")
        self.detection_status_label.setWordWrap(True)

        detection_group = QGroupBox("Peak Detection")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.addWidget(self.detection_input_label)
        detection_layout.addWidget(self.detection_input_combo)
        detection_layout.addWidget(QLabel("Prominence"))
        detection_layout.addWidget(self.prominence_spin)
        detection_layout.addWidget(QLabel("Minimum separation (samples)"))
        detection_layout.addWidget(self.distance_spin)
        detection_layout.addWidget(self.height_check)
        detection_layout.addWidget(self.height_spin)
        detection_layout.addWidget(self.width_check)
        detection_layout.addWidget(self.width_spin)
        detection_layout.addWidget(self.find_peaks_button)
        detection_layout.addWidget(self.detection_status_label)

        # --- Manual peak editing -----------------------------------------
        # The detailed peak table itself lives in the bottom Results tab
        # now (`gui.widgets.analysis_result_view.AnalysisResultView`, fed by
        # `XRDAnalysisResult.detail_table()`) -- this narrow sidebar keeps
        # only the actions, which operate on whatever rows are selected
        # there (see `set_selected_peak_rows`). One authoritative table, no
        # hidden duplicate.
        self.add_peak_button = QPushButton("Add Peak (click graph)")
        self.add_peak_button.setCheckable(True)
        self.remove_peak_button = QPushButton("Remove Selected")
        self.toggle_enabled_button = QPushButton("Enable/Disable Selected")
        self.peak_actions_hint = QLabel(
            "Select peak rows in the Results tab below, then use these actions."
        )
        self.peak_actions_hint.setWordWrap(True)
        # Stacked vertically (not a wide button row) so this whole section
        # fits the same ordinary drawer width Curve Fitting uses -- the one
        # wide widget, the peak table, is in the bottom Results tab now.
        manual_row = QVBoxLayout()
        manual_row.addWidget(self.add_peak_button)
        manual_row.addWidget(self.remove_peak_button)
        manual_row.addWidget(self.toggle_enabled_button)

        # --- Labels -----------------------------------------------------
        self.label_mode_combo = QComboBox()
        self.label_mode_combo.addItems(
            [_LABEL_MODE_OFF, _LABEL_MODE_NUMBER, _LABEL_MODE_TWO_THETA, _LABEL_MODE_D_SPACING]
        )

        # --- Export -----------------------------------------------------
        self.export_table_button = QPushButton("Export Peak Table (CSV)…")

        results_group = QGroupBox("Detected Peaks")
        results_layout = QVBoxLayout(results_group)
        results_layout.addLayout(manual_row)
        results_layout.addWidget(self.peak_actions_hint)
        results_layout.addWidget(QLabel("Graph labels"))
        results_layout.addWidget(self.label_mode_combo)
        results_layout.addWidget(self.export_table_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.source_label)
        layout.addWidget(self.source_combo)
        layout.addWidget(self.status_label)
        layout.addWidget(radiation_group)
        layout.addWidget(background_group)
        layout.addWidget(smoothing_group)
        layout.addWidget(detection_group)
        layout.addWidget(results_group)
        layout.addStretch(1)

        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.radiation_combo.currentIndexChanged.connect(self._on_radiation_changed)
        self.custom_wavelength_spin.valueChanged.connect(self._on_radiation_changed)
        self.background_method_combo.currentIndexChanged.connect(self._on_background_method_changed)
        self.preview_background_button.clicked.connect(self._on_preview_background_clicked)
        self.add_corrected_button.clicked.connect(self._on_add_corrected_clicked)
        self.smoothing_enabled_check.toggled.connect(self._on_smoothing_toggled)
        self.preview_smoothed_button.clicked.connect(self._on_preview_smoothed_clicked)
        self.add_smoothed_button.clicked.connect(self._on_add_smoothed_clicked)
        self.prominence_spin.valueChanged.connect(self._on_detection_param_edited)
        self.distance_spin.valueChanged.connect(self._on_detection_param_edited)
        self.find_peaks_button.clicked.connect(self._on_find_peaks_clicked)
        self.add_peak_button.toggled.connect(self._on_add_peak_toggled)
        self.remove_peak_button.clicked.connect(self._on_remove_selected_clicked)
        self.toggle_enabled_button.clicked.connect(self._on_toggle_enabled_clicked)
        self.label_mode_combo.currentIndexChanged.connect(lambda _i: self.overlay_changed.emit())
        self.export_table_button.clicked.connect(self._on_export_clicked)
        self.detection_input_combo.currentIndexChanged.connect(
            lambda _i: self.detection_input_label.setText(
                f"Peak detection input: {self.detection_input_combo.currentText()}"
            )
        )

        self._on_background_method_changed()
        self._on_smoothing_toggled(False)
        self.refresh()

    # --- wiring from AnalysisPanel -----------------------------------------

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint at a different `GnoviFigure` -- a Workbench switch or a
        New/Open Project (both funnel through `MainWindow`'s single
        Figure-retargeting method, see its own docstring). Unconditionally
        invalidates any background/smoothing preview: it was computed
        against the OLD figure's source series, which no longer has any
        relationship to whatever `refresh()` (below) resolves as the
        selected source next -- never left for `refresh()`'s own
        same-series-id check to catch, since a coincidentally-matching id
        across two different figures/projects isn't a case worth risking
        (see `_invalidate_previews`'s own docstring for what this
        clears)."""
        self._figure = figure
        self._current_result = None
        self._results_selected_rows = []
        self._invalidate_previews()
        self._set_manual_peak_mode(False)
        self.refresh()

    def set_manager(self, dataset_manager: DatasetManager) -> None:
        self._manager = dataset_manager

    def refresh(self) -> None:
        """Rebuild the source-series list; disable everything with a clear
        explanation when the active panel is a Panel3D (see `_eligible_
        series`) or has no eligible 2D series yet.

        Called for far more than a source-series change -- e.g. active-
        panel switch (`MainWindow._on_panel_switched`) and any figure-
        content change (`_on_figure_content_changed`) -- so a background/
        smoothing preview computed against whatever series WAS selected
        must be invalidated whenever this ends up resolving a DIFFERENT
        series id as current (`target_index < 0`, below): an active-panel
        switch has no series in common with the previous panel, so this
        is what actually covers that case (`set_figure`, above, already
        covers a Workbench switch/project open unconditionally). Left
        alone when the exact same series id is still selected -- an
        unrelated refresh (e.g. editing some other panel's series style)
        must not discard an in-progress preview for no reason."""
        is_panel3d = isinstance(self._figure.active_panel, Panel3D)
        eligible = _eligible_series(self._figure)

        previous_id = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        target_index = -1
        for i, series in enumerate(eligible):
            self.source_combo.addItem(series.label, series.id)
            if series.id == previous_id:
                target_index = i
        self.source_combo.blockSignals(False)
        if target_index >= 0:
            self.source_combo.setCurrentIndex(target_index)
        elif eligible:
            self.source_combo.setCurrentIndex(0)
        if target_index < 0 and previous_id is not None:
            self._invalidate_previews()

        # `setCurrentIndex` above does NOT reliably emit `currentIndexChanged`
        # (so `_on_source_changed` does NOT reliably fire) for two real
        # cases: the very first population of an empty combo (Qt silently
        # auto-selects index 0 as items are added, before the explicit
        # `setCurrentIndex(0)` call above even runs -- which is then a
        # no-op, since the index isn't actually changing), and a
        # resolved-series change that happens to land on the same INDEX
        # (e.g. series A at index 0 is gone, eligible[0] is now series B --
        # `setCurrentIndex(0)` doesn't change the index value, so no
        # signal fires, even though the actual series did change). Both
        # would otherwise silently skip a fresh data-dependent detection
        # default for a genuinely new series -- compare the resolved
        # series id directly (not the index) and reset explicitly rather
        # than relying on the signal.
        current_source_id = self.source_combo.currentData()
        if current_source_id != previous_id:
            self._detection_defaults_touched = False
        self._maybe_apply_default_detection_params()

        has_eligible = bool(eligible)
        enabled = has_eligible and not is_panel3d
        self.source_combo.setEnabled(enabled)
        for widget in (
            self.radiation_combo,
            self.custom_wavelength_spin,
            self.background_method_combo,
            self.preview_background_button,
            self.smoothing_enabled_check,
            self.preview_smoothed_button,
            self.find_peaks_button,
            self.add_peak_button,
            self.remove_peak_button,
            self.toggle_enabled_button,
        ):
            widget.setEnabled(enabled)

        if is_panel3d:
            self.status_label.setText(_PANEL3D_TEXT)
            self.status_label.setVisible(True)
        elif not has_eligible:
            self.status_label.setText(_NO_SOURCE_TEXT)
            self.status_label.setVisible(True)
        else:
            self.status_label.setVisible(False)

    # --- state accessors used by AnalysisPanel/MainWindow -------------------

    def current_result(self) -> XRDAnalysisResult | None:
        return self._current_result

    def is_manual_peak_mode(self) -> bool:
        return self._manual_peak_mode

    def load_result(self, result: AnalysisResult | None) -> None:
        """Called when the shared Analysis History selection changes --
        restores `result` (if it's an XRDAnalysisResult) as the working
        radiation/detection settings/source selection, without rerunning
        detection. Never called for a FitResult selection (AnalysisPanel
        only calls this when the newly-current result is an
        XRDAnalysisResult or None).

        The detailed peak table itself is redisplayed independently by
        `AnalysisResultView` in the bottom Results tab (MainWindow drives
        both from the same history-selection event) -- this method only
        needs to drop any stale Results-table row selection carried over
        from the previously-shown result.

        Deliberately does NOT restore a live background/smoothing
        PREVIEW: those are transient, computed artifacts (see `set_
        figure`/`_refresh_detection_input_options`'s own docstrings), and
        silently recomputing arPLS/Savitzky-Golay as a side effect of
        clicking a History row would be surprising, not helpful --
        `result.parameters["preprocessing"]` (recorded at detection time)
        remains inspectable via `provenance_details`/the Results view for
        exactly what was actually used, even though this method doesn't
        regenerate it as a preview overlay."""
        self._current_result = result if isinstance(result, XRDAnalysisResult) else None
        if self._current_result is not None:
            self._radiation = self._current_result.radiation
            self._sync_radiation_combo()
            source_index = self.source_combo.findData(self._current_result.source_series_id)
            if source_index >= 0:
                self.source_combo.setCurrentIndex(source_index)
            self._restore_detection_settings(self._current_result.parameters.get("detection", {}))
        self._results_selected_rows = []
        self._refresh_detection_input_options()
        self.overlay_changed.emit()

    def _restore_detection_settings(self, detection_params: dict) -> None:
        """Reflects a stored result's own `prominence`/`distance`/`height`/
        `width` back into the detection controls -- best-effort, `dict.get`
        with each spinbox's own current value as the fallback, so a result
        saved before a given key existed (or with that key `None`, meaning
        "not used") leaves the corresponding control alone/unchecked
        rather than raising or zeroing it out.

        Signals are blocked for the duration: this reflects a HISTORICAL
        choice back into the controls, not a fresh edit by the researcher
        right now, so it must not itself set `_detection_defaults_touched`
        (see `_on_detection_param_edited`) -- restoring an old result and
        then switching to a different source series should still get a
        freshly computed data-dependent default for that series, exactly
        as if nothing had been restored."""
        self.prominence_spin.blockSignals(True)
        self.distance_spin.blockSignals(True)
        try:
            prominence = detection_params.get("prominence")
            if prominence is not None:
                self.prominence_spin.setValue(prominence)
            distance = detection_params.get("distance")
            if distance is not None:
                self.distance_spin.setValue(distance)
        finally:
            self.prominence_spin.blockSignals(False)
            self.distance_spin.blockSignals(False)
        height = detection_params.get("height")
        self.height_check.setChecked(height is not None)
        if height is not None:
            self.height_spin.setValue(height)
        width = detection_params.get("width")
        self.width_check.setChecked(width is not None)
        if width is not None:
            self.width_spin.setValue(width)

    def overlay_points(self) -> list[tuple[float, float, str]] | None:
        """(x, y, label) for every ENABLED peak of the current result, in
        the currently-selected label mode -- or None if there's nothing
        to show (no current result, active panel doesn't match, or the
        result has no enabled peaks)."""
        if self._current_result is None:
            return None
        if self._current_result.source_panel_id != self._figure.active_panel.id:
            return None
        mode = self.label_mode_combo.currentText()
        points = []
        for position, peak in enumerate(self._current_result.peaks, start=1):
            if not peak.enabled:
                continue
            label = ""
            if mode == _LABEL_MODE_NUMBER:
                label = str(position)
            elif mode == _LABEL_MODE_TWO_THETA:
                label = f"{peak.two_theta:.3f}°"
            elif mode == _LABEL_MODE_D_SPACING:
                d = self._peak_d_spacing(peak)
                label = f"{d:.4f} Å" if d is not None else ""
            points.append((peak.two_theta, peak.intensity, label))
        return points

    def preview_curve(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The transient preview curve (smoothed if present, else
        background) to draw on the canvas -- None once neither preview is
        current (inputs changed, or nothing previewed yet)."""
        if self._smooth_preview is not None:
            return self._smooth_preview.two_theta, self._smooth_preview.smoothed_intensity
        if self._background_preview is not None:
            return self._background_preview.two_theta, self._background_preview.baseline
        return None

    # --- source / radiation --------------------------------------------------

    def _current_source_series(self) -> PlotSeries | None:
        series_id = self.source_combo.currentData()
        if series_id is None:
            return None
        return self._figure.get_series(series_id)

    def _raw_xy(self) -> tuple[np.ndarray, np.ndarray] | None:
        series = self._current_source_series()
        if series is None:
            return None
        try:
            x, y = numeric_xy(series.dataframe, series.x_column, series.y_column)
        except (KeyError, InsufficientNumericDataError):
            return None
        return x.to_numpy(), y.to_numpy()

    def _on_source_changed(self) -> None:
        # "Add Peak" armed for whatever series/panel was previously
        # selected must not silently carry over to a different one --
        # see `disarm_manual_peak_mode`'s own docstring.
        self.disarm_manual_peak_mode()
        self._invalidate_previews()
        # A new source series is new DATA -- its own noise/intensity
        # scale deserves a freshly computed default (see `_detection_
        # defaults_touched`'s own docstring), not whatever was left over
        # from a previously selected series.
        self._detection_defaults_touched = False
        self._maybe_apply_default_detection_params()

    def _maybe_apply_default_detection_params(self) -> None:
        """Sets a conservative, data-dependent first-run Prominence (see
        `_default_prominence_from_signal`) from the newly-selected source
        series' RAW data -- unless the researcher has already edited
        Prominence/Minimum separation themselves for this series
        (`_detection_defaults_touched`), in which case their value is
        left alone. A no-op if no source is selected/resolvable yet."""
        if self._detection_defaults_touched:
            return
        xy = self._raw_xy()
        if xy is None:
            return
        _, y = xy
        prominence = _default_prominence_from_signal(y)
        self.prominence_spin.blockSignals(True)
        self.prominence_spin.setValue(prominence)
        self.prominence_spin.blockSignals(False)

    def _on_detection_param_edited(self, *_args) -> None:
        """Marks Prominence/Minimum separation as deliberately set by the
        researcher for the current source series -- see `_detection_
        defaults_touched`'s own docstring. Only fires for a REAL edit:
        `_maybe_apply_default_detection_params`/`_restore_detection_
        settings` both block these spinboxes' signals while they set a
        value programmatically, so this is never triggered by GNOVI's own
        code, only by the researcher actually touching a spinbox (typing,
        the up/down arrows, or the mouse wheel)."""
        self._detection_defaults_touched = True

    def _invalidate_previews(self) -> None:
        """Clears any transient background/smoothing preview -- and
        whatever Detection Input option depended on it -- because the
        resolved source data it was computed against no longer applies.
        The one shared place every preview-invalidating context change
        (`_on_source_changed`, `refresh`, `set_figure` -- see each of
        their own docstrings for exactly which real-world action routes
        through them) goes through, so "what makes a preview go stale"
        is answered in exactly one place. `_refresh_detection_input_
        options` (which this always calls) is what actually removes any
        now-unavailable Detection Input option and falls back to Raw."""
        self._background_preview = None
        self._smooth_preview = None
        self.add_corrected_button.setEnabled(False)
        self.add_smoothed_button.setEnabled(False)
        self.background_status_label.clear()
        self.smoothing_status_label.clear()
        self._refresh_detection_input_options()
        self.overlay_changed.emit()

    def _on_radiation_changed(self, *_args) -> None:
        data = self.radiation_combo.currentData()
        self.custom_wavelength_spin.setVisible(data == "custom")
        if data is None:
            self._radiation = None
        elif data == "custom":
            try:
                self._radiation = Radiation.custom(self.custom_wavelength_spin.value())
            except InvalidRadiationError:
                self._radiation = None
        else:
            self._radiation = RADIATION_PRESETS.get(data)

        if self._radiation is not None:
            self.resolved_wavelength_label.setText(
                f"Resolved wavelength: {self._radiation.wavelength_angstrom:.6g} Å"
            )
        else:
            self.resolved_wavelength_label.setText("Resolved wavelength: —")

        if self._current_result is not None and self._radiation is not None:
            self._current_result.radiation = self._radiation
            self.result_updated.emit(self._current_result)
            self.overlay_changed.emit()

    def _sync_radiation_combo(self) -> None:
        """Reflect `self._radiation` (e.g. restored via `load_result`)
        back into the combo/spin without re-triggering `_on_radiation_
        changed`'s own side effects."""
        if self._radiation is None:
            return
        self.radiation_combo.blockSignals(True)
        self.custom_wavelength_spin.blockSignals(True)
        matched = False
        for preset_id, radiation in RADIATION_PRESETS.items():
            if radiation.wavelength_angstrom == self._radiation.wavelength_angstrom and radiation.label == self._radiation.label:
                index = self.radiation_combo.findData(preset_id)
                if index >= 0:
                    self.radiation_combo.setCurrentIndex(index)
                    matched = True
                break
        if not matched:
            index = self.radiation_combo.findData("custom")
            self.radiation_combo.setCurrentIndex(index)
            self.custom_wavelength_spin.setValue(self._radiation.wavelength_angstrom)
            self.custom_wavelength_spin.setVisible(True)
        self.radiation_combo.blockSignals(False)
        self.custom_wavelength_spin.blockSignals(False)
        self.resolved_wavelength_label.setText(
            f"Resolved wavelength: {self._radiation.wavelength_angstrom:.6g} Å"
        )

    def _peak_d_spacing(self, peak: XRDPeakSeed) -> float | None:
        if self._current_result is None:
            return None
        try:
            return float(d_spacing(peak.two_theta, self._current_result.radiation.wavelength_angstrom))
        except InvalidBraggInputError:
            return None

    # --- background -----------------------------------------------------

    def _on_background_method_changed(self) -> None:
        method = self.background_method_combo.currentText()
        self.arpls_lam_spin.setVisible(method == _BACKGROUND_ARPLS)
        self.baseline_points_edit.setVisible(method == _BACKGROUND_POLYNOMIAL)
        self.polynomial_degree_spin.setVisible(method == _BACKGROUND_POLYNOMIAL)
        self._background_preview = None
        self.add_corrected_button.setEnabled(False)
        self.background_status_label.clear()
        self._refresh_detection_input_options()

    def _on_preview_background_clicked(self) -> None:
        xy = self._raw_xy()
        if xy is None:
            QMessageBox.warning(self, "XRD Peak Analysis", "Select a plotted 2D series first.")
            return
        x, y = xy
        method = self.background_method_combo.currentText()
        try:
            if method == _BACKGROUND_ARPLS:
                result = arpls_baseline(x, y, lam=self.arpls_lam_spin.value())
            elif method == _BACKGROUND_POLYNOMIAL:
                indices = _parse_index_ranges(self.baseline_points_edit.text(), len(x) - 1)
                result = polynomial_baseline(x, y, indices, degree=self.polynomial_degree_spin.value())
            else:
                self._background_preview = None
                self.add_corrected_button.setEnabled(False)
                self.background_status_label.setText("Background method is None -- nothing to preview.")
                self._refresh_detection_input_options()
                self.overlay_changed.emit()
                return
        except PybaselinesNotAvailableError as exc:
            QMessageBox.critical(
                self,
                "XRD Peak Analysis",
                "pybaselines is not installed. Install GNOVI with XRD support "
                "(pip install gnovi-plot[xrd]) to use arPLS background correction.",
            )
            self.background_status_label.setText(str(exc))
            return
        except (InvalidPreprocessingError, ValueError) as exc:
            QMessageBox.critical(self, "XRD Peak Analysis", str(exc))
            return

        self._background_preview = result
        self.add_corrected_button.setEnabled(True)
        rms = float(np.sqrt(np.mean(result.baseline**2))) if len(result.baseline) else 0.0
        self.background_status_label.setText(
            f"Background previewed ({result.method}) -- baseline RMS ≈ {rms:.4g}."
        )
        self._refresh_detection_input_options()
        self.overlay_changed.emit()

    def _on_add_corrected_clicked(self) -> None:
        if self._background_preview is None:
            return
        series = self._current_source_series()
        if series is None:
            return
        result = self._background_preview
        metadata = {
            "source_dataset_id": series.dataset.id,
            "source_series_id": series.id,
            "engine": "gnovi",
            "engine_version": _APP_VERSION,
            "operation": "xrd_background_correction",
            "parameters": {"method": result.method, **result.parameters},
        }
        dataset = Dataset(
            name=f"XRD background-corrected: {series.dataset.name}",
            dataframe=pd.DataFrame({series.x_column: result.two_theta, series.y_column: result.corrected}),
            metadata=metadata,
        )
        self._manager.add(dataset)
        new_series = PlotSeries.line(dataset, series.x_column, series.y_column, label=dataset.name)
        self.add_to_plot_requested.emit([new_series])
        self.status_message.emit(f"Added to plot: {new_series.label}")

    # --- smoothing -----------------------------------------------------

    def _on_smoothing_toggled(self, checked: bool) -> None:
        self.smoothing_window_spin.setVisible(checked)
        self.smoothing_order_spin.setVisible(checked)
        self.preview_smoothed_button.setVisible(checked)
        self._smooth_preview = None
        self.add_smoothed_button.setEnabled(False)
        if not checked:
            self.smoothing_status_label.clear()
        self._refresh_detection_input_options()

    def _on_preview_smoothed_clicked(self) -> None:
        xy = self._raw_xy()
        if xy is None:
            QMessageBox.warning(self, "XRD Peak Analysis", "Select a plotted 2D series first.")
            return
        x, y = xy
        try:
            result = savgol_smooth(
                x, y, window_length=self.smoothing_window_spin.value(), polyorder=self.smoothing_order_spin.value()
            )
        except InvalidPreprocessingError as exc:
            QMessageBox.critical(self, "XRD Peak Analysis", str(exc))
            return

        self._smooth_preview = result
        self.add_smoothed_button.setEnabled(True)
        self.smoothing_status_label.setText(
            f"Smoothed preview ready (window={result.parameters['window_length']}, "
            f"order={result.parameters['polyorder']}). Smoothing can change peak width/shape."
        )
        self._refresh_detection_input_options()
        self.overlay_changed.emit()

    def _on_add_smoothed_clicked(self) -> None:
        if self._smooth_preview is None:
            return
        series = self._current_source_series()
        if series is None:
            return
        result = self._smooth_preview
        metadata = {
            "source_dataset_id": series.dataset.id,
            "source_series_id": series.id,
            "engine": "gnovi",
            "engine_version": _APP_VERSION,
            "operation": "xrd_smoothing",
            "parameters": {"method": result.method, **result.parameters},
        }
        dataset = Dataset(
            name=f"XRD smoothed: {series.dataset.name}",
            dataframe=pd.DataFrame({series.x_column: result.two_theta, series.y_column: result.smoothed_intensity}),
            metadata=metadata,
        )
        self._manager.add(dataset)
        new_series = PlotSeries.line(dataset, series.x_column, series.y_column, label=dataset.name)
        self.add_to_plot_requested.emit([new_series])
        self.status_message.emit(f"Added to plot: {new_series.label}")

    # --- detection input chain --------------------------------------------

    def _refresh_detection_input_options(self) -> None:
        """Only offer inputs that are actually available right now -- never
        silently run background/smoothing just because a later input
        option requires it (see this class's own docstring)."""
        previous = self.detection_input_combo.currentData()
        options = [_INPUT_RAW]
        if self._background_preview is not None:
            options.append(_INPUT_BACKGROUND)
        if self.smoothing_enabled_check.isChecked():
            options.append(_INPUT_SMOOTH_RAW)
            if self._background_preview is not None:
                options.append(_INPUT_SMOOTH_BACKGROUND)
        self.detection_input_combo.blockSignals(True)
        self.detection_input_combo.clear()
        for option in options:
            self.detection_input_combo.addItem(_INPUT_LABELS[option], option)
        target = self.detection_input_combo.findData(previous)
        self.detection_input_combo.setCurrentIndex(target if target >= 0 else 0)
        self.detection_input_combo.blockSignals(False)
        self.detection_input_label.setText(
            f"Peak detection input: {self.detection_input_combo.currentText()}"
        )

    def _resolve_detection_xy(self) -> tuple[np.ndarray, np.ndarray] | None:
        raw = self._raw_xy()
        if raw is None:
            return None
        x, y = raw
        choice = self.detection_input_combo.currentData()
        if choice in (None, _INPUT_RAW):
            return x, y
        if choice == _INPUT_BACKGROUND:
            if self._background_preview is None:
                return None
            return self._background_preview.two_theta, self._background_preview.corrected
        if choice == _INPUT_SMOOTH_RAW:
            try:
                result = savgol_smooth(
                    x, y, window_length=self.smoothing_window_spin.value(), polyorder=self.smoothing_order_spin.value()
                )
            except InvalidPreprocessingError:
                return None
            return result.two_theta, result.smoothed_intensity
        if choice == _INPUT_SMOOTH_BACKGROUND:
            if self._background_preview is None:
                return None
            try:
                result = savgol_smooth(
                    self._background_preview.two_theta,
                    self._background_preview.corrected,
                    window_length=self.smoothing_window_spin.value(),
                    polyorder=self.smoothing_order_spin.value(),
                )
            except InvalidPreprocessingError:
                return None
            return result.two_theta, result.smoothed_intensity
        return x, y

    # --- peak detection -----------------------------------------------------

    def _on_find_peaks_clicked(self) -> None:
        series = self._current_source_series()
        if series is None:
            QMessageBox.warning(self, "XRD Peak Analysis", "Select a plotted 2D series first.")
            return
        if self._radiation is None:
            QMessageBox.warning(self, "XRD Peak Analysis", _NO_RADIATION_TEXT)
            return
        xy = self._resolve_detection_xy()
        if xy is None:
            QMessageBox.warning(self, "XRD Peak Analysis", "Could not resolve the selected detection input.")
            return
        x, y = xy

        detection_params = {
            "prominence": self.prominence_spin.value() if self.prominence_spin.value() > 0 else None,
            "distance": self.distance_spin.value() if self.distance_spin.value() > 0 else None,
            "height": self.height_spin.value() if self.height_check.isChecked() else None,
            "width": self.width_spin.value() if self.width_check.isChecked() else None,
        }
        try:
            peaks = detect_peaks(x, y, **detection_params)
        except InvalidPeakDetectionError as exc:
            QMessageBox.critical(self, "XRD Peak Analysis", str(exc))
            return

        if not peaks:
            self.detection_status_label.setText(
                "No peaks detected with the current prominence/settings."
            )
        else:
            self.detection_status_label.setText(f"{len(peaks)} peak candidate(s) detected.")

        parameters = {
            "detection": {k: v for k, v in detection_params.items()},
            "detection_input": self.detection_input_combo.currentData() or _INPUT_RAW,
            "preprocessing": {
                "background": (
                    {"method": self._background_preview.method, **self._background_preview.parameters}
                    if self._background_preview is not None
                    else None
                ),
                "smoothing": (
                    {**self._smooth_preview.parameters}
                    if self.smoothing_enabled_check.isChecked() and self._smooth_preview is not None
                    else None
                ),
            },
        }
        result = build_xrd_analysis_result(
            source_dataset_id=series.dataset.id,
            x_column=series.x_column,
            y_column=series.y_column,
            radiation=self._radiation,
            peaks=peaks,
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

    # --- manual peak editing --------------------------------------------------

    def _set_manual_peak_mode(self, active: bool) -> None:
        self._manual_peak_mode = active
        self.add_peak_button.blockSignals(True)
        self.add_peak_button.setChecked(active)
        self.add_peak_button.blockSignals(False)
        self.manual_peak_mode_changed.emit(active)

    def _on_add_peak_toggled(self, checked: bool) -> None:
        self._set_manual_peak_mode(checked)

    def disarm_manual_peak_mode(self) -> None:
        """Publicly disarm "Add Peak" (a no-op if it wasn't armed) --
        called from every context change where an already-armed click
        target stops making sense: this widget's own `_on_source_changed`/
        `set_figure` (source series/Workbench/project change), and
        `AnalysisPanel` reaching in on an active-panel switch
        (`disarm_xrd_manual_peak_mode`) or a switch away from the XRD
        tool (`_update_tool_visibility`) -- see each call site's own
        comment. A successful `add_manual_peak` or the researcher
        toggling the button off both already disarm directly; this method
        exists for every OTHER exit path Part 8 of this milestone's own
        bug-report notes lists, so none of them can leave a stale armed
        state (and its checked button/status text) behind."""
        self._set_manual_peak_mode(False)

    def add_manual_peak(self, two_theta: float, intensity: float) -> None:
        """Called by MainWindow after a canvas click while manual-peak
        mode is active. Adds a SEED at `(two_theta, intensity)` -- never a
        claim about a scientifically measured peak center (see
        `XRDPeakSeed.manual`). If no current result exists yet for the
        active panel, requires a radiation selection first and starts a
        fresh (empty) result to hold it."""
        self._set_manual_peak_mode(False)
        if self._radiation is None:
            QMessageBox.warning(self, "XRD Peak Analysis", _NO_RADIATION_TEXT)
            return
        series = self._current_source_series()
        if self._current_result is None:
            if series is None:
                QMessageBox.warning(self, "XRD Peak Analysis", "Select a plotted 2D series first.")
                return
            self._current_result = build_xrd_analysis_result(
                source_dataset_id=series.dataset.id,
                x_column=series.x_column,
                y_column=series.y_column,
                radiation=self._radiation,
                peaks=[],
                source_dataset_name=series.dataset.name,
                source_series_id=series.id,
                source_series_label=series.label,
                row_range=series.row_range,
                source_panel_id=self._figure.active_panel.id,
                parameters={"detection": None, "detection_input": _INPUT_RAW, "preprocessing": {"background": None, "smoothing": None}},
            )
            self._results_selected_rows = []
            self._current_result.peaks.append(XRDPeakSeed.manual(two_theta, intensity))
            self.overlay_changed.emit()
            self.analysis_result_ready.emit(self._current_result)
            return

        self._current_result.peaks.append(XRDPeakSeed.manual(two_theta, intensity))
        self.overlay_changed.emit()
        self.result_updated.emit(self._current_result)

    def set_selected_peak_rows(self, rows: list[int]) -> None:
        """Record which peak rows are selected in the bottom Results-tab
        detail table -- pushed in by `MainWindow` via `AnalysisPanel.
        xrd_set_selected_peak_rows` whenever that table's selection
        changes. Remove Selected / Enable-Disable act on exactly this
        (see `_selected_peak_rows`)."""
        self._results_selected_rows = sorted({int(r) for r in rows})

    def _selected_peak_rows(self) -> list[int]:
        """The currently-selected peak rows, clamped to the current
        result's actual peak count -- guards against a stale selection
        index surviving a change to the peak list."""
        if self._current_result is None:
            return []
        count = len(self._current_result.peaks)
        return [row for row in self._results_selected_rows if 0 <= row < count]

    def _on_remove_selected_clicked(self) -> None:
        if self._current_result is None:
            return
        rows = self._selected_peak_rows()
        if not rows:
            return
        for row in reversed(rows):
            del self._current_result.peaks[row]
        self._results_selected_rows = []
        self.overlay_changed.emit()
        self.result_updated.emit(self._current_result)

    def _on_toggle_enabled_clicked(self) -> None:
        if self._current_result is None:
            return
        rows = self._selected_peak_rows()
        if not rows:
            return
        for row in rows:
            peak = self._current_result.peaks[row]
            peak.enabled = not peak.enabled
        self.overlay_changed.emit()
        self.result_updated.emit(self._current_result)

    # --- export -----------------------------------------------------

    def _on_export_clicked(self) -> None:
        if self._current_result is None or not self._current_result.peaks:
            QMessageBox.information(self, "XRD Peak Analysis", "No peak table to export yet.")
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Export Peak Table", "", "CSV files (*.csv)")
        if not path:
            return
        self.export_peak_table_csv(path)

    def export_peak_table_csv(self, path: str) -> None:
        """Write the current result's peak table to `path` as CSV --
        exposed as a plain method (not only the click handler) so tests
        can exercise it without a file dialog.

        Explicit ``utf-8-sig`` encoding (not the platform default): the
        header row carries legitimate scientific characters -- ``θ`` (U+03B8),
        ``°`` (U+00B0), ``Å`` (U+00C5) -- which cp1252, Python's default
        text encoding on Windows, cannot represent, so the previous
        default-encoding ``open`` raised ``UnicodeEncodeError`` on the
        Windows CI runner during peak-table export. The ``-sig`` variant
        also prepends a UTF-8 BOM so Windows Excel opens the file as UTF-8
        rather than mojibake. ``newline=""`` stays for ``csv.writer``'s own
        line-ending control."""
        if self._current_result is None:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(PEAK_TABLE_COLUMNS)
            for row, peak in enumerate(self._current_result.peaks, start=1):
                d = self._peak_d_spacing(peak)
                writer.writerow(
                    [
                        row,
                        f"{peak.two_theta:.6f}",
                        f"{peak.intensity:.6g}",
                        f"{peak.prominence:.6g}" if peak.prominence is not None else "",
                        f"{d:.6f}" if d is not None else "",
                        peak.origin,
                        peak.enabled,
                    ]
                )
