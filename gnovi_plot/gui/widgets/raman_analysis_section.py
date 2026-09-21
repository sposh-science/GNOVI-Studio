from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.core.app_info import __version__ as _APP_VERSION
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.gui.widgets.analysis_section import AnalysisSection, eligible_analysis_series
from gnovi_plot.gui.widgets.scroll_safe_controls import (
    ScrollSafeComboBox,
    ScrollSafeDoubleSpinBox,
    ScrollSafeSpinBox,
)
from gnovi_plot.modules.raman.peaks import InvalidPeakDetectionError, RamanPeakSeed, detect_raman_peaks
from gnovi_plot.modules.raman.results import RamanAnalysisResult, build_raman_analysis_result
from gnovi_plot.modules.xrd.preprocessing import (
    InvalidPreprocessingError,
    PybaselinesNotAvailableError,
    arpls_baseline,
    polynomial_baseline,
    savgol_smooth,
)
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries

_NO_SOURCE_TEXT = (
    "No plotted line/scatter series in the active panel yet -- add one "
    "from the 2D page first."
)
_PANEL3D_TEXT = (
    "Raman Peak Analysis works on a 2D Panel's plotted series -- switch to "
    "(or add) a 2D panel first."
)

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
_LABEL_MODE_RAMAN_SHIFT = "Raman shift"

# See `_default_prominence_from_signal`'s own docstring -- mirrors
# `xrd_analysis_section._PROMINENCE_NOISE_MULTIPLIER`'s exact reasoning.
# This is a small, private, pure-NumPy statistical helper (takes a bare
# `y: np.ndarray`, nothing XRD- or Raman-specific about the math itself)
# -- reimplemented here rather than imported across GUI modules, since
# the XRD original is module-private (leading underscore) and importing
# a private symbol from a sibling GUI file is exactly the kind of
# cross-module coupling this milestone's own scope keeps small.
_PROMINENCE_NOISE_MULTIPLIER = 5.0


def _default_prominence_from_signal(y: np.ndarray) -> float:
    """A conservative, transparent, data-dependent STARTING Prominence for
    `scipy.signal.find_peaks` -- without this, a first-run Prominence of 0
    (no threshold at all) on real noisy data returns essentially every
    local maximum, including noise fluctuations, as a "peak" (the same
    failure mode `xrd_analysis_section`'s own version of this helper was
    written to guard against).

    Uses `1.4826 * median(abs(d - median(d)))` of the signal's first
    differences `d` -- the standard robust estimator of a signal's local
    noise standard deviation. NOT an automatic "correct" prominence or a
    claim about how many real peaks exist -- one simple, reproducible
    number the researcher sees in the Prominence field and can freely
    override before or after running Find Peaks."""
    if y.size < 2:
        return 0.0
    diffs = np.diff(y)
    noise_scale = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs))))
    return max(noise_scale * _PROMINENCE_NOISE_MULTIPLIER, 0.0)


def _parse_index_ranges(text: str, max_index: int) -> list[int]:
    """Parse "0-15, 180-200, 250" (row positions, end-inclusive) into a
    sorted, de-duplicated list of valid indices -- mirrors
    `xrd_analysis_section._parse_index_ranges` exactly (pure text
    parsing, nothing domain-specific). Raises `ValueError` with a clear
    message for malformed/out-of-range input -- never silently drops or
    clamps a bad range."""
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


class RamanAnalysisSection(AnalysisSection):
    """Raman Peak Analysis -- an Analysis-page tool alongside Curve
    Fitting/XRD/CV (see `AnalysisPanel`'s own docstring). Native GNOVI
    numerical foundation only (`gnovi_plot.modules.raman`) -- detection
    only, no peak-profile fitting, no FWHM/linewidth, no material
    identification (see Issue #73's own scope notes; peak-profile
    fitting is explicitly deferred, matching Issue #72's own scope).

    Mirrors `XRDAnalysisSection`'s established shape (source selection,
    background/smoothing preprocessing with a preview, peak detection,
    manual peak add/remove/enable, Analysis History integration) minus
    everything genuinely XRD-specific: no radiation/wavelength/d-spacing
    (Raman shift needs no per-measurement calibration constant), no Peak
    Profile Fitting subsection.

    Background/smoothing correction reuses `polynomial_baseline`/
    `arpls_baseline`/`savgol_smooth` directly from
    `gnovi_plot.modules.xrd.preprocessing`, unchanged -- these are
    numerically domain-independent (see the read-only Raman audits this
    milestone is based on); they are not moved, duplicated, or renamed.

    Background/smoothing previews are transient: `_background_preview`/
    `_smooth_preview` are plain local state, never registered as a
    `Dataset`/`PlotSeries` until the scientist explicitly clicks "Add
    Corrected/Smoothed Curve to Plot" -- the same derived-`Dataset`
    pattern every other GNOVI analysis tool already uses.

    "Find Peaks" always produces a BRAND NEW `RamanAnalysisResult` (a new
    Analysis History entry) -- the same convention XRD's "Find Peaks"/
    Curve Fitting's "Run Fit" already use. Manual add/remove/enable/
    disable, by contrast, edit `_current_result` IN PLACE and emit
    `result_updated` -- dirty + redisplay, no new history entry.
    """

    # analysis_result_ready/result_updated/overlay_changed/manual_peak_
    # mode_changed/status_message are inherited from AnalysisSection
    # unchanged. add_to_plot_requested stays declared here (not on the
    # shared base) -- mirrors XRDAnalysisSection's own reasoning: a
    # derived-curve-add signal is only declared by a section that
    # actually has one. Raman has no fitted-curve-removal workflow (no
    # fitting in this milestone), so unlike XRD there is no
    # remove_fit_curve_requested here.
    add_to_plot_requested = Signal(list)  # list[PlotSeries]

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(figure, dataset_manager, parent)
        self._current_result: RamanAnalysisResult | None = None
        self._background_preview = None
        self._smooth_preview = None
        # See `_maybe_apply_default_detection_params`'s own docstring --
        # True once the researcher has edited Prominence/Minimum
        # separation themselves for the currently-selected source series,
        # so a freshly computed data-dependent default never silently
        # overwrites a deliberate choice.
        self._detection_defaults_touched = False

        # --- Source -----------------------------------------------------
        self.source_label = QLabel("Source series")
        self.source_combo = ScrollSafeComboBox()
        # Bounded the same way XRD/CV's source combos already are -- see
        # `XRDAnalysisSection.source_combo`'s own comment for exactly why
        # (AdjustToContentsOnFirstShow only measures once, so an
        # unbounded combo can get stuck wider than the drawer).
        self.source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.source_combo.setMinimumContentsLength(20)
        self.status_label = QLabel(_NO_SOURCE_TEXT)
        self.status_label.setWordWrap(True)

        # --- Preprocessing: Background --------------------------------
        self.background_method_combo = ScrollSafeComboBox()
        self.background_method_combo.addItems([_BACKGROUND_NONE, _BACKGROUND_ARPLS, _BACKGROUND_POLYNOMIAL])
        self.arpls_lam_spin = ScrollSafeDoubleSpinBox()
        self.arpls_lam_spin.setDecimals(0)
        self.arpls_lam_spin.setRange(1.0, 1e12)
        self.arpls_lam_spin.setValue(1e5)
        self.baseline_points_edit = QLineEdit()
        self.baseline_points_edit.setPlaceholderText("e.g. 0-15, 180-200")
        self.polynomial_degree_spin = ScrollSafeSpinBox()
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

        # --- Preprocessing: Smoothing -----------------------------------
        self.smoothing_enabled_check = QCheckBox("Enable Savitzky–Golay")
        self.smoothing_window_spin = ScrollSafeSpinBox()
        self.smoothing_window_spin.setRange(3, 9999)
        self.smoothing_window_spin.setSingleStep(2)
        self.smoothing_window_spin.setValue(11)
        self.smoothing_order_spin = ScrollSafeSpinBox()
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

        preprocessing_group = QGroupBox("Preprocessing")
        preprocessing_layout = QVBoxLayout(preprocessing_group)
        preprocessing_layout.addWidget(background_group)
        preprocessing_layout.addWidget(smoothing_group)

        # --- Detection input chain -----------------------------------------
        self.detection_input_combo = ScrollSafeComboBox()
        self.detection_input_label = QLabel("Peak detection input: Raw")

        # --- Peak detection -----------------------------------------------
        self.prominence_spin = ScrollSafeDoubleSpinBox()
        self.prominence_spin.setDecimals(4)
        self.prominence_spin.setRange(0.0, 1e12)
        self.prominence_spin.setValue(0.0)
        self.distance_spin = ScrollSafeSpinBox()
        self.distance_spin.setRange(0, 100000)
        self.distance_spin.setValue(0)
        self.height_spin = ScrollSafeDoubleSpinBox()
        self.height_spin.setDecimals(4)
        self.height_spin.setRange(0.0, 1e12)
        self.height_spin.setValue(0.0)
        self.width_spin = ScrollSafeDoubleSpinBox()
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

        # --- Peak actions -----------------------------------------------
        # The detailed peak table itself lives in the bottom Results tab
        # (`gui.widgets.analysis_result_view.AnalysisResultView`, fed by
        # `RamanAnalysisResult.detail_table()`) -- this narrow sidebar
        # keeps only the actions, which operate on whatever rows are
        # selected there (see `set_selected_peak_rows`, inherited).
        self.add_peak_button = QPushButton("Add Peak (click graph)")
        self.add_peak_button.setCheckable(True)
        self.remove_peak_button = QPushButton("Remove Selected")
        self.toggle_enabled_button = QPushButton("Enable/Disable Selected")
        self.peak_actions_hint = QLabel(
            "Select peak rows in the Results tab below, then use these actions."
        )
        self.peak_actions_hint.setWordWrap(True)
        self.label_mode_combo = ScrollSafeComboBox()
        self.label_mode_combo.addItems([_LABEL_MODE_OFF, _LABEL_MODE_NUMBER, _LABEL_MODE_RAMAN_SHIFT])

        actions_group = QGroupBox("Peak Actions")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.addWidget(self.add_peak_button)
        actions_layout.addWidget(self.remove_peak_button)
        actions_layout.addWidget(self.toggle_enabled_button)
        actions_layout.addWidget(self.peak_actions_hint)
        actions_layout.addWidget(QLabel("Graph labels"))
        actions_layout.addWidget(self.label_mode_combo)

        layout = QVBoxLayout(self)
        layout.addWidget(self.source_label)
        layout.addWidget(self.source_combo)
        layout.addWidget(self.status_label)
        layout.addWidget(preprocessing_group)
        layout.addWidget(detection_group)
        layout.addWidget(actions_group)
        layout.addStretch(1)

        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
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
        New/Open Project. Unconditionally invalidates any background/
        smoothing preview: it was computed against the OLD figure's
        source series, which no longer has any relationship to whatever
        `refresh()` resolves as the selected source next (mirrors
        `XRDAnalysisSection.set_figure`'s own reasoning)."""
        self._figure = figure
        self._current_result = None
        self._results_selected_rows = []
        self._invalidate_previews()
        self._set_manual_peak_mode(False)
        self.refresh()

    # set_manager/is_manual_peak_mode/set_selected_peak_rows/
    # disarm_manual_peak_mode are inherited from AnalysisSection unchanged.

    def refresh(self) -> None:
        """Rebuild the source-series list; disable everything with a clear
        explanation when the active panel is a Panel3D (see
        `eligible_analysis_series`) or has no eligible 2D series yet.
        Called after any figure-content change or active-panel switch,
        not just a source-series edit (mirrors
        `XRDAnalysisSection.refresh`'s own reasoning)."""
        is_panel3d = isinstance(self._figure.active_panel, Panel3D)
        eligible = eligible_analysis_series(self._figure)

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

        # See `XRDAnalysisSection.refresh`'s own comment on exactly why
        # `setCurrentIndex` above does not reliably emit
        # `currentIndexChanged` for a first population or a same-index
        # resolved-series change -- compare the resolved id directly
        # rather than relying on the signal.
        current_source_id = self.source_combo.currentData()
        if current_source_id != previous_id:
            self._detection_defaults_touched = False
        self._maybe_apply_default_detection_params()

        has_eligible = bool(eligible)
        enabled = has_eligible and not is_panel3d
        self.source_combo.setEnabled(enabled)
        for widget in (
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

    def current_result(self) -> RamanAnalysisResult | None:
        return self._current_result

    def load_result(self, result: AnalysisResult | None) -> None:
        """Called when the shared Analysis History selection changes --
        restores `result` (if it's a `RamanAnalysisResult`) as the
        working source selection/detection settings, without rerunning
        detection. Deliberately does NOT restore a live background/
        smoothing PREVIEW: those are transient, computed artifacts, and
        silently recomputing arPLS/Savitzky-Golay as a side effect of
        clicking a History row would be surprising, not helpful --
        `result.parameters["preprocessing"]` (recorded at detection time)
        remains inspectable via the Results view for exactly what was
        actually used, even though this method doesn't regenerate it as
        a preview overlay (mirrors `XRDAnalysisSection.load_result`'s own
        reasoning exactly)."""
        self._current_result = result if isinstance(result, RamanAnalysisResult) else None
        if self._current_result is not None:
            source_index = self.source_combo.findData(self._current_result.source_series_id)
            if source_index >= 0:
                self.source_combo.setCurrentIndex(source_index)
            self._restore_detection_settings(self._current_result.parameters.get("detection", {}))
        self._results_selected_rows = []
        self._refresh_detection_input_options()
        self.overlay_changed.emit()

    def _restore_detection_settings(self, detection_params: dict) -> None:
        """Reflects a stored result's own `prominence`/`distance`/
        `height`/`width` back into the detection controls -- best-effort,
        mirrors `XRDAnalysisSection._restore_detection_settings` exactly.
        Signals are blocked for the duration: this reflects a HISTORICAL
        choice, not a fresh edit, so it must not itself set
        `_detection_defaults_touched`."""
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
        result has no enabled peaks). No d-spacing label mode: there is
        no Raman equivalent."""
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
            elif mode == _LABEL_MODE_RAMAN_SHIFT:
                label = f"{peak.raman_shift:.1f} cm⁻¹"
            points.append((peak.raman_shift, peak.intensity, label))
        return points

    def preview_curve(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The transient preview curve (smoothed if present, else
        background) to draw on the canvas -- None once neither preview is
        current. Never both at once: the smoothed preview already
        reflects the baseline correction too when both are active (see
        `_on_preview_smoothed_clicked`), so showing the background
        preview alongside it would just be a stale, misleading duplicate
        (mirrors `XRDAnalysisSection.preview_curve`'s own reasoning)."""
        if self._smooth_preview is not None:
            return self._smooth_preview.two_theta, self._smooth_preview.smoothed_intensity
        if self._background_preview is not None:
            return self._background_preview.two_theta, self._background_preview.baseline
        return None

    # --- source ---------------------------------------------------------

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
        # scale deserves a freshly computed default, not whatever was
        # left over from a previously selected series.
        self._detection_defaults_touched = False
        self._maybe_apply_default_detection_params()

    def _maybe_apply_default_detection_params(self) -> None:
        """Sets a conservative, data-dependent first-run Prominence (see
        `_default_prominence_from_signal`) from the newly-selected source
        series' RAW data -- unless the researcher has already edited
        Prominence/Minimum separation themselves for this series. A
        no-op if no source is selected/resolvable yet."""
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
        self._detection_defaults_touched = True

    def _invalidate_previews(self) -> None:
        """Clears any transient background/smoothing preview -- and
        whatever Detection Input option depended on it -- because the
        resolved source data it was computed against no longer applies.
        The one shared place every preview-invalidating context change
        goes through (mirrors `XRDAnalysisSection._invalidate_previews`'s
        own reasoning)."""
        self._background_preview = None
        self._smooth_preview = None
        self.add_corrected_button.setEnabled(False)
        self.add_smoothed_button.setEnabled(False)
        self.background_status_label.clear()
        self.smoothing_status_label.clear()
        self._refresh_detection_input_options()
        self.overlay_changed.emit()

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
            QMessageBox.warning(self, "Raman Peak Analysis", "Select a plotted 2D series first.")
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
                "Raman Peak Analysis",
                "pybaselines is not installed. Install GNOVI with baseline-correction "
                "support (pip install gnovi-plot[xrd]) to use arPLS background correction.",
            )
            self.background_status_label.setText(str(exc))
            return
        except (InvalidPreprocessingError, ValueError) as exc:
            QMessageBox.critical(self, "Raman Peak Analysis", str(exc))
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
            "operation": "raman_background_correction",
            "parameters": {"method": result.method, **result.parameters},
        }
        dataset = Dataset(
            name=f"Raman background-corrected: {series.dataset.name}",
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
            QMessageBox.warning(self, "Raman Peak Analysis", "Select a plotted 2D series first.")
            return
        # Smoothing operates on the baseline-corrected spectrum when one
        # is active, otherwise on the raw spectrum -- baseline correction
        # runs first in the processing order (raw -> baseline -> smoothing
        # -> detection), so the preview always shows the actual FINAL
        # processed spectrum, never smoothing alone drawn against raw
        # data that detection would never actually use.
        if self._background_preview is not None:
            x, y = self._background_preview.two_theta, self._background_preview.corrected
        else:
            x, y = xy
        try:
            result = savgol_smooth(
                x, y, window_length=self.smoothing_window_spin.value(), polyorder=self.smoothing_order_spin.value()
            )
        except InvalidPreprocessingError as exc:
            QMessageBox.critical(self, "Raman Peak Analysis", str(exc))
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
            "operation": "raman_smoothing",
            "parameters": {"method": result.method, **result.parameters},
        }
        dataset = Dataset(
            name=f"Raman smoothed: {series.dataset.name}",
            dataframe=pd.DataFrame({series.x_column: result.two_theta, series.y_column: result.smoothed_intensity}),
            metadata=metadata,
        )
        self._manager.add(dataset)
        new_series = PlotSeries.line(dataset, series.x_column, series.y_column, label=dataset.name)
        self.add_to_plot_requested.emit([new_series])
        self.status_message.emit(f"Added to plot: {new_series.label}")

    # --- detection input chain --------------------------------------------

    def _refresh_detection_input_options(self) -> None:
        """Only offer inputs that are actually available right now --
        never silently run background/smoothing just because a later
        input option requires it (mirrors
        `XRDAnalysisSection._refresh_detection_input_options` exactly)."""
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
        """The exact (x, y) peak detection will run against, for whichever
        input the researcher selected -- raw, background-corrected,
        smoothed raw, or smoothed background-corrected (smoothing always
        applied AFTER baseline correction when both are active). Mirrors
        `XRDAnalysisSection._resolve_detection_xy` exactly; never mutates
        the raw arrays it starts from."""
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
            QMessageBox.warning(self, "Raman Peak Analysis", "Select a plotted 2D series first.")
            return
        xy = self._resolve_detection_xy()
        if xy is None:
            QMessageBox.warning(self, "Raman Peak Analysis", "Could not resolve the selected detection input.")
            return
        x, y = xy

        detection_params = {
            "prominence": self.prominence_spin.value() if self.prominence_spin.value() > 0 else None,
            "distance": self.distance_spin.value() if self.distance_spin.value() > 0 else None,
            "height": self.height_spin.value() if self.height_check.isChecked() else None,
            "width": self.width_spin.value() if self.width_check.isChecked() else None,
        }
        try:
            peaks = detect_raman_peaks(x, y, **detection_params)
        except InvalidPeakDetectionError as exc:
            QMessageBox.critical(self, "Raman Peak Analysis", str(exc))
            return

        if not peaks:
            self.detection_status_label.setText(
                "No peaks detected with the current prominence/settings."
            )
        else:
            self.detection_status_label.setText(f"{len(peaks)} peak candidate(s) detected.")

        # Provenance: makes unambiguous what data were actually used for
        # detection -- the resolved detection_input plus exactly which
        # preprocessing (if any) produced it. Mirrors the established
        # GNOVI schema `XRDAnalysisSection._on_find_peaks_clicked` already
        # uses.
        parameters = {
            "detection": dict(detection_params),
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
        result = build_raman_analysis_result(
            source_dataset_id=series.dataset.id,
            x_column=series.x_column,
            y_column=series.y_column,
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

    # disarm_manual_peak_mode is inherited from AnalysisSection unchanged.

    def add_manual_peak(self, x: float, y: float) -> None:
        """Called by MainWindow after a canvas click while manual-peak
        mode is active (via the generic dispatch from Issue #62). Adds a
        SEED at `(x, y)` -- never a claim about a scientifically measured
        peak center (see `RamanPeakSeed.manual`). If no current result
        exists yet for the active panel, starts a fresh (empty) result to
        hold it -- simpler than XRD's equivalent, which additionally
        requires a radiation selection first; Raman has no such
        precondition."""
        self._set_manual_peak_mode(False)
        series = self._current_source_series()
        if self._current_result is None:
            if series is None:
                QMessageBox.warning(self, "Raman Peak Analysis", "Select a plotted 2D series first.")
                return
            self._current_result = build_raman_analysis_result(
                source_dataset_id=series.dataset.id,
                x_column=series.x_column,
                y_column=series.y_column,
                peaks=[],
                source_dataset_name=series.dataset.name,
                source_series_id=series.id,
                source_series_label=series.label,
                row_range=series.row_range,
                source_panel_id=self._figure.active_panel.id,
                parameters={"detection": None, "detection_input": _INPUT_RAW, "preprocessing": {"background": None, "smoothing": None}},
            )
            self._results_selected_rows = []
            self._current_result.peaks.append(RamanPeakSeed.manual(x, y))
            self.overlay_changed.emit()
            self.analysis_result_ready.emit(self._current_result)
            return

        self._current_result.peaks.append(RamanPeakSeed.manual(x, y))
        self.overlay_changed.emit()
        self.result_updated.emit(self._current_result)

    # set_selected_peak_rows is inherited from AnalysisSection unchanged --
    # pushed in by MainWindow whenever the bottom Results-tab detail
    # table's selection changes; Remove Selected/Enable-Disable act on
    # exactly this (see `_selected_peak_rows` below).

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
