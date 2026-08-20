from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.analysis.fitting import (
    EXPONENTIAL,
    GAUSSIAN,
    LINEAR,
    POLYNOMIAL,
    FitError,
    FitResult,
    fit_curve,
    sample_fit_curve,
)
from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.gui.widgets.active_panel_label import ActivePanelLabel
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries

_MODEL_OPTIONS = [
    ("Linear", LINEAR),
    ("Polynomial", POLYNOMIAL),
    ("Exponential", EXPONENTIAL),
    ("Gaussian", GAUSSIAN),
]

_NO_SOURCE_TEXT = (
    "No plotted line/scatter series in the active panel yet -- add one "
    "from the Plot page first."
)


def _eligible_series(figure: GnoviFigure) -> list[PlotSeries]:
    """Line/scatter series in the *active panel* that are safe to fit:
    excludes histograms (no `y_column` -- there is no curve to fit) and
    stale series (a missing column or invalid `row_range` -- fitting them
    would either crash or silently fit garbage)."""
    return [s for s in figure.series if s.y_column is not None and not s.stale]


class AnalysisPanel(QWidget):
    """The general Analysis drawer page: one `CollapsibleSection` per
    analysis tool, run against the *active panel's* plotted series.

    This milestone has exactly one section -- Curve Fitting. Adding a later
    tool (statistics, peak analysis, FFT, smoothing, a domain-specific
    module) means adding another `CollapsibleSection` to this same page,
    not a new top-level drawer page and not a new Results-tab mechanism:
    every tool reports through the same `analysis_result_ready` signal,
    which carries the generic `AnalysisResult` base type -- this panel
    never imports or checks for a specific subclass, so `MainWindow`'s
    wiring to `AnalysisResultView` doesn't change either when a second
    tool is added.

    Mirrors `PlotSeriesPanel`'s `set_figure`/`refresh` pattern: no
    push-based signal from `GnoviFigure` itself, the owner (`MainWindow`)
    calls `refresh()` after anything that could change the active panel's
    plotted series (add/remove/edit, panel switch, Workbench switch).

    `dataset_manager` is used for exactly one thing: registering the
    derived Dataset a successful fit's "Add Fit Curve to Plot" creates --
    mirrors `DatasetPanel`'s own `_manager`/`set_manager()` pattern, since
    (like datasets generally) it's project-scoped, not figure-scoped, and
    needs its own repoint on Open/New Project (see `set_manager`).
    """

    analysis_result_ready = Signal(AnalysisResult)
    # Same name/shape as DatasetPanel.add_to_plot_requested -- MainWindow
    # wires both to the same existing handler (undo checkpoint, dirty,
    # re-render), so a fit curve joins the plot through the identical path
    # any other new series does.
    add_to_plot_requested = Signal(list)  # list[PlotSeries]

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._manager = dataset_manager

        # The last successful fit, kept only until it's either added to the
        # plot or invalidated (source series or model changed) -- never
        # persisted, never a history (see class docstring / PR scope).
        self._pending_fit: FitResult | None = None
        self._pending_fit_x_range: tuple[float, float] | None = None

        self.active_panel_label = ActivePanelLabel(figure)

        self.source_label = QLabel("Source series")
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(90)

        self.status_label = QLabel(_NO_SOURCE_TEXT)
        self.status_label.setWordWrap(True)

        self.model_label = QLabel("Model")
        self.model_combo = QComboBox()
        for text, model in _MODEL_OPTIONS:
            self.model_combo.addItem(text, model)
        self.model_combo.setMinimumWidth(90)

        self.degree_label = QLabel("Polynomial order")
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 10)
        self.degree_spin.setValue(2)

        self.run_fit_button = QPushButton("Run Fit")
        self.run_fit_button.setProperty("primary", True)

        self.pending_fit_label = QLabel("")
        self.pending_fit_label.setWordWrap(True)
        self.pending_fit_label.setVisible(False)

        self.add_fit_curve_button = QPushButton("Add Fit Curve to Plot")
        self.add_fit_curve_button.setEnabled(False)

        fit_group = QGroupBox("Curve Fitting")
        fit_layout = QVBoxLayout(fit_group)
        fit_layout.addWidget(self.active_panel_label)
        fit_layout.addWidget(self.source_label)
        fit_layout.addWidget(self.source_combo)
        fit_layout.addWidget(self.status_label)
        fit_layout.addWidget(self.model_label)
        fit_layout.addWidget(self.model_combo)
        fit_layout.addWidget(self.degree_label)
        fit_layout.addWidget(self.degree_spin)
        fit_layout.addWidget(self.run_fit_button)
        fit_layout.addWidget(self.pending_fit_label)
        fit_layout.addWidget(self.add_fit_curve_button)

        self.fit_section = CollapsibleSection("Curve Fitting", fit_group)

        layout = QVBoxLayout(self)
        layout.addWidget(self.fit_section)
        layout.addStretch(1)

        self.model_combo.currentIndexChanged.connect(self._update_model_controls)
        self.model_combo.currentIndexChanged.connect(self._invalidate_pending_fit)
        self.source_combo.currentIndexChanged.connect(self._invalidate_pending_fit)
        self.run_fit_button.clicked.connect(self._on_run_fit_clicked)
        self.add_fit_curve_button.clicked.connect(self._on_add_fit_curve_clicked)

        self._update_model_controls()
        self.refresh()

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint this panel at a different `GnoviFigure` (e.g. a
        Workbench switch) and reload from it."""
        self._figure = figure
        self.refresh()

    def set_manager(self, dataset_manager: DatasetManager) -> None:
        """Repoint this panel at a different `DatasetManager` (Open/New
        Project only -- datasets are project-scoped, not figure-scoped;
        see the class docstring)."""
        self._manager = dataset_manager

    def refresh(self) -> None:
        """Rebuild the source-series list from the active panel's current
        series, preserving the current selection by id where it still
        exists. Call after anything that can change which series are
        plotted in the active panel, or which panel is active."""
        self.active_panel_label.refresh(self._figure)

        previous_id = self.source_combo.currentData()
        eligible = _eligible_series(self._figure)

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

        has_eligible = bool(eligible)
        self.source_combo.setEnabled(has_eligible)
        self.run_fit_button.setEnabled(has_eligible)
        self.status_label.setVisible(not has_eligible)

    def _current_source_series(self) -> PlotSeries | None:
        series_id = self.source_combo.currentData()
        if series_id is None:
            return None
        return self._figure.get_series(series_id)

    def _update_model_controls(self) -> None:
        is_polynomial = self.model_combo.currentData() == POLYNOMIAL
        self.degree_label.setVisible(is_polynomial)
        self.degree_spin.setVisible(is_polynomial)

    def _invalidate_pending_fit(self) -> None:
        """Clear a pending (not-yet-added) fit result -- called whenever
        the source series or model selection changes, so a fit for one
        series/model can never be added to the plot under a different
        one's name. A degree change or a re-run that fails does *not*
        invalidate an already-successful pending fit -- it's still a
        genuine, self-consistent result for the series/model it was run
        against."""
        self._pending_fit = None
        self._pending_fit_x_range = None
        self.pending_fit_label.clear()
        self.pending_fit_label.setVisible(False)
        self.add_fit_curve_button.setEnabled(False)

    def _on_run_fit_clicked(self) -> None:
        series = self._current_source_series()
        if series is None:
            QMessageBox.warning(self, "Curve Fitting", "Select a source series to fit.")
            return

        try:
            x, y = numeric_xy(series.dataframe, series.x_column, series.y_column)
        except (KeyError, InsufficientNumericDataError) as exc:
            QMessageBox.critical(self, "Curve Fitting", str(exc))
            return

        model = self.model_combo.currentData()
        try:
            result = fit_curve(
                x.to_numpy(),
                y.to_numpy(),
                model,
                source_dataset_id=series.dataset.id,
                source_series_id=series.id,
                x_column=series.x_column,
                y_column=series.y_column,
                row_range=series.row_range,
                degree=self.degree_spin.value(),
            )
        except FitError as exc:
            QMessageBox.critical(self, "Curve Fitting", str(exc))
            return

        self._pending_fit = result
        self._pending_fit_x_range = (float(x.min()), float(x.max()))
        self.pending_fit_label.setText(f"Ready to add: {result.summary()}")
        self.pending_fit_label.setVisible(True)
        self.add_fit_curve_button.setEnabled(True)

        self.analysis_result_ready.emit(result)

    def _on_add_fit_curve_clicked(self) -> None:
        if self._pending_fit is None or self._pending_fit_x_range is None:
            return  # button is disabled in this state; defensive no-op only

        result = self._pending_fit
        x_min, x_max = self._pending_fit_x_range
        x_smooth, y_smooth = sample_fit_curve(result, x_min, x_max)

        metadata = result.to_dict()
        metadata["x_min"] = x_min
        metadata["x_max"] = x_max
        metadata["num_points"] = len(x_smooth)

        fit_dataset = Dataset(
            name=f"Fit: {result.model}",
            dataframe=pd.DataFrame({result.x_column: x_smooth, result.y_column: y_smooth}),
            metadata=metadata,
        )
        self._manager.add(fit_dataset)

        series = PlotSeries.line(fit_dataset, result.x_column, result.y_column)
        self.add_to_plot_requested.emit([series])
