from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.analysis.fitting import (
    DEFAULT_CURVE_SAMPLES,
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
from gnovi_plot.gui.widgets.analysis_result_view import resolve_live_xy
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries

_HISTORY_EMPTY_TEXT = "No completed analysis results for this panel yet."

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

    The Analysis History section lists the active panel's full result
    history (see `core.workbench.Workbench.analysis_results`), oldest
    first, pushed in by `MainWindow` via `sync_history()`. Selecting an
    entry there makes it `_current_result` -- the one Results/Add/Remove
    Fit Curve all act on -- without rerunning anything; see
    `_on_history_row_changed`.

    Invariant: one `FitResult.result_id` maps to at most one generated
    fit `PlotSeries` in its source panel -- "Add Fit Curve to Plot" and
    "Remove Fit Curve from Plot" are a strict toggle on `_current_result`
    (see `_refresh_fit_curve_buttons`), never a duplication mechanism. A
    deliberate second styled copy is a job for a future "Duplicate
    Series" command on the Series page, not repeated clicking here.
    """

    analysis_result_ready = Signal(AnalysisResult)
    # Same name/shape as DatasetPanel.add_to_plot_requested -- MainWindow
    # wires both to the same existing handler (undo checkpoint, dirty,
    # re-render), so a fit curve joins the plot through the identical path
    # any other new series does.
    add_to_plot_requested = Signal(list)  # list[PlotSeries]
    # `list[str]` of PlotSeries ids to remove -- same shape/spirit as
    # `add_to_plot_requested`; MainWindow routes both through the figure's
    # normal series-removal path (undo checkpoint, dirty, re-render).
    remove_fit_curve_requested = Signal(list)
    # The scientist picked a different entry in the Analysis History list
    # -- MainWindow records this as the active panel's new selection (see
    # `core.workbench.Workbench.analysis_results.set_current`) and pushes
    # it to `AnalysisResultView`. Never emitted for a programmatic
    # `sync_history()` rebuild (panel/Workbench switch, undo/redo, project
    # load) -- only for an actual click, via `_on_history_row_changed`.
    history_result_selected = Signal(AnalysisResult)

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._manager = dataset_manager

        # The last successful fit, kept only until the source/model
        # selection changes -- used solely for the ephemeral "Ready to
        # add: ..." label right after "Run Fit" (see `_on_run_fit_clicked`/
        # `_invalidate_pending_fit`). Add/Remove Fit Curve themselves act
        # on `_current_result`, not this -- a freshly run fit becomes
        # `_current_result` too, synchronously, via the
        # `analysis_result_ready` -> `sync_history` round trip.
        self._pending_fit: FitResult | None = None

        # Whichever result is selected -- Results/History-list-highlight/
        # Add-Remove Fit Curve all act on this one. Pushed in by
        # MainWindow via `sync_history()` on every panel/Workbench switch,
        # undo/redo, figure-content change, and project load; updated
        # locally the instant the scientist clicks a different History
        # row (see `_on_history_row_changed`), which also tells
        # MainWindow via `history_result_selected` so the selection is
        # recorded and persisted.
        self._current_result: FitResult | None = None
        self._matched_fit_curve_series_ids: list[str] = []

        # The active panel's full result history, oldest first, as last
        # pushed by `sync_history()` -- looked up by row when the History
        # list's selection changes.
        self._history_results: list[AnalysisResult] = []

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

        self.remove_fit_curve_button = QPushButton("Remove Fit Curve from Plot")
        self.remove_fit_curve_button.setEnabled(False)

        self.added_feedback_label = QLabel("")
        self.added_feedback_label.setWordWrap(True)
        self.added_feedback_label.setVisible(False)

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
        fit_layout.addWidget(self.remove_fit_curve_button)
        fit_layout.addWidget(self.added_feedback_label)

        self.fit_section = CollapsibleSection("Curve Fitting", fit_group)

        self.history_list = QListWidget()
        self.history_status_label = QLabel(_HISTORY_EMPTY_TEXT)
        self.history_status_label.setWordWrap(True)

        history_group = QGroupBox("Analysis History")
        history_layout = QVBoxLayout(history_group)
        history_layout.addWidget(self.history_status_label)
        history_layout.addWidget(self.history_list)

        self.history_section = CollapsibleSection("Analysis History", history_group)

        layout = QVBoxLayout(self)
        layout.addWidget(self.fit_section)
        layout.addWidget(self.history_section)
        layout.addStretch(1)

        self.model_combo.currentIndexChanged.connect(self._update_model_controls)
        self.model_combo.currentIndexChanged.connect(self._invalidate_pending_fit)
        self.source_combo.currentIndexChanged.connect(self._invalidate_pending_fit)
        self.run_fit_button.clicked.connect(self._on_run_fit_clicked)
        self.add_fit_curve_button.clicked.connect(self._on_add_fit_curve_clicked)
        self.remove_fit_curve_button.clicked.connect(self._on_remove_fit_curve_clicked)
        self.history_list.currentRowChanged.connect(self._on_history_row_changed)

        self._update_model_controls()
        self.refresh()

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint this panel at a different `GnoviFigure` (e.g. a
        Workbench switch) and reload from it. Also clears any pending fit,
        the History list, and the Add/Remove Fit Curve target -- all
        meaningless once `figure` no longer matches the panel they were
        computed against; `MainWindow` supplies the new figure's own
        history right after, via `sync_history`."""
        self._figure = figure
        self._current_result = None
        self._history_results = []
        self.history_list.blockSignals(True)
        self.history_list.clear()
        self.history_list.blockSignals(False)
        self.history_status_label.setVisible(True)
        self._invalidate_pending_fit()  # also refreshes Add/Remove button state
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
        """Clear a pending fit result -- called whenever the source series
        or model selection changes, so a fit for one series/model can
        never be added to the plot under a different one's name. A degree
        change or a re-run that fails does *not* invalidate an
        already-successful pending fit -- it's still a genuine,
        self-consistent result for the series/model it was run against.

        Note this clears the fit even if it was already successfully
        added to the plot once -- that's correct: the *pending* fit
        (what a fresh "Run Fit" would report as "Ready to add") is tied
        to the current source/model selection, independent of whether a
        copy of it is already on the canvas. Never touches
        `_current_result`/the History selection -- those describe which
        *result* Add/Remove Fit Curve act on, unrelated to what the
        source/model controls are set up to run next."""
        self._pending_fit = None
        self.pending_fit_label.clear()
        self.pending_fit_label.setVisible(False)
        self.added_feedback_label.clear()
        self.added_feedback_label.setVisible(False)
        self._refresh_fit_curve_buttons()

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
                source_dataset_name=series.dataset.name,
                source_series_id=series.id,
                source_series_label=series.label,
                x_column=series.x_column,
                y_column=series.y_column,
                row_range=series.row_range,
                source_panel_id=self._figure.active_panel.id,
                degree=self.degree_spin.value(),
            )
        except FitError as exc:
            QMessageBox.critical(self, "Curve Fitting", str(exc))
            return

        self._pending_fit = result
        self.pending_fit_label.setText(f"Ready to add: {result.summary()}")
        self.pending_fit_label.setVisible(True)
        # A fresh fit hasn't been added yet -- any "Added to plot: ..."
        # feedback from a previous fit no longer describes this one. A
        # brand new FitResult always has a brand new result_id, so it can
        # never already have a matching curve -- Add enabled, Remove not.
        self.added_feedback_label.clear()
        self.added_feedback_label.setVisible(False)
        self._refresh_fit_curve_buttons()

        self.analysis_result_ready.emit(result)

    def _resolve_curve_range(self, result: FitResult) -> tuple[float, float, int] | None:
        """The `(x_min, x_max, num_points)` to resample `result`'s fit
        curve across, for "Add Fit Curve to Plot" -- prefers `result`'s
        own stored `curve_x_min`/`curve_x_max`/`curve_num_points` (see
        that field's docstring: exactly what was fitted, captured at fit
        time), so regenerating a curve for a historical result -- one
        whose derived `PlotSeries` was removed, possibly after a project
        reload -- reproduces what was actually plotted, not whatever the
        source data's range happens to be *now*. Only a `FitResult`
        persisted before that field existed lacks it; for that case only,
        falls back to the source's current live data range via
        `resolve_live_xy`. Returns `None` if neither is available (source
        dataset/series no longer exists) -- the caller must fail
        gracefully, never fabricate a range."""
        if result.curve_x_min is not None and result.curve_x_max is not None and result.curve_num_points is not None:
            return result.curve_x_min, result.curve_x_max, result.curve_num_points
        xy = resolve_live_xy(self._figure, self._manager, result)
        if xy is None:
            return None
        x, _y = xy
        return float(x.min()), float(x.max()), DEFAULT_CURVE_SAMPLES

    def _add_target(self) -> FitResult | None:
        """The result "Add Fit Curve to Plot" would act on right now:
        `_current_result` (an explicit History selection, or whatever a
        just-run fit synchronously became current as, once wired to
        MainWindow's `analysis_result_ready` -> `sync_history` round
        trip) if set, else `_pending_fit` -- the session-local, not-yet-
        selected-anywhere fit a bare `AnalysisPanel` (no MainWindow
        wiring at all, e.g. in isolation tests) can still add on its own.
        Never the reverse preference: an explicit History selection must
        always win over a stale pending fit for a *different* result."""
        return self._current_result if self._current_result is not None else self._pending_fit

    def _on_add_fit_curve_clicked(self) -> None:
        result = self._add_target()
        if result is None:
            return  # button is disabled in this state; defensive no-op only
        if self._matching_series(result):
            return  # defensive -- button should already be disabled (one curve per result_id)

        curve_range = self._resolve_curve_range(result)
        if curve_range is None:
            QMessageBox.warning(
                self,
                "Curve Fitting",
                "Can't regenerate this fit curve: its source dataset/series no longer exists "
                "and this result has no stored fit-time range to fall back on.",
            )
            return
        x_min, x_max, num_points = curve_range
        x_smooth, y_smooth = sample_fit_curve(result, x_min, x_max, num_points=num_points)

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
        self.added_feedback_label.setText(f"Added to plot: {series.label}")
        self.added_feedback_label.setVisible(True)
        self.add_to_plot_requested.emit([series])
        # `emit()` synchronously runs the connected handler (MainWindow adds
        # the series to this same live `self._figure`), so re-checking here
        # already sees it -- Add becomes disabled, Remove enabled. Relied on
        # again, redundantly but harmlessly, via MainWindow's own
        # `_on_figure_content_changed` -> `sync_history` chain.
        self._refresh_fit_curve_buttons()

    def _on_remove_fit_curve_clicked(self) -> None:
        if not self._matched_fit_curve_series_ids:
            return  # button is disabled in this state; defensive no-op only
        self.remove_fit_curve_requested.emit(list(self._matched_fit_curve_series_ids))
        self._refresh_fit_curve_buttons()  # see _on_add_fit_curve_clicked's own note

    @staticmethod
    def _history_item_labels(results: list[AnalysisResult]) -> list[str]:
        """One compact, human-readable label per entry in `results`
        (oldest first, same order as `PanelResultHistory.all()`) --
        `"<Model> fit — <y column>"` for a `FitResult`, `summary()` for
        any other future `AnalysisResult` subclass. Never a raw
        `result_id`/UUID.

        Repeated fits of the same model against the same column would
        otherwise render as visually identical rows -- disambiguated with
        a stable `" · #N"` ordinal (2nd, 3rd, ... occurrence of that same
        label; the 1st is left unsuffixed), counted by position in this
        same oldest-first order so it never changes as long as nothing is
        deleted. No timestamp: `AnalysisResult` doesn't record one, and
        this is purely a display disambiguator -- it changes no
        provenance field."""

        def base_label(result: AnalysisResult) -> str:
            if isinstance(result, FitResult):
                return f"{result.model.capitalize()} fit — {result.y_column}"
            return result.summary()

        seen: dict[str, int] = {}
        labels = []
        for result in results:
            label = base_label(result)
            seen[label] = seen.get(label, 0) + 1
            labels.append(f"{label} · #{seen[label]}" if seen[label] > 1 else label)
        return labels

    def sync_history(self, results: list[AnalysisResult], current: AnalysisResult | None) -> None:
        """Called by `MainWindow` whenever the active panel's analysis
        history changes -- a fresh fit, a panel/Workbench switch,
        undo/redo, or a project load -- to repopulate the Analysis
        History list and repoint Add/Remove Fit Curve at whichever result
        is current (`results` is `core.workbench.Workbench.
        analysis_results.all()`; `current` is whatever `AnalysisResultView`
        is showing right now -- always the *same object* MainWindow just
        passed to `show_result()`, never re-derived here).

        `current` sets `_current_result` directly -- it is deliberately
        NOT required to be a member of `results`. A `FitResult` with
        `source_panel_id=None` is never added to any panel's history (see
        `PanelResultHistory`/`Workbench.analysis_results`), so it can
        never appear in `results`, yet it can still be `current` (Results
        is showing it) -- Add/Remove Fit Curve must still target that
        exact result, never silently fall back to a stale, unrelated
        `_pending_fit` just because history has nothing recorded for it.
        Precedence is unconditional: an explicit `current` always wins;
        `_pending_fit` is only ever consulted by `_add_target()` when
        `current` is `None` (see that method's own docstring).

        Rebuilds the list with signals blocked so this programmatic
        repopulation never re-emits `history_result_selected` (that
        signal is for an actual click only; see `_on_history_row_changed`)."""
        self._history_results = list(results)
        current_result_id = current.result_id if current is not None else None

        self.history_list.blockSignals(True)
        self.history_list.clear()
        target_row = -1
        for i, (result, label) in enumerate(zip(results, self._history_item_labels(results))):
            item = QListWidgetItem(label)
            self.history_list.addItem(item)
            if result.result_id == current_result_id:
                target_row = i
        self.history_list.setCurrentRow(target_row)
        self.history_list.blockSignals(False)

        self.history_status_label.setVisible(not results)
        self.history_list.setVisible(bool(results))

        self._current_result = current if isinstance(current, FitResult) else None
        self._refresh_fit_curve_buttons()

    def _on_history_row_changed(self, row: int) -> None:
        """The scientist selected a different Analysis History entry --
        never fired for `sync_history`'s own programmatic rebuild (see
        its `blockSignals` guard). `row == -1` (nothing selected, e.g. an
        empty history) is a no-op: there's nothing to select or emit."""
        if row < 0 or row >= len(self._history_results):
            return
        result = self._history_results[row]
        self._current_result = result if isinstance(result, FitResult) else None
        # Reselecting a different result makes any "Ready to add: ..."/
        # "Added to plot: ..." label from a still-pending fresh fit stale
        # -- it described *that* fit, not whatever's selected now.
        self.pending_fit_label.clear()
        self.pending_fit_label.setVisible(False)
        self.added_feedback_label.clear()
        self.added_feedback_label.setVisible(False)
        self._refresh_fit_curve_buttons()
        self.history_result_selected.emit(result)

    def _matching_series(self, result: FitResult) -> list[PlotSeries]:
        """Every `PlotSeries` in the *active* panel whose derived Dataset
        traces back to `result` -- matched by the stable `result_id`
        stored in `Dataset.metadata` (see `_on_add_fit_curve_clicked`),
        never by label (editable, non-unique) or any other heuristic."""
        return [
            s
            for s in self._figure.active_panel.series
            if s.dataset.metadata.get("result_id") == result.result_id
        ]

    def _refresh_fit_curve_buttons(self) -> None:
        """Recompute Add/Remove Fit Curve enable state from scratch --
        idempotent, safe to call as often as needed. 'Remove' always acts
        on `_current_result` specifically (an explicit History selection,
        or whatever MainWindow's `sync_history` last reported) -- it's
        meaningless for a merely-pending, not-yet-selected-anywhere fit,
        since nothing has told this panel it's the one to remove yet
        (mirrors the pre-History "must be told externally" contract).
        'Add' acts on `_add_target()` (falls back to `_pending_fit` when
        there's no History selection at all, e.g. a bare `AnalysisPanel`
        with no MainWindow wired up). Invariant: one `FitResult.result_id`
        maps to at most one generated `PlotSeries` in its panel, so
        whichever single result is being considered is always exactly one
        of Add/Remove enabled, the other not (see the class docstring)."""
        current_matches = self._matching_series(self._current_result) if self._current_result else []
        self._matched_fit_curve_series_ids = [s.id for s in current_matches]
        self.remove_fit_curve_button.setEnabled(bool(current_matches))

        add_target = self._add_target()
        add_matches = self._matching_series(add_target) if add_target else []
        self.add_fit_curve_button.setEnabled(add_target is not None and not add_matches)
