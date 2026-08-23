from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gnovi_plot.analysis.results import AnalysisResult
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.gui.widgets.residual_window import ResidualWindow
from gnovi_plot.plotting.figure import GnoviFigure

_EMPTY_STATE_TEXT = (
    "No results yet.\n\n"
    "Run an analysis -- curve fitting, and (as they're added) statistics, "
    "peak analysis, FFT, smoothing, or domain-specific tools -- to see its "
    "output here."
)

_RESIDUALS_UNAVAILABLE_TEXT = "Residuals unavailable -- the source dataset/series no longer exists."


def resolve_live_xy(figure: GnoviFigure | None, manager: DatasetManager | None, result: AnalysisResult):
    """The source series'/dataset's *current* numeric (x, y) data for
    `result` -- `None` if nothing resolves any more (dataset/series
    removed) or the data isn't numeric. Shared by `AnalysisResultView`
    (on-demand residual computation, see `_show_residuals_for`) and
    `gui.widgets.analysis_panel.AnalysisPanel` (regenerating a fit curve
    for a result whose own stored `curve_x_min`/`curve_x_max` are
    unavailable, see `AnalysisPanel._resolve_curve_range`) -- both need
    exactly this same live resolution, never a frozen snapshot."""
    series = None
    if result.source_series_id is not None and figure is not None:
        series = figure.get_series(result.source_series_id)

    if series is not None:
        dataframe = series.dataframe
    else:
        dataset = manager.get(result.source_dataset_id) if manager is not None else None
        if dataset is None:
            return None
        dataframe = dataset.dataframe
        if result.row_range is not None:
            start, end = result.row_range
            dataframe = dataframe.iloc[start:end]

    try:
        return numeric_xy(dataframe, result.x_column, result.y_column)
    except (KeyError, InsufficientNumericDataError):
        return None


class AnalysisResultView(QWidget):
    """Read-only display for the most recently produced `AnalysisResult`.

    Renders any `AnalysisResult` through its `summary()`/`details()`/
    `provenance_details()`/`supports_residuals()`/`compute_residuals()`/
    `report_text()` contract only -- never imports or references
    `FitResult` or any other concrete subclass. Adding a new analysis tool
    (statistics, peak analysis, FFT, smoothing, a domain-specific module)
    only ever means giving it its own `AnalysisResult` subclass; this view
    does not change.

    Holds `figure`/`dataset_manager` references (mirrors `AnalysisPanel`'s
    own `set_figure`/`set_manager` pattern) for two purposes only, both
    read-only:
      - Resolving a friendly dataset/series *name* when the result's own
        stored snapshot (`source_dataset_name`/`source_series_label`) is
        missing -- e.g. an older result. The stored snapshot is always
        preferred; this is a fallback, not the primary path.
      - Resolving the *live* (x, y) data needed to compute residuals on
        demand (never persisted -- recomputed fresh each time "View
        Residuals..." is clicked, or an already-open residual window is
        refreshed for a new result).

    Residual diagnostics are never embedded here -- they display in a
    single reusable `ResidualWindow` (a non-modal, independently
    resizable top-level window; see that class), created lazily on first
    use and kept alive for this view's whole lifetime so repeated clicks
    or successive fits reuse one window instead of accumulating them. If
    that window is open when a *new* result is shown and the new result
    doesn't support residuals or its source can no longer be resolved,
    the window is hidden (never left showing stale diagnostics) but the
    instance itself is kept for later reuse.

    Shows a single result at a time (the most recent), plus its own empty
    state when nothing has been shown yet -- there is no history list in
    this milestone.
    """

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)

        self._figure = figure
        self._manager = dataset_manager
        self._result: AnalysisResult | None = None
        self._residual_window: ResidualWindow | None = None

        self._empty_label = QLabel(_EMPTY_STATE_TEXT)
        self._empty_label.setWordWrap(True)

        self._dataset_label = QLabel()
        self._dataset_label.setWordWrap(True)
        self._series_label = QLabel()
        self._series_label.setWordWrap(True)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-weight: 600;")

        self._details_widget = QWidget()
        self._details_form = QFormLayout(self._details_widget)
        self._details_form.setContentsMargins(0, 0, 0, 0)

        self._provenance_widget = QWidget()
        self._provenance_form = QFormLayout(self._provenance_widget)
        self._provenance_form.setContentsMargins(0, 0, 0, 0)
        self._provenance_section = CollapsibleSection("Provenance", self._provenance_widget, expanded=False)

        self._view_residuals_button = QPushButton("View Residuals…")
        self._copy_summary_button = QPushButton("Copy Fit Summary")

        button_row = QHBoxLayout()
        button_row.addWidget(self._view_residuals_button)
        button_row.addWidget(self._copy_summary_button)
        button_row.addStretch(1)

        self._residuals_unavailable_label = QLabel(_RESIDUALS_UNAVAILABLE_TEXT)
        self._residuals_unavailable_label.setWordWrap(True)
        self._residuals_unavailable_label.setVisible(False)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._dataset_label)
        content_layout.addWidget(self._series_label)
        content_layout.addWidget(self._summary_label)
        content_layout.addWidget(self._details_widget)
        content_layout.addWidget(self._provenance_section)
        content_layout.addLayout(button_row)
        content_layout.addWidget(self._residuals_unavailable_label)
        content_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._empty_label)
        layout.addWidget(self._content)

        self._view_residuals_button.clicked.connect(self._on_view_residuals_clicked)
        self._copy_summary_button.clicked.connect(self._on_copy_summary_clicked)

        self.clear()

    @property
    def result(self) -> AnalysisResult | None:
        """The result currently shown, or `None` in the empty state."""
        return self._result

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint at a different `GnoviFigure` (e.g. a Workbench switch)
        -- only affects series-label/residual-data resolution, never the
        currently-shown result itself."""
        self._figure = figure

    def set_manager(self, dataset_manager: DatasetManager) -> None:
        """Repoint at a different `DatasetManager` (Open/New Project) --
        only affects dataset-name/residual-data resolution."""
        self._manager = dataset_manager

    def show_result(self, result: AnalysisResult) -> None:
        """Display `result`, replacing whatever was shown before."""
        self._result = result

        dataset_name = self._resolve_dataset_name(result)
        self._dataset_label.setText(f"Dataset: {dataset_name or result.source_dataset_id}")

        series_label = self._resolve_series_label(result)
        self._series_label.setVisible(result.source_series_id is not None)
        if result.source_series_id is not None:
            self._series_label.setText(f"Series: {series_label or result.source_series_id}")

        self._summary_label.setText(result.summary())
        self._rebuild_form(self._details_form, result.details())
        self._rebuild_form(self._provenance_form, result.provenance_details())
        self._provenance_section.set_expanded(False)

        self._view_residuals_button.setVisible(result.supports_residuals())
        self._residuals_unavailable_label.setVisible(False)

        # If a residual window from a previous fit is currently open,
        # update it in place for this (new) result rather than leaving it
        # showing stale diagnostics -- see class docstring. A closed/hidden
        # window is left alone; its content is recomputed lazily on the
        # next "View Residuals..." click.
        if self._residual_window is not None and self._residual_window.isVisible():
            if result.supports_residuals():
                self._show_residuals_for(result)
            else:
                self._residual_window.hide()

        self._empty_label.setVisible(False)
        self._content.setVisible(True)

    def clear(self) -> None:
        """Return to the empty state -- no result shown."""
        self._result = None
        self._dataset_label.clear()
        self._series_label.clear()
        self._summary_label.clear()
        self._rebuild_form(self._details_form, [])
        self._rebuild_form(self._provenance_form, [])
        self._view_residuals_button.setVisible(False)
        self._residuals_unavailable_label.setVisible(False)
        if self._residual_window is not None:
            self._residual_window.hide()
        self._empty_label.setVisible(True)
        self._content.setVisible(False)

    def _rebuild_form(self, form: QFormLayout, rows: list[tuple[str, str]]) -> None:
        while form.rowCount():
            form.removeRow(0)
        for label, value in rows:
            form.addRow(f"{label}:", QLabel(value))

    def _resolve_dataset_name(self, result: AnalysisResult) -> str | None:
        """A genuine friendly name, preferring the result's own stored
        snapshot over live lookup -- `None` (never a raw id) if nothing
        resolves, so callers can each decide their own id-fallback
        behavior (the on-screen label shows the id; the copy summary
        omits the line instead, per "don't make ids prominent there")."""
        if result.source_dataset_name:
            return result.source_dataset_name
        dataset = self._manager.get(result.source_dataset_id) if self._manager is not None else None
        return dataset.name if dataset is not None else None

    def _resolve_series_label(self, result: AnalysisResult) -> str | None:
        if result.source_series_id is None:
            return None
        if result.source_series_label:
            return result.source_series_label
        series = self._figure.get_series(result.source_series_id) if self._figure is not None else None
        return series.label if series is not None else None

    def _resolve_live_xy(self, result: AnalysisResult):
        """The source series'/dataset's *current* numeric (x, y) data, for
        on-demand residual computation -- see the shared `resolve_live_xy`
        module function."""
        return resolve_live_xy(self._figure, self._manager, result)

    def _on_view_residuals_clicked(self) -> None:
        if self._result is not None:
            self._show_residuals_for(self._result)

    def _show_residuals_for(self, result: AnalysisResult) -> None:
        """Compute `result`'s residuals from current live data and display
        them in the reusable `ResidualWindow`, creating it on first use.
        If the source can no longer be resolved, hides any open window
        (without destroying it) and shows the inline unavailable message
        instead -- never opens a window with nothing to show."""
        xy = self._resolve_live_xy(result)
        if xy is None:
            self._residuals_unavailable_label.setVisible(True)
            if self._residual_window is not None:
                self._residual_window.hide()
            return

        self._residuals_unavailable_label.setVisible(False)
        x, y = xy
        residual_data = result.compute_residuals(x.to_numpy(), y.to_numpy())
        if self._residual_window is None:
            self._residual_window = ResidualWindow(self)
        self._residual_window.show_residuals(
            residual_data,
            x_label=result.x_column,
            y_label=f"Residual ({result.y_column})",
            title=self._residual_window_title(result),
        )

    def _residual_window_title(self, result: AnalysisResult) -> str:
        series_label = self._resolve_series_label(result)
        subtitle = result.residual_window_subtitle()
        if series_label:
            return f"Residuals — {subtitle} — {series_label}"
        return f"Residuals — {subtitle}"

    def _on_copy_summary_clicked(self) -> None:
        if self._result is None:
            return
        text = self._result.report_text(
            dataset_name=self._resolve_dataset_name(self._result),
            series_label=self._resolve_series_label(self._result),
        )
        QGuiApplication.clipboard().setText(text)
