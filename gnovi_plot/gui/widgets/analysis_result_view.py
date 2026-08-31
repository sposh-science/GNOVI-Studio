from __future__ import annotations

from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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

# A bounded height for the detail table (see `AnalysisResult.detail_table`).
# `QTableWidget` (a `QAbstractScrollArea`) already scrolls its rows
# internally with the header pinned, and its own `sizeHint`/
# `minimumSizeHint` do NOT scale with row count -- this cap only keeps a
# large result (thousands of XRD peak candidates) from making the table
# tall enough to crowd the compact `details()` summary above it or the
# central splitter around it. Row count never drives layout here.
_DETAIL_TABLE_MAX_HEIGHT = 260


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

    When the shown result provides a `detail_table()` (see
    `AnalysisResult.detail_table`), a bounded, internally-scrolling
    `QTableWidget` is rendered below the compact `details()` summary -- for
    XRD this is the one authoritative detailed peak table (moved here out
    of the left `XRDAnalysisSection` sidebar, which was too narrow for it).
    Row selection there is forwarded via `detail_selection_changed` so the
    left sidebar's own peak actions (Remove Selected, Enable/Disable) act
    on exactly what's selected in this table; the view itself never mutates
    a result.
    """

    # Selected row indices in the detail table changed -- `list[int]`,
    # ascending, empty when nothing is selected or the current result has
    # no detail table. `MainWindow` forwards this to `AnalysisPanel` so the
    # XRD sidebar's Remove Selected / Enable-Disable act on this selection
    # (see `gui.widgets.xrd_analysis_section.XRDAnalysisSection.
    # set_selected_peak_rows`).
    detail_selection_changed = Signal(list)

    def __init__(self, figure: GnoviFigure, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)

        self._figure = figure
        self._manager = dataset_manager
        self._result: AnalysisResult | None = None
        self._residual_window: ResidualWindow | None = None
        # `result_id` of whatever was last shown -- lets `show_result`
        # preserve the detail-table row selection across an in-place edit
        # of the same result (manual peak add/enable-disable re-displays
        # it) while clearing it when a genuinely different result is shown.
        self._shown_result_id: str | None = None
        # Guard so programmatic table repopulation/reselection in
        # `show_result` never re-emits `detail_selection_changed`.
        self._suppress_detail_selection_signal = False

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

        self._detail_table_label = QLabel()
        self._detail_table_label.setStyleSheet("font-weight: 600;")
        self._detail_table = QTableWidget(0, 0)
        self._detail_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._detail_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._detail_table.verticalHeader().setVisible(False)
        self._detail_table.horizontalHeader().setStretchLastSection(True)
        self._detail_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Bounded height + its own internal row scrolling with the header
        # pinned -- see `_DETAIL_TABLE_MAX_HEIGHT`.
        self._detail_table.setMaximumHeight(_DETAIL_TABLE_MAX_HEIGHT)
        self._detail_table.setMinimumHeight(120)
        self._detail_table.itemSelectionChanged.connect(self._on_detail_selection_changed)

        self._provenance_widget = QWidget()
        self._provenance_form = QFormLayout(self._provenance_widget)
        self._provenance_form.setContentsMargins(0, 0, 0, 0)
        self._provenance_section = CollapsibleSection("Provenance", self._provenance_widget, expanded=False)

        self._view_residuals_button = QPushButton("View Residuals…")
        self._copy_summary_button = QPushButton("Copy Summary")

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
        content_layout.addWidget(self._detail_table_label)
        content_layout.addWidget(self._detail_table)
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

        self._rebuild_detail_table(result)

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
        self._shown_result_id = None
        self._clear_detail_table()
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

    # --- detail table (optional wide row-per-record view) ------------------

    def selected_detail_rows(self) -> list[int]:
        """Ascending row indices currently selected in the detail table --
        empty when there's no table or nothing selected."""
        return sorted({index.row() for index in self._detail_table.selectionModel().selectedRows()})

    def _clear_detail_table(self) -> None:
        self._suppress_detail_selection_signal = True
        try:
            self._detail_table.clearSelection()
            self._detail_table.setRowCount(0)
            self._detail_table.setColumnCount(0)
        finally:
            self._suppress_detail_selection_signal = False
        self._detail_table.setVisible(False)
        self._detail_table_label.setVisible(False)

    def _rebuild_detail_table(self, result: AnalysisResult) -> None:
        """Populate the detail table from `result.detail_table()`, or hide
        it entirely when the result type has none (e.g. a `FitResult`).

        Selection is preserved by row index only across an in-place edit of
        the *same* result (its `result_id` is unchanged) -- a manual XRD
        peak add/enable-disable re-displays the same result and the
        researcher's row selection should survive it. A genuinely different
        result (History switch, panel/Workbench switch) always starts with
        nothing selected, and `detail_selection_changed` is emitted so the
        left sidebar drops any now-stale selection too."""
        table = result.detail_table()
        same_result = result.result_id == self._shown_result_id
        previous_rows = self.selected_detail_rows() if same_result else []
        self._shown_result_id = result.result_id

        if table is None:
            self._clear_detail_table()
            if not same_result:
                self.detail_selection_changed.emit([])
            return

        columns, rows = table
        self._suppress_detail_selection_signal = True
        try:
            self._detail_table.clearSelection()
            self._detail_table.setColumnCount(len(columns))
            self._detail_table.setHorizontalHeaderLabels(columns)
            self._detail_table.setRowCount(len(rows))
            for r, row_values in enumerate(rows):
                for c, value in enumerate(row_values):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._detail_table.setItem(r, c, item)
            self._detail_table.horizontalHeader().setStretchLastSection(True)
            # Size columns to content only for a modest table -- a
            # `ResizeToContents` header mode (or `resizeColumnsToContents`)
            # scans every row on each rebuild, and this table is rebuilt on
            # every in-place edit; skip it once the peak list is large.
            if len(rows) <= 250:
                self._detail_table.resizeColumnsToContents()
            self._reselect_detail_rows([row for row in previous_rows if row < len(rows)])
        finally:
            self._suppress_detail_selection_signal = False

        self._detail_table_label.setText(result.detail_table_title())
        self._detail_table_label.setVisible(True)
        self._detail_table.setVisible(True)

        current_rows = self.selected_detail_rows()
        if current_rows != previous_rows:
            self.detail_selection_changed.emit(current_rows)

    def _reselect_detail_rows(self, rows: list[int]) -> None:
        """Re-highlight `rows` (already validated against the row count) --
        one whole-row range per index, so a multi-row selection survives a
        same-result rebuild, not just the last row (`selectRow` in a loop
        would replace under the table's default extended-selection mode)."""
        if not rows:
            return
        model = self._detail_table.model()
        last_col = max(self._detail_table.columnCount() - 1, 0)
        selection = QItemSelection()
        for row in rows:
            selection.select(model.index(row, 0), model.index(row, last_col))
        self._detail_table.selectionModel().select(
            selection, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        )

    def _on_detail_selection_changed(self) -> None:
        if self._suppress_detail_selection_signal:
            return
        self.detail_selection_changed.emit(self.selected_detail_rows())

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

        x, y = xy
        x_arr, y_arr = x.to_numpy(), y.to_numpy()
        residual_range = result.residual_x_range()
        if residual_range is not None:
            lo, hi = residual_range
            mask = (x_arr >= lo) & (x_arr <= hi)
            x_arr, y_arr = x_arr[mask], y_arr[mask]
            if x_arr.size == 0:
                self._residuals_unavailable_label.setVisible(True)
                if self._residual_window is not None:
                    self._residual_window.hide()
                return

        self._residuals_unavailable_label.setVisible(False)
        residual_data = result.compute_residuals(x_arr, y_arr)
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
