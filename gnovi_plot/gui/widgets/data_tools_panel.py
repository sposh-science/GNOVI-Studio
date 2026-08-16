from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.equations.parser import FormulaError
from gnovi_plot.gui.dialogs.calculated_column_dialog import CalculatedColumnDialog
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection

# Shown before every destructive Working Data operation (Keep/Exclude
# Selected Rows) -- these mutate `Dataset.dataframe` and can invalidate
# existing PlotSeries whose `row_range` no longer applies (see
# `Panel.invalidate_series_for_dataset`), unlike "Plot Selected Rows" which
# never touches Working Data at all.
_ROW_OPERATION_CONFIRMATION = (
    "This operation changes the Working Data and may invalidate plots that "
    "depend on the current row structure. Existing plotted data may need to "
    "be re-added."
)


class DataToolsPanel(QWidget):
    """Right-side companion to the Data tab's table (see `gui.bottom_panel`):
    a plot-only row selection action, calculated columns, non-destructive
    Working Data transformations, and transformation history for the
    currently selected Dataset.

    Every operation in the "Working Data" / "Data Tools" groups mutates only
    `Dataset.dataframe` (the working data); the raw imported DataFrame is
    never touched. "Plot Selected Rows" is deliberately kept separate: it
    never touches `Dataset.dataframe` at all (raw or working) and never adds
    a transformation-history entry -- it only asks the owner to add a new
    PlotSeries scoped to the current row selection via `row_range`, so
    existing panels/series are completely unaffected. This panel knows
    nothing about the plot/series layer -- it emits `transformation_applied`
    after each Working Data operation and `plot_selected_rows_requested`
    for the plot-only action, so the owner can refresh dependent UI
    (Data tab model, plot column selectors, PlotSeries staleness).

    The Transformation History list (`history_group`) is built here --
    where the history is actually tracked and refreshed -- but is not added
    to this panel's own layout; the owner relocates it into the bottom
    panel's "Transformations" tab (see `gui.widgets.bottom_panel`), keeping
    the quick action buttons a user reaches for while looking at the plot
    separate from a log they glance at occasionally.
    """

    transformation_applied = Signal(object, bool)  # Dataset, row_set_changed
    plot_selected_rows_requested = Signal(list)  # list[int] selected row positions

    def __init__(self, preview_table: QTableView, parent=None):
        super().__init__(parent)
        self._preview_table = preview_table
        self._dataset: Dataset | None = None

        self.row_count_label = QLabel("No dataset selected")
        self.calculated_columns_label = QLabel("")
        self.calculated_columns_label.setWordWrap(True)

        self.plot_selected_rows_button = QPushButton("Plot Selected Rows")
        self.plot_selected_rows_button.setProperty("primary", True)

        self.calculated_column_button = QPushButton("Calculated Column…")
        self.calculated_column_button.setProperty("primary", True)
        # "...from/in Working Data" dropped from the label text -- the
        # enclosing "Working Data Actions" group (and the drawer's own
        # "Working" header) already say so; the full sentence no longer fit
        # the right drawer's compact width without clipping (see
        # `MainWindow`'s `_right_drawer_min_width` note on the drawer's own
        # responsive floor, which now guarantees these buttons -- at their
        # shorter natural width -- can never clip regardless).
        self.exclude_button = QPushButton("Exclude Selected Rows")
        self.keep_button = QPushButton("Keep Selected Rows")
        self.reset_button = QPushButton("Reset Working Data")

        self.history_list = QListWidget()

        # "Row Selection", not "Data Preview Actions": the Bottom panel's
        # tab hosting this table is just named "Data" (see gui.bottom_panel)
        # -- "Data Preview" isn't UI terminology used anywhere else, so
        # keeping it here was a stale, redundant label with nothing left to
        # be consistent with.
        row_selection_group = QGroupBox("Row Selection")
        row_selection_layout = QVBoxLayout(row_selection_group)
        row_selection_layout.addWidget(self.plot_selected_rows_button)

        status_group = QGroupBox("Working Data")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.row_count_label)
        status_layout.addWidget(self.calculated_columns_label)

        tools_group = QGroupBox("Working Data Actions")
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.addWidget(self.calculated_column_button)
        # Stacked, not a side-by-side row: the right Working Data drawer is
        # narrow (~21% of window width), and these two buttons' combined
        # label width (~540px) never fits it -- with the drawer's scroll
        # area horizontal scrollbar disabled (see main_window._wrap_scrollable),
        # a side-by-side row clipped most of both buttons with no way to
        # reach the clipped part.
        tools_layout.addWidget(self.exclude_button)
        tools_layout.addWidget(self.keep_button)
        tools_layout.addWidget(self.reset_button)

        # Built here (where the history is tracked) but not added to this
        # panel's own layout -- see the class docstring.
        self.history_group = QGroupBox("Transformation History")
        history_layout = QVBoxLayout(self.history_group)
        history_layout.addWidget(self.history_list)

        self.row_selection_section = CollapsibleSection("Row Selection", row_selection_group)
        self.status_section = CollapsibleSection("Working Data", status_group)
        self.tools_section = CollapsibleSection("Working Data Actions", tools_group)

        layout = QVBoxLayout(self)
        layout.addWidget(self.row_selection_section)
        layout.addWidget(self.status_section)
        layout.addWidget(self.tools_section)
        layout.addStretch(1)

        self.plot_selected_rows_button.clicked.connect(self._on_plot_selected_rows_clicked)
        self.calculated_column_button.clicked.connect(self._on_calculated_column)
        self.exclude_button.clicked.connect(self._on_exclude_selection)
        self.keep_button.clicked.connect(self._on_keep_selection)
        self.reset_button.clicked.connect(self._on_reset)

        self.set_dataset(None)

    def set_dataset(self, dataset: Dataset | None) -> None:
        self._dataset = dataset
        self._refresh_status()

    def _refresh_status(self) -> None:
        has_dataset = self._dataset is not None
        for widget in (
            self.plot_selected_rows_button,
            self.calculated_column_button,
            self.exclude_button,
            self.keep_button,
            self.reset_button,
        ):
            widget.setEnabled(has_dataset)

        if not has_dataset:
            self.row_count_label.setText("No dataset selected")
            self.calculated_columns_label.clear()
            self.history_list.clear()
            return

        dataset = self._dataset
        self.row_count_label.setText(f"Working Data: {dataset.row_count} / {dataset.raw_row_count} rows")

        if dataset.calculated_columns:
            self.calculated_columns_label.setText(
                f"Calculated columns: {', '.join(dataset.calculated_columns)}"
            )
        else:
            self.calculated_columns_label.setText("Calculated columns: (none)")

        self.history_list.clear()
        for i, transformation in enumerate(dataset.transformations, start=1):
            self.history_list.addItem(QListWidgetItem(f"{i}. {transformation.description}"))
        self.history_list.scrollToBottom()

    def _selected_row_positions(self) -> list[int]:
        selection_model = self._preview_table.selectionModel()
        if selection_model is None:
            return []
        return sorted({index.row() for index in selection_model.selectedIndexes()})

    def _on_calculated_column(self) -> None:
        if self._dataset is None:
            return
        dialog = CalculatedColumnDialog(self._dataset.dataframe, self._dataset.columns, self)
        if dialog.exec() != CalculatedColumnDialog.Accepted:
            return
        try:
            self._dataset.add_calculated_column(dialog.name, dialog.formula)
        except (ValueError, FormulaError) as exc:
            QMessageBox.critical(self, "Calculated Column", str(exc))
            return
        self._refresh_status()
        self.transformation_applied.emit(self._dataset, False)

    def _on_plot_selected_rows_clicked(self) -> None:
        if self._dataset is None:
            return
        positions = self._selected_row_positions()
        if len(positions) < 2:
            QMessageBox.warning(
                self, "Plot Selected Rows", "Select at least 2 rows in the Data tab first."
            )
            return
        self.plot_selected_rows_requested.emit(positions)

    def _on_exclude_selection(self) -> None:
        self._apply_row_operation("Exclude Selected Rows", self._dataset.exclude_rows if self._dataset else None)

    def _on_keep_selection(self) -> None:
        self._apply_row_operation("Keep Only Selected Rows", self._dataset.keep_rows if self._dataset else None)

    def _apply_row_operation(self, title: str, operation) -> None:
        if self._dataset is None or operation is None:
            return
        positions = self._selected_row_positions()
        if not positions:
            QMessageBox.warning(self, title, "Select at least one row in the Data tab first.")
            return
        confirmed = QMessageBox.question(
            self,
            title,
            _ROW_OPERATION_CONFIRMATION,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            return
        try:
            operation(positions)
        except ValueError as exc:
            QMessageBox.critical(self, title, str(exc))
            return
        self._refresh_status()
        self.transformation_applied.emit(self._dataset, True)

    def _on_reset(self) -> None:
        if self._dataset is None:
            return
        self._dataset.reset_working_data()
        self._refresh_status()
        self.transformation_applied.emit(self._dataset, True)
