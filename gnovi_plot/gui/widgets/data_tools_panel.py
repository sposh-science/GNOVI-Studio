from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
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


class DataToolsPanel(QWidget):
    """Right-side companion to the Data Preview table: calculated columns,
    non-destructive row-selection operations, and transformation history for
    the currently selected Dataset.

    Every operation here mutates only `Dataset.dataframe` (the working
    data); the raw imported DataFrame is never touched. This panel knows
    nothing about the plot/series layer -- it emits `transformation_applied`
    after each successful operation so the owner can refresh dependent UI
    (Data Preview model, plot column selectors, PlotSeries staleness).
    """

    transformation_applied = Signal(object, bool)  # Dataset, row_set_changed

    def __init__(self, preview_table: QTableView, parent=None):
        super().__init__(parent)
        self._preview_table = preview_table
        self._dataset: Dataset | None = None

        self.row_count_label = QLabel("No dataset selected")
        self.calculated_columns_label = QLabel("")
        self.calculated_columns_label.setWordWrap(True)

        self.calculated_column_button = QPushButton("Calculated Column…")
        self.calculated_column_button.setProperty("primary", True)
        self.exclude_button = QPushButton("Exclude Selection")
        self.keep_button = QPushButton("Keep Selection")
        self.reset_button = QPushButton("Reset Working Data")

        self.history_list = QListWidget()

        status_group = QGroupBox("Working Data")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.row_count_label)
        status_layout.addWidget(self.calculated_columns_label)

        tools_group = QGroupBox("Data Tools")
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.addWidget(self.calculated_column_button)
        selection_row = QHBoxLayout()
        selection_row.addWidget(self.exclude_button)
        selection_row.addWidget(self.keep_button)
        tools_layout.addLayout(selection_row)
        tools_layout.addWidget(self.reset_button)

        history_group = QGroupBox("Transformation History")
        history_layout = QVBoxLayout(history_group)
        history_layout.addWidget(self.history_list)

        self.status_section = CollapsibleSection("Working Data", status_group)
        self.tools_section = CollapsibleSection("Data Tools", tools_group)
        self.history_section = CollapsibleSection("Transformation History", history_group)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_section)
        layout.addWidget(self.tools_section)
        layout.addWidget(self.history_section, 1)

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

    def _on_exclude_selection(self) -> None:
        self._apply_row_operation("Exclude Selected Rows", self._dataset.exclude_rows if self._dataset else None)

    def _on_keep_selection(self) -> None:
        self._apply_row_operation("Keep Only Selected Rows", self._dataset.keep_rows if self._dataset else None)

    def _apply_row_operation(self, title: str, operation) -> None:
        if self._dataset is None or operation is None:
            return
        positions = self._selected_row_positions()
        if not positions:
            QMessageBox.warning(self, title, "Select at least one row in the Data Preview first.")
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
