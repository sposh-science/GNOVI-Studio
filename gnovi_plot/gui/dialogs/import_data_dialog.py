from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.data.importers.text_importer import (
    DELIMITER_OPTIONS,
    DataImportError,
    ImportResult,
    detect_table_start,
    import_table,
    read_raw_lines,
    resolve_delimiter,
)
from gnovi_plot.gui.widgets.dataframe_table_model import DataFrameTableModel

_RAW_PREVIEW_LINES = 500
_PARSED_PREVIEW_ROWS = 200


class ImportDataDialog(QDialog):
    """Preview a raw data file and let the user pin down the delimiter, header
    row, and data-start row before it becomes a Dataset.

    Needed because scientific instrument exports (e.g. cyclic voltammetry CSVs)
    often place free-form metadata above the real column-header row; row 0 is
    not a safe assumption for the header.
    """

    def __init__(self, path: str | Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle(f"Import Data - {self.path.name}")
        self.resize(900, 700)

        self._lines = read_raw_lines(self.path, max_lines=_RAW_PREVIEW_LINES)
        self.result: ImportResult | None = None

        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems(DELIMITER_OPTIONS)

        self.header_row_spin = QSpinBox()
        self.header_row_spin.setMinimum(0)
        self.header_row_spin.setMaximum(max(len(self._lines) - 1, 0))

        self.data_start_row_spin = QSpinBox()
        self.data_start_row_spin.setMinimum(0)
        self.data_start_row_spin.setMaximum(len(self._lines))

        self.raw_table = QTableWidget()
        self.raw_table.setColumnCount(1)
        self.raw_table.setHorizontalHeaderLabels(["Raw line"])
        self.raw_table.setRowCount(len(self._lines))
        self.raw_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.raw_table.setSelectionBehavior(QTableWidget.SelectRows)
        for i, line in enumerate(self._lines):
            self.raw_table.setItem(i, 0, QTableWidgetItem(line))
        self.raw_table.horizontalHeader().setStretchLastSection(True)

        self.parsed_table = QTableView()
        self.parsed_model = DataFrameTableModel()
        self.parsed_table.setModel(self.parsed_model)
        self.parsed_table.setEditTriggers(QTableView.NoEditTriggers)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Delimiter:"))
        controls.addWidget(self.delimiter_combo)
        controls.addWidget(QLabel("Header row:"))
        controls.addWidget(self.header_row_spin)
        controls.addWidget(QLabel("Data start row:"))
        controls.addWidget(self.data_start_row_spin)
        controls.addStretch(1)

        splitter = QSplitter(Qt.Vertical)

        raw_container = QWidget()
        raw_layout = QVBoxLayout(raw_container)
        raw_layout.addWidget(QLabel("Raw file preview (click a row to use it as the header row)"))
        raw_layout.addWidget(self.raw_table)
        splitter.addWidget(raw_container)

        parsed_container = QWidget()
        parsed_layout = QVBoxLayout(parsed_container)
        parsed_layout.addWidget(QLabel("Parsed preview"))
        parsed_layout.addWidget(self.parsed_table)
        splitter.addWidget(parsed_container)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(splitter)
        layout.addWidget(self.status_label)
        layout.addWidget(self.button_box)

        self.delimiter_combo.currentTextChanged.connect(self._refresh_preview)
        self.header_row_spin.valueChanged.connect(self._on_header_row_changed)
        self.data_start_row_spin.valueChanged.connect(self._refresh_preview)
        self.raw_table.currentCellChanged.connect(self._on_raw_row_clicked)

        self._auto_detect_and_apply()

    def _auto_detect_and_apply(self) -> None:
        delimiter = resolve_delimiter(self._lines, "auto")
        header_row, data_start_row = detect_table_start(self._lines, delimiter)
        self.header_row_spin.blockSignals(True)
        self.data_start_row_spin.blockSignals(True)
        self.header_row_spin.setValue(header_row)
        self.data_start_row_spin.setValue(data_start_row)
        self.header_row_spin.blockSignals(False)
        self.data_start_row_spin.blockSignals(False)
        self._refresh_preview()

    def _on_header_row_changed(self, value: int) -> None:
        # Keep data start row valid (after the header) unless the user has
        # already pushed it further down manually.
        if self.data_start_row_spin.value() <= value:
            self.data_start_row_spin.blockSignals(True)
            self.data_start_row_spin.setValue(value + 1)
            self.data_start_row_spin.blockSignals(False)
        self._refresh_preview()

    def _on_raw_row_clicked(self, row: int, *_args) -> None:
        if row < 0:
            return
        self.header_row_spin.setValue(row)

    def _delimiter_option(self) -> str:
        return self.delimiter_combo.currentText().lower()

    def _refresh_preview(self, *_args) -> None:
        try:
            result = import_table(
                self.path,
                delimiter_option=self._delimiter_option(),
                header_row=self.header_row_spin.value(),
                data_start_row=self.data_start_row_spin.value(),
            )
        except DataImportError as exc:
            self.result = None
            self.parsed_model.set_dataframe(None)
            self.status_label.setText(str(exc))
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            return

        self.result = result
        self.parsed_model.set_dataframe(result.dataframe.head(_PARSED_PREVIEW_ROWS))
        self.status_label.setText(
            f"{result.dataframe.shape[0]} rows x {result.dataframe.shape[1]} columns detected."
        )
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

    def _on_accept(self) -> None:
        if self.result is None:
            return
        self.accept()
