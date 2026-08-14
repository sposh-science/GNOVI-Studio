from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.dialogs.import_data_dialog import ImportDataDialog

_FILE_FILTER = "Data files (*.csv *.txt *.tsv *.dat);;All files (*)"


class DatasetPanel(QWidget):
    """Left-side panel: dataset list/import/remove plus X/Y column selection."""

    dataset_selected = Signal(object)  # Dataset | None
    add_to_plot_requested = Signal(object, str, str)  # Dataset, x_col, y_col
    clear_plot_requested = Signal()

    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self._manager = dataset_manager

        self.dataset_list = QListWidget()
        self.import_button = QPushButton("Import Data")
        self.remove_button = QPushButton("Remove Dataset")
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.add_to_plot_button = QPushButton("Add to Plot")
        self.clear_plot_button = QPushButton("Clear Plot")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Datasets"))
        layout.addWidget(self.dataset_list)

        dataset_buttons = QHBoxLayout()
        dataset_buttons.addWidget(self.import_button)
        dataset_buttons.addWidget(self.remove_button)
        layout.addLayout(dataset_buttons)

        layout.addWidget(QLabel("X column"))
        layout.addWidget(self.x_combo)
        layout.addWidget(QLabel("Y column"))
        layout.addWidget(self.y_combo)
        layout.addWidget(self.add_to_plot_button)
        layout.addWidget(self.clear_plot_button)
        layout.addStretch(1)

        self.import_button.clicked.connect(self._on_import_clicked)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.dataset_list.currentItemChanged.connect(self._on_selection_changed)
        self.add_to_plot_button.clicked.connect(self._on_add_to_plot_clicked)
        self.clear_plot_button.clicked.connect(self.clear_plot_requested)

    def _on_import_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Data", "", _FILE_FILTER)
        if not path:
            return

        dialog = ImportDataDialog(path, self)
        if dialog.exec() != ImportDataDialog.Accepted or dialog.result is None:
            return

        result = dialog.result
        dataset = Dataset(
            name=Path(path).stem,
            dataframe=result.dataframe,
            source_path=path,
            metadata={
                "raw_header_lines": result.raw_header_lines,
                "delimiter": result.delimiter,
                "header_row": result.header_row,
                "data_start_row": result.data_start_row,
            },
        )
        self._manager.add(dataset)
        self._refresh_list(select_id=dataset.id)

    def _on_remove_clicked(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            return
        self._manager.remove(dataset.id)
        self._refresh_list()

    def _refresh_list(self, select_id: str | None = None) -> None:
        self.dataset_list.blockSignals(True)
        self.dataset_list.clear()
        target_item = None
        for dataset in self._manager.datasets:
            item = QListWidgetItem(dataset.name)
            item.setData(Qt.UserRole, dataset.id)
            self.dataset_list.addItem(item)
            if select_id is not None and dataset.id == select_id:
                target_item = item
        self.dataset_list.blockSignals(False)

        if target_item is not None:
            self.dataset_list.setCurrentItem(target_item)
        else:
            self.dataset_list.setCurrentRow(-1)
            self._on_selection_changed(None, None)

    def _current_dataset(self) -> Dataset | None:
        item = self.dataset_list.currentItem()
        if item is None:
            return None
        return self._manager.get(item.data(Qt.UserRole))

    def _on_selection_changed(self, current, previous) -> None:
        dataset = self._current_dataset()
        self._populate_columns(dataset)
        self.dataset_selected.emit(dataset)

    def _populate_columns(self, dataset: Dataset | None) -> None:
        self.x_combo.clear()
        self.y_combo.clear()
        if dataset is None:
            return
        columns = [str(c) for c in dataset.columns]
        self.x_combo.addItems(columns)
        self.y_combo.addItems(columns)
        if len(columns) > 1:
            self.y_combo.setCurrentIndex(1)

    def _on_add_to_plot_clicked(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            QMessageBox.warning(self, "No Dataset Selected", "Select a dataset to plot.")
            return

        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "No Columns Selected", "Select X and Y columns to plot.")
            return

        self.add_to_plot_requested.emit(dataset, x_col, y_col)
