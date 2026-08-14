from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.analysis.cycles import CycleDetectionError, detect_cycles
from gnovi_plot.analysis.segments import (
    InvalidRowRangeError,
    OverlappingRowRangeError,
    RowRangeCollection,
)
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_column, numeric_xy
from gnovi_plot.gui.dialogs.import_data_dialog import ImportDataDialog
from gnovi_plot.plotting.series import PlotSeries, PlotType

_FILE_FILTER = "Data files (*.csv *.txt *.tsv *.dat);;All files (*)"

_AUTO_DETECTION_FAILED_MESSAGE = (
    "Automatic cycle detection could not identify repeating sweeps. "
    "Select cycle ranges manually from the data table."
)

_PLOT_TYPE_OPTIONS = [
    ("Line", PlotType.LINE),
    ("Scatter", PlotType.SCATTER),
    ("Histogram", PlotType.HISTOGRAM),
]

_PLOT_MODE_ENTIRE = "entire"
_PLOT_MODE_CYCLES = "cycles"

_PLOT_MODE_OPTIONS = [
    ("Entire dataset", _PLOT_MODE_ENTIRE),
    ("Split into cycles", _PLOT_MODE_CYCLES),
]

_CYCLE_SOURCE_AUTO = "auto"
_CYCLE_SOURCE_MANUAL = "manual"

_CYCLE_SOURCE_OPTIONS = [
    ("Auto detect", _CYCLE_SOURCE_AUTO),
    ("Select from data table", _CYCLE_SOURCE_MANUAL),
]


class DatasetPanel(QWidget):
    """Left-side panel: dataset list/import/remove plus plot-type and column
    selection for adding a new PlotSeries to the plot."""

    dataset_selected = Signal(object)  # Dataset | None
    add_to_plot_requested = Signal(list)  # list[PlotSeries]
    clear_plot_requested = Signal()

    def __init__(self, dataset_manager: DatasetManager, preview_table: QTableView, parent=None):
        super().__init__(parent)
        self._manager = dataset_manager
        self._preview_table = preview_table
        self._manual_ranges: RowRangeCollection | None = None

        self.dataset_list = QListWidget()
        self.import_button = QPushButton("Import Data")
        self.remove_button = QPushButton("Remove Dataset")

        self.plot_type_combo = QComboBox()
        for text, plot_type in _PLOT_TYPE_OPTIONS:
            self.plot_type_combo.addItem(text, plot_type)

        self.x_label = QLabel("X column")
        self.x_combo = QComboBox()
        self.y_label = QLabel("Y column")
        self.y_combo = QComboBox()
        self.bins_label = QLabel("Bins")
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(0, 1000)
        self.bins_spin.setValue(0)
        self.bins_spin.setSpecialValueText("Auto")

        self.plot_mode_label = QLabel("Plot mode")
        self.plot_mode_combo = QComboBox()
        for text, mode in _PLOT_MODE_OPTIONS:
            self.plot_mode_combo.addItem(text, mode)
        self.cycle_status_label = QLabel("")
        self.cycle_status_label.setWordWrap(True)

        self.cycle_source_label = QLabel("Cycle source")
        self.cycle_source_combo = QComboBox()
        for text, source in _CYCLE_SOURCE_OPTIONS:
            self.cycle_source_combo.addItem(text, source)

        self.manual_cycle_list = QListWidget()
        self.add_cycle_button = QPushButton("Add Cycle from Selection")
        self.remove_cycle_button = QPushButton("Remove Selected Cycle")
        self.clear_cycles_button = QPushButton("Clear Manual Cycles")
        self.add_manual_cycles_button = QPushButton("Add Manual Cycles to Plot")

        self.add_to_plot_button = QPushButton("Add to Plot")
        self.clear_plot_button = QPushButton("Clear Plot")

        dataset_group = QGroupBox("Datasets")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.addWidget(self.dataset_list)
        dataset_buttons = QHBoxLayout()
        dataset_buttons.addWidget(self.import_button)
        dataset_buttons.addWidget(self.remove_button)
        dataset_layout.addLayout(dataset_buttons)

        plot_group = QGroupBox("Add to Plot")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.addWidget(QLabel("Plot type"))
        plot_layout.addWidget(self.plot_type_combo)
        plot_layout.addWidget(self.x_label)
        plot_layout.addWidget(self.x_combo)
        plot_layout.addWidget(self.y_label)
        plot_layout.addWidget(self.y_combo)
        plot_layout.addWidget(self.bins_label)
        plot_layout.addWidget(self.bins_spin)
        plot_layout.addWidget(self.plot_mode_label)
        plot_layout.addWidget(self.plot_mode_combo)
        plot_layout.addWidget(self.cycle_status_label)
        plot_layout.addWidget(self.cycle_source_label)
        plot_layout.addWidget(self.cycle_source_combo)

        manual_cycle_group = QGroupBox("Manual Cycles")
        manual_cycle_layout = QVBoxLayout(manual_cycle_group)
        manual_cycle_layout.addWidget(self.manual_cycle_list)
        manual_cycle_layout.addWidget(self.add_cycle_button)
        manual_cycle_layout.addWidget(self.remove_cycle_button)
        manual_cycle_layout.addWidget(self.clear_cycles_button)
        manual_cycle_layout.addWidget(self.add_manual_cycles_button)
        self.manual_cycle_group = manual_cycle_group
        plot_layout.addWidget(manual_cycle_group)

        plot_layout.addWidget(self.add_to_plot_button)
        plot_layout.addWidget(self.clear_plot_button)

        layout = QVBoxLayout(self)
        layout.addWidget(dataset_group)
        layout.addWidget(plot_group)
        layout.addStretch(1)

        self.import_button.clicked.connect(self._on_import_clicked)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.dataset_list.currentItemChanged.connect(self._on_selection_changed)
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)
        self.plot_mode_combo.currentIndexChanged.connect(self._update_cycle_preview)
        self.plot_mode_combo.currentIndexChanged.connect(self._update_manual_cycle_visibility)
        self.cycle_source_combo.currentIndexChanged.connect(self._update_manual_cycle_visibility)
        self.x_combo.currentIndexChanged.connect(self._update_cycle_preview)
        self.add_to_plot_button.clicked.connect(self._on_add_to_plot_clicked)
        self.clear_plot_button.clicked.connect(self.clear_plot_requested)
        self.add_cycle_button.clicked.connect(self._on_add_cycle_from_selection)
        self.remove_cycle_button.clicked.connect(self._on_remove_selected_cycle)
        self.clear_cycles_button.clicked.connect(self._on_clear_manual_cycles)
        self.add_manual_cycles_button.clicked.connect(self._on_add_manual_cycles_to_plot)

        self._on_plot_type_changed()
        self._reset_manual_cycles()

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

    def _current_plot_type(self) -> PlotType:
        return self.plot_type_combo.currentData()

    def _on_selection_changed(self, current, previous) -> None:
        dataset = self._current_dataset()
        self._populate_columns(dataset)
        self._reset_manual_cycles()
        self.dataset_selected.emit(dataset)
        self._update_cycle_preview()

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

    def _on_plot_type_changed(self, *_args) -> None:
        is_histogram = self._current_plot_type() == PlotType.HISTOGRAM
        is_line = self._current_plot_type() == PlotType.LINE
        self.y_label.setVisible(not is_histogram)
        self.y_combo.setVisible(not is_histogram)
        self.bins_label.setVisible(is_histogram)
        self.bins_spin.setVisible(is_histogram)
        self.plot_mode_label.setVisible(is_line)
        self.plot_mode_combo.setVisible(is_line)
        self._update_manual_cycle_visibility()

    def _is_cycle_split_mode(self) -> bool:
        return (
            self._current_plot_type() == PlotType.LINE
            and self.plot_mode_combo.currentData() == _PLOT_MODE_CYCLES
        )

    def _cycle_source(self) -> str:
        return self.cycle_source_combo.currentData()

    def _is_manual_cycle_mode(self) -> bool:
        return self._is_cycle_split_mode() and self._cycle_source() == _CYCLE_SOURCE_MANUAL

    def _update_manual_cycle_visibility(self, *_args) -> None:
        is_cycles = self._is_cycle_split_mode()
        manual = self._is_manual_cycle_mode()
        self.cycle_source_label.setVisible(is_cycles)
        self.cycle_source_combo.setVisible(is_cycles)
        self.manual_cycle_group.setVisible(manual)
        self.cycle_status_label.setVisible(is_cycles and not manual)
        self.add_to_plot_button.setVisible(not manual)
        self._update_cycle_preview()

    def _update_cycle_preview(self, *_args) -> None:
        if not self._is_cycle_split_mode() or self._cycle_source() == _CYCLE_SOURCE_MANUAL:
            self.cycle_status_label.clear()
            return

        dataset = self._current_dataset()
        x_col = self.x_combo.currentText()
        if dataset is None or not x_col:
            self.cycle_status_label.clear()
            return

        try:
            cycles = detect_cycles(dataset.dataframe, x_col)
        except (CycleDetectionError, KeyError):
            self.cycle_status_label.setText(_AUTO_DETECTION_FAILED_MESSAGE)
            return
        self.cycle_status_label.setText(f"Detected {len(cycles)} cycle(s) in '{x_col}'.")

    # --- Manual cycle selection (data-table driven) -----------------------

    def _reset_manual_cycles(self) -> None:
        dataset = self._current_dataset()
        self._manual_ranges = RowRangeCollection(dataset.row_count) if dataset is not None else None
        self._refresh_manual_cycle_list()

    def _refresh_manual_cycle_list(self) -> None:
        self.manual_cycle_list.clear()
        if self._manual_ranges is None:
            return
        for i, (start, end) in enumerate(self._manual_ranges.ranges):
            points = end - start
            text = f"Cycle {i + 1} — rows {start}–{end - 1} — {points} points"
            self.manual_cycle_list.addItem(QListWidgetItem(text))

    def _selected_table_row_positions(self) -> list[int]:
        selection_model = self._preview_table.selectionModel()
        if selection_model is None:
            return []
        return sorted({index.row() for index in selection_model.selectedIndexes()})

    def _on_add_cycle_from_selection(self) -> None:
        if self._manual_ranges is None:
            QMessageBox.warning(self, "No Dataset Selected", "Select a dataset to record a manual cycle.")
            return

        positions = self._selected_table_row_positions()
        try:
            self._manual_ranges.add_from_positions(positions)
        except (InvalidRowRangeError, OverlappingRowRangeError) as exc:
            QMessageBox.warning(self, "Manual Cycle", str(exc))
            return
        self._refresh_manual_cycle_list()

    def _on_remove_selected_cycle(self) -> None:
        if self._manual_ranges is None:
            return
        row = self.manual_cycle_list.currentRow()
        if row < 0:
            return
        self._manual_ranges.remove_at(row)
        self._refresh_manual_cycle_list()

    def _on_clear_manual_cycles(self) -> None:
        if self._manual_ranges is None:
            return
        self._manual_ranges.clear()
        self._refresh_manual_cycle_list()

    def _on_add_manual_cycles_to_plot(self) -> None:
        dataset = self._current_dataset()
        if dataset is None or self._manual_ranges is None:
            QMessageBox.warning(self, "No Dataset Selected", "Select a dataset to plot.")
            return

        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "No Columns Selected", "Select X and Y columns to plot.")
            return

        if not self._manual_ranges.ranges:
            QMessageBox.warning(
                self, "No Manual Cycles", "Add at least one manual cycle from the data table first."
            )
            return

        try:
            numeric_xy(dataset.dataframe, x_col, y_col)
        except (KeyError, InsufficientNumericDataError) as exc:
            QMessageBox.critical(self, "Plot Error", str(exc))
            return

        series_list = [
            PlotSeries.line(
                dataset,
                x_col,
                y_col,
                label=f"{dataset.name} — Cycle {i + 1}",
                row_range=row_range,
            )
            for i, row_range in enumerate(self._manual_ranges.ranges)
        ]
        self.add_to_plot_requested.emit(series_list)

    def _on_add_to_plot_clicked(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            QMessageBox.warning(self, "No Dataset Selected", "Select a dataset to plot.")
            return

        plot_type = self._current_plot_type()
        x_col = self.x_combo.currentText()
        if not x_col:
            QMessageBox.warning(self, "No Column Selected", "Select a column to plot.")
            return

        try:
            if plot_type == PlotType.HISTOGRAM:
                numeric_column(dataset.dataframe, x_col)
                bins = "auto" if self.bins_spin.value() == 0 else self.bins_spin.value()
                series_list = [PlotSeries.histogram(dataset, x_col, bins=bins)]
            else:
                y_col = self.y_combo.currentText()
                if not y_col:
                    QMessageBox.warning(self, "No Columns Selected", "Select X and Y columns to plot.")
                    return
                numeric_xy(dataset.dataframe, x_col, y_col)

                if self._is_cycle_split_mode():
                    try:
                        cycles = detect_cycles(dataset.dataframe, x_col)
                    except CycleDetectionError:
                        QMessageBox.warning(self, "Cycle Detection", _AUTO_DETECTION_FAILED_MESSAGE)
                        return
                    series_list = [
                        PlotSeries.line(
                            dataset,
                            x_col,
                            y_col,
                            label=f"{dataset.name} — Cycle {i + 1}",
                            row_range=row_range,
                        )
                        for i, row_range in enumerate(cycles)
                    ]
                    self.cycle_status_label.setText(f"Detected {len(cycles)} cycle(s) in '{x_col}'.")
                else:
                    factory = PlotSeries.line if plot_type == PlotType.LINE else PlotSeries.scatter
                    series_list = [factory(dataset, x_col, y_col)]
        except KeyError as exc:
            QMessageBox.critical(self, "Plot Error", f"Column not found: {exc}")
            return
        except InsufficientNumericDataError as exc:
            QMessageBox.critical(self, "Plot Error", str(exc))
            return

        self.add_to_plot_requested.emit(series_list)
