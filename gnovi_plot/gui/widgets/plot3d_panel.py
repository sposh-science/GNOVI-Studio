from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xyz
from gnovi_plot.gui.styles import STALE_COLOR
from gnovi_plot.gui.widgets.active_panel_label import ActivePanelLabel
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.graph_library import GraphLibrary
from gnovi_plot.plotting.series3d import Series3D

# Only "Scatter" this milestone -- the combo exists (rather than a plain
# label) so 3D Line/Curve Family (a later, separate PR -- see this
# milestone's own architecture inspection) has an obvious place to land as
# a second option, without another sidebar page or control ever being
# needed for it.
_PLOT_TYPE_OPTIONS = [("Scatter", "scatter")]


class Plot3DPanel(QWidget):
    """Left-side "3D" drawer page: the creation/management workspace for
    `Panel3D` content -- Dataset + X/Y/Z column selection to add a new
    `Series3D` to the active panel, plus a read-only summary list of the
    active panel's current 3D series.

    Deliberately does NOT duplicate per-series styling (color/marker/alpha/
    visibility) -- that stays on the Series tab's adaptive 3D page (see
    `plot_series_panel.PlotSeriesPanel`), exactly mirroring how 2D's own
    "Add to Plot" (`DatasetPanel.plot_section`) creates series while
    `PlotSeriesPanel` styles them, never duplicating the split.

    This panel never decides whether "Add to 3D Plot" should convert the
    active panel, append to it, or ask for confirmation first -- that
    decision needs `GnoviFigure.active_panel`'s current type/content, which
    only the owner (`MainWindow`) resolves against together with the rest
    of the application's state. This panel only validates the Dataset/X/Y/Z
    choice is numerically usable (`numeric_xyz`, the same controlled-error
    pattern `DatasetPanel._on_add_to_plot_clicked` already uses for 2D) and
    emits a fully-formed `Series3D` (no color yet -- see
    `Panel3D.add_series`, which assigns one from the theme cycle).
    """

    add_3d_series_requested = Signal(object)  # Series3D
    clear_3d_plot_requested = Signal()

    def __init__(
        self,
        dataset_manager: DatasetManager,
        figure: GnoviFigure,
        get_graph_library: Callable[[], GraphLibrary] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._manager = dataset_manager
        self._figure = figure

        self.active_panel_label = ActivePanelLabel(figure, get_graph_library)

        self.dataset_combo = QComboBox()
        self.plot_type_combo = QComboBox()
        for text, code in _PLOT_TYPE_OPTIONS:
            self.plot_type_combo.addItem(text, code)
        # A visible, disabled combo (not a hidden one) -- "there is exactly
        # one option right now" should read as an honest, temporary fact
        # about this milestone, not a control that mysteriously vanished
        # once a second plot type exists.
        self.plot_type_combo.setEnabled(len(_PLOT_TYPE_OPTIONS) > 1)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.z_combo = QComboBox()

        self.add_button = QPushButton("Add to 3D Plot")
        self.add_button.setProperty("primary", True)
        self.clear_button = QPushButton("Clear 3D Plot")

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {STALE_COLOR};")
        self.error_label.setVisible(False)

        add_group = QGroupBox("Add 3D Series")
        add_layout = QVBoxLayout(add_group)
        add_layout.addWidget(QLabel("Dataset"))
        add_layout.addWidget(self.dataset_combo)
        add_layout.addWidget(QLabel("Plot type"))
        add_layout.addWidget(self.plot_type_combo)
        add_layout.addWidget(QLabel("X column"))
        add_layout.addWidget(self.x_combo)
        add_layout.addWidget(QLabel("Y column"))
        add_layout.addWidget(self.y_combo)
        add_layout.addWidget(QLabel("Z column"))
        add_layout.addWidget(self.z_combo)
        add_layout.addWidget(self.error_label)
        add_layout.addWidget(self.add_button)
        add_layout.addWidget(self.clear_button)

        self.series_list = QListWidget()

        list_group = QGroupBox("3D Series")
        list_layout = QVBoxLayout(list_group)
        list_layout.addWidget(self.series_list)

        self.add_section = CollapsibleSection("Add 3D Series", add_group)
        self.list_section = CollapsibleSection("3D Series", list_group)

        layout = QVBoxLayout(self)
        layout.addWidget(self.active_panel_label)
        layout.addWidget(self.add_section)
        layout.addWidget(self.list_section)
        layout.addStretch(1)

        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        self.add_button.clicked.connect(self._on_add_clicked)
        self.clear_button.clicked.connect(self.clear_3d_plot_requested)

        self._refresh_dataset_combo()
        self.refresh()

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint this panel at a different `GnoviFigure` (e.g. after
        Open/New Project swaps the active project) and reload from it."""
        self._figure = figure
        self.refresh()

    def set_manager(self, dataset_manager: DatasetManager) -> None:
        """Repoint this panel at a different `DatasetManager` (e.g. after
        Open/New Project swaps the active project's dataset set)."""
        self._manager = dataset_manager
        self._refresh_dataset_combo()

    def _refresh_dataset_combo(self) -> None:
        current_id = self.dataset_combo.currentData()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for dataset in self._manager.datasets:
            self.dataset_combo.addItem(dataset.name, dataset.id)
        if current_id is not None:
            index = self.dataset_combo.findData(current_id)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        self.dataset_combo.blockSignals(False)
        self._on_dataset_changed(self.dataset_combo.currentIndex())

    def _current_dataset(self):
        dataset_id = self.dataset_combo.currentData()
        return self._manager.get(dataset_id) if dataset_id else None

    def _on_dataset_changed(self, _index: int) -> None:
        dataset = self._current_dataset()
        columns = [str(c) for c in dataset.columns] if dataset is not None else []
        for combo in (self.x_combo, self.y_combo, self.z_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            if current in columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def refresh(self) -> None:
        """Reload the active panel's 3D series list and "Clear 3D Plot"'s
        enabled state. The "Add 3D Series" form itself stays enabled
        regardless of the active panel's current type -- it's how an empty
        or 2D panel becomes a `Panel3D` in the first place (see
        `MainWindow._on_add_3d_series_requested`)."""
        self.active_panel_label.refresh(self._figure)
        panel = self._figure.active_panel
        series_list = panel.series if isinstance(panel, Panel3D) else []

        self.series_list.blockSignals(True)
        self.series_list.clear()
        for series in series_list:
            item = QListWidgetItem(self._item_text(series))
            if series.stale:
                item.setForeground(QColor(STALE_COLOR))
            self.series_list.addItem(item)
        self.series_list.blockSignals(False)

        self.clear_button.setEnabled(bool(series_list))

    @staticmethod
    def _item_text(series: Series3D) -> str:
        label = series.label or "(untitled series)"
        return f"{label}  [stale — re-add]" if series.stale else label

    def _on_add_clicked(self) -> None:
        self.error_label.setVisible(False)
        dataset = self._current_dataset()
        x_col, y_col, z_col = self.x_combo.currentText(), self.y_combo.currentText(), self.z_combo.currentText()
        if dataset is None or not x_col or not y_col or not z_col:
            self.error_label.setText("Choose a Dataset and X/Y/Z columns.")
            self.error_label.setVisible(True)
            return
        try:
            numeric_xyz(dataset.dataframe, x_col, y_col, z_col)
        except (KeyError, InsufficientNumericDataError) as exc:
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            return

        series = Series3D(dataset=dataset, x_column=x_col, y_column=y_col, z_column=z_col, label=dataset.name)
        self.add_3d_series_requested.emit(series)
