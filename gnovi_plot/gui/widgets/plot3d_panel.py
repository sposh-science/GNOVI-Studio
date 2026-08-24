from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, group_row_positions
from gnovi_plot.gui.styles import STALE_COLOR
from gnovi_plot.gui.widgets.active_panel_label import ActivePanelLabel
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.graph_library import GraphLibrary
from gnovi_plot.plotting.series3d import Plot3DType, Series3D

_PLOT_TYPE_OPTIONS = [
    ("Scatter", Plot3DType.SCATTER),
    ("Line", Plot3DType.LINE),
    ("Line + Markers", Plot3DType.LINE_MARKER),
]

# No marker for a pure LINE series (matches `PlotSeries.line()`'s own
# `overrides.setdefault("marker", "")` convention); scatter/line+markers
# both get a real marker so a point is actually visible.
_DEFAULT_MARKER_BY_PLOT_TYPE = {
    Plot3DType.SCATTER: "o",
    Plot3DType.LINE: "",
    Plot3DType.LINE_MARKER: "o",
}

_GROUP_BY_NONE = "__none__"


def _format_group_label(value: object) -> str:
    """Plain value label for a "Group by" group -- e.g. `25.0` -> `"25"`,
    `25.5` -> `"25.5"`, a categorical string used as-is. No unit is ever
    invented from the group column's name (GNOVI has no trusted unit
    metadata system to draw one from) -- see this milestone's own notes."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class Plot3DPanel(QWidget):
    """Left-side "3D" drawer page: the creation workspace for `Panel3D`
    content -- Dataset + plot type + X/Y/Z column selection, optionally
    split into a "Group by" curve family, to add new `Series3D` to the
    active panel.

    Deliberately does NOT duplicate per-series styling OR a series list --
    both stay on the Series tab's adaptive 3D page (see `plot_series_panel.
    PlotSeriesPanel`), exactly mirroring how 2D's own "Add to Plot"
    (`DatasetPanel.plot_section`) creates series while `PlotSeriesPanel`
    lists and styles them, never duplicating the split. This page used to
    also carry its own read-only summary list of the active panel's 3D
    series; it was removed (see PR "Sidebar Navigation & 2D/3D Workflow
    Polish"'s own audit) once the adaptive Series page's `series3d_list`
    was confirmed to fully cover selecting/renaming/styling/removing/
    clearing every `Series3D` this page can create -- the removed list
    offered no selection, editing, or per-item removal of its own (no
    `currentRowChanged` connection existed on it), so there was no
    workflow only available through it. "Clear 3D Plot" below is kept
    (its own quick-clear counterpart to 2D's "Clear Plot") -- Series still
    also offers "Remove Series"/"Clear All" for finer-grained management.

    This panel never decides whether "Add to 3D Plot" should convert the
    active panel, append to it, or ask for confirmation first -- that
    decision needs `GnoviFigure.active_panel`'s current type/content, which
    only the owner (`MainWindow`) resolves against together with the rest
    of the application's state. This panel only validates the Dataset/X/Y/Z
    (and, if set, Group by) choice is numerically usable (`data.numeric.
    group_row_positions`, the same controlled-error convention `numeric_xy`/
    `numeric_xyz` already use) and emits a fully-formed list of `Series3D`
    (no color yet -- see `Panel3D.add_series`, which assigns one from the
    theme cycle) as ONE signal -- "Group by Temperature" creating 7 series
    is one atomic Add operation, not 7 (see `MainWindow.
    _on_add_3d_series_requested`, which commits exactly one undo checkpoint
    for the whole list).
    """

    add_3d_series_requested = Signal(list)  # list[Series3D]
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
        for text, plot_type in _PLOT_TYPE_OPTIONS:
            self.plot_type_combo.addItem(text, plot_type)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.z_combo = QComboBox()

        self.group_by_combo = QComboBox()

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
        add_layout.addWidget(QLabel("Group by"))
        add_layout.addWidget(self.group_by_combo)
        add_layout.addWidget(self.error_label)
        add_layout.addWidget(self.add_button)
        add_layout.addWidget(self.clear_button)

        self.add_section = CollapsibleSection("Add 3D Series", add_group)

        layout = QVBoxLayout(self)
        layout.addWidget(self.active_panel_label)
        layout.addWidget(self.add_section)
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

        current_group = self.group_by_combo.currentData()
        self.group_by_combo.blockSignals(True)
        self.group_by_combo.clear()
        self.group_by_combo.addItem("None", _GROUP_BY_NONE)
        for column in columns:
            self.group_by_combo.addItem(column, column)
        if current_group is not None:
            index = self.group_by_combo.findData(current_group)
            if index >= 0:
                self.group_by_combo.setCurrentIndex(index)
        self.group_by_combo.blockSignals(False)

    def refresh(self) -> None:
        """Reload the active-panel context and "Clear 3D Plot"'s enabled
        state (see this class's own docstring for why there's no series
        list here to reload alongside it -- that's the Series page's job).
        The "Add 3D Series" form itself stays enabled regardless of the
        active panel's current type -- it's how an empty or 2D panel
        becomes a `Panel3D` in the first place (see `MainWindow.
        _on_add_3d_series_requested`)."""
        self.active_panel_label.refresh(self._figure)
        panel = self._figure.active_panel
        has_series = isinstance(panel, Panel3D) and bool(panel.series)
        self.clear_button.setEnabled(has_series)

    def _on_add_clicked(self) -> None:
        self.error_label.setVisible(False)
        dataset = self._current_dataset()
        x_col, y_col, z_col = self.x_combo.currentText(), self.y_combo.currentText(), self.z_combo.currentText()
        if dataset is None or not x_col or not y_col or not z_col:
            self.error_label.setText("Choose a Dataset and X/Y/Z columns.")
            self.error_label.setVisible(True)
            return

        group_data = self.group_by_combo.currentData()
        group_col = None if group_data in (None, _GROUP_BY_NONE) else group_data
        try:
            groups = group_row_positions(dataset.dataframe, x_col, y_col, z_col, group_col)
        except (KeyError, InsufficientNumericDataError) as exc:
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            return

        # `QComboBox.currentData()` round-trips a str-subclassed Enum
        # through QVariant and hands back a plain `str` -- same Qt
        # marshalling quirk `dataset_panel._current_plot_type` already
        # documents/normalizes for 2D; without this, `Series3D.plot_type`
        # would be a plain string that happens to `==`-compare equal to the
        # right `Plot3DType` member (so rendering/tests using `==` don't
        # notice) but crashes in `to_dict()` (`.value` doesn't exist on a
        # plain str) the moment the panel is saved.
        raw_plot_type = self.plot_type_combo.currentData()
        plot_type = raw_plot_type if isinstance(raw_plot_type, Plot3DType) else Plot3DType(raw_plot_type)
        marker = _DEFAULT_MARKER_BY_PLOT_TYPE[plot_type]

        series_list = []
        for group_value, positions in groups.items():
            if group_col is None:
                label = dataset.name
                row_indices = None
            else:
                label = _format_group_label(group_value)
                row_indices = tuple(positions)
            series_list.append(
                Series3D(
                    dataset=dataset,
                    x_column=x_col,
                    y_column=y_col,
                    z_column=z_col,
                    label=label,
                    plot_type=plot_type,
                    marker=marker,
                    row_indices=row_indices,
                )
            )
        self.add_3d_series_requested.emit(series_list)
