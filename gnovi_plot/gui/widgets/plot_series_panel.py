from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.gui.styles import STALE_COLOR
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries, PlotType
from gnovi_plot.plotting.stacking import auto_stack_offsets, reset_offsets

_LINE_STYLE_OPTIONS = [
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dotted", ":"),
    ("Dash-dot", "-."),
]

_MARKER_OPTIONS = [
    ("None", ""),
    ("Circle", "o"),
    ("Square", "s"),
    ("Triangle", "^"),
    ("Diamond", "D"),
    ("Plus", "+"),
    ("Cross", "x"),
    ("Star", "*"),
]

_HIST_MODE_OPTIONS = [
    ("Frequency", "frequency"),
    ("Percentage", "percentage"),
    ("Cumulative", "cumulative"),
]

_DEFAULT_COLOR = "#1f77b4"
_OFFSET_RANGE = 1e9


class PlotSeriesPanel(QWidget):
    """Lists every PlotSeries on the currently active Panel and edits the
    selected one, plus panel-wide stacked/offset controls.

    Editing a series mutates only that PlotSeries instance; nothing else on
    the figure is touched. `changed` is emitted after every mutation so the
    owner can re-render the plot. Since `figure.series`/`add_series`/etc.
    delegate to `figure.active_panel`, this panel automatically follows
    whichever panel is active -- call `refresh()` again after switching
    panels to reload the list for the new one.
    """

    changed = Signal()

    def __init__(self, figure: GnoviFigure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._updating = False

        self.series_list = QListWidget()
        self.remove_button = QPushButton("Remove Series")
        self.clear_button = QPushButton("Clear All")

        self.label_edit = QLineEdit()
        self.color_button = QPushButton()
        self.color_button.setFixedWidth(48)
        self.visible_check = QCheckBox("Visible")

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 10.0)
        self.width_spin.setSingleStep(0.5)

        self.style_combo = QComboBox()
        for text, code in _LINE_STYLE_OPTIONS:
            self.style_combo.addItem(text, code)

        self.marker_combo = QComboBox()
        for text, code in _MARKER_OPTIONS:
            self.marker_combo.addItem(text, code)

        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(1.0, 30.0)

        self.marker_filled_check = QCheckBox("Filled marker")
        self.marker_edge_width_spin = QDoubleSpinBox()
        self.marker_edge_width_spin.setRange(0.1, 10.0)
        self.marker_edge_width_spin.setSingleStep(0.1)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)

        self.zorder_spin = QDoubleSpinBox()
        self.zorder_spin.setRange(-100.0, 100.0)

        self.offset_label = QLabel("Y offset")
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-_OFFSET_RANGE, _OFFSET_RANGE)
        self.offset_spin.setDecimals(4)

        self.normalize_check = QCheckBox("Normalize to max")

        self.bins_label = QLabel("Bins")
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(0, 1000)
        self.bins_spin.setSpecialValueText("Auto")
        self.hist_mode_label = QLabel("Histogram mode")
        self.hist_mode_combo = QComboBox()
        for text, code in _HIST_MODE_OPTIONS:
            self.hist_mode_combo.addItem(text, code)

        list_group = QGroupBox("Plot Series")
        list_layout = QVBoxLayout(list_group)
        list_layout.addWidget(self.series_list)
        buttons = QHBoxLayout()
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.clear_button)
        list_layout.addLayout(buttons)

        props_group = QGroupBox("Series Properties")
        form = QFormLayout(props_group)
        form.addRow("Label", self.label_edit)
        form.addRow("Color", self.color_button)
        form.addRow("Line width", self.width_spin)
        form.addRow("Line style", self.style_combo)
        form.addRow("Marker", self.marker_combo)
        form.addRow("Marker size", self.marker_size_spin)
        form.addRow(self.marker_filled_check)
        form.addRow("Marker edge width", self.marker_edge_width_spin)
        form.addRow("Transparency", self.alpha_spin)
        form.addRow("Z-order", self.zorder_spin)
        form.addRow(self.offset_label, self.offset_spin)
        form.addRow(self.normalize_check)
        form.addRow(self.bins_label, self.bins_spin)
        form.addRow(self.hist_mode_label, self.hist_mode_combo)
        form.addRow(self.visible_check)

        self.offset_step_spin = QDoubleSpinBox()
        self.offset_step_spin.setRange(0.0, _OFFSET_RANGE)
        self.offset_step_spin.setDecimals(4)
        self.offset_step_spin.setSpecialValueText("Auto")
        self.auto_stack_button = QPushButton("Auto-Stack Offsets")
        self.reset_offsets_button = QPushButton("Reset Offsets")

        stack_group = QGroupBox("Stacked / Offset Curves")
        stack_form = QFormLayout(stack_group)
        stack_form.addRow("Offset step (0 = auto)", self.offset_step_spin)
        stack_buttons = QHBoxLayout()
        stack_buttons.addWidget(self.auto_stack_button)
        stack_buttons.addWidget(self.reset_offsets_button)
        stack_form.addRow(stack_buttons)

        self.list_section = CollapsibleSection("Plot Series", list_group)
        self.props_section = CollapsibleSection("Series Properties", props_group)
        self.stack_section = CollapsibleSection("Stacked / Offset Curves", stack_group)

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_section)
        layout.addWidget(self.props_section)
        layout.addWidget(self.stack_section)
        layout.addStretch(1)

        self.series_list.currentRowChanged.connect(self._on_selection_changed)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.label_edit.editingFinished.connect(self._apply_label)
        self.color_button.clicked.connect(self._pick_color)
        self.visible_check.toggled.connect(self._apply_visible)
        self.width_spin.valueChanged.connect(self._apply_width)
        self.style_combo.currentIndexChanged.connect(self._apply_style)
        self.marker_combo.currentIndexChanged.connect(self._apply_marker)
        self.marker_size_spin.valueChanged.connect(self._apply_marker_size)
        self.marker_filled_check.toggled.connect(self._apply_marker_filled)
        self.marker_edge_width_spin.valueChanged.connect(self._apply_marker_edge_width)
        self.alpha_spin.valueChanged.connect(self._apply_alpha)
        self.zorder_spin.valueChanged.connect(self._apply_zorder)
        self.offset_spin.valueChanged.connect(self._apply_offset)
        self.normalize_check.toggled.connect(self._apply_normalize)
        self.bins_spin.valueChanged.connect(self._apply_bins)
        self.hist_mode_combo.currentIndexChanged.connect(self._apply_hist_mode)

        self.auto_stack_button.clicked.connect(self._on_auto_stack)
        self.reset_offsets_button.clicked.connect(self._on_reset_offsets)

        self.refresh()

    @staticmethod
    def _item_text(series: PlotSeries) -> str:
        return f"{series.label}  [stale — re-add]" if series.stale else series.label

    def refresh(self, select_id: str | None = None) -> None:
        self.series_list.blockSignals(True)
        self.series_list.clear()
        target_row = -1
        for i, series in enumerate(self._figure.series):
            item = QListWidgetItem(self._item_text(series))
            if series.stale:
                item.setForeground(QColor(STALE_COLOR))
            item.setData(Qt.UserRole, series.id)
            self.series_list.addItem(item)
            if select_id is not None and series.id == select_id:
                target_row = i
        self.series_list.blockSignals(False)

        if target_row >= 0:
            self.series_list.setCurrentRow(target_row)
        elif self.series_list.count() > 0:
            self.series_list.setCurrentRow(0)
        else:
            self.series_list.setCurrentRow(-1)
            self._on_selection_changed(-1)

    def _current_series(self) -> PlotSeries | None:
        item = self.series_list.currentItem()
        if item is None:
            return None
        return self._figure.get_series(item.data(Qt.UserRole))

    def _set_editors_enabled(self, enabled: bool) -> None:
        for widget in (
            self.label_edit,
            self.color_button,
            self.visible_check,
            self.width_spin,
            self.style_combo,
            self.marker_combo,
            self.marker_size_spin,
            self.marker_filled_check,
            self.marker_edge_width_spin,
            self.alpha_spin,
            self.zorder_spin,
            self.offset_spin,
            self.normalize_check,
            self.bins_spin,
            self.hist_mode_combo,
        ):
            widget.setEnabled(enabled)

    def _set_color_swatch(self, color: str | None) -> None:
        self.color_button.setStyleSheet(f"background-color: {color or _DEFAULT_COLOR};")

    def _on_selection_changed(self, row: int) -> None:
        series = self._current_series()
        self._set_editors_enabled(series is not None)
        if series is None:
            return

        self._updating = True
        self.label_edit.setText(series.label)
        self._set_color_swatch(series.color)
        self.visible_check.setChecked(series.visible)
        self.width_spin.setValue(series.line_width)
        self.style_combo.setCurrentIndex(max(self.style_combo.findData(series.line_style), 0))
        self.marker_combo.setCurrentIndex(max(self.marker_combo.findData(series.marker), 0))
        self.marker_size_spin.setValue(series.marker_size)
        self.marker_filled_check.setChecked(series.marker_filled)
        self.marker_edge_width_spin.setValue(series.marker_edge_width)
        self.alpha_spin.setValue(series.alpha)
        self.zorder_spin.setValue(series.zorder)

        is_histogram = series.plot_type == PlotType.HISTOGRAM
        self.bins_label.setVisible(is_histogram)
        self.bins_spin.setVisible(is_histogram)
        self.hist_mode_label.setVisible(is_histogram)
        self.hist_mode_combo.setVisible(is_histogram)
        if is_histogram:
            self.bins_spin.setValue(0 if series.bins == "auto" else int(series.bins))
            self.hist_mode_combo.setCurrentIndex(max(self.hist_mode_combo.findData(series.hist_mode), 0))

        self.offset_label.setVisible(not is_histogram)
        self.offset_spin.setVisible(not is_histogram)
        self.normalize_check.setVisible(not is_histogram)
        if not is_histogram:
            self.offset_spin.setValue(series.y_offset)
            self.normalize_check.setChecked(series.normalize_to_max)
        self._updating = False

    def _refresh_current_item_text(self) -> None:
        item = self.series_list.currentItem()
        series = self._current_series()
        if item is not None and series is not None:
            item.setText(self._item_text(series))

    def _apply_label(self) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.label = self.label_edit.text()
        self._refresh_current_item_text()
        self.changed.emit()

    def _pick_color(self) -> None:
        series = self._current_series()
        if series is None:
            return
        initial = QColor(series.color or _DEFAULT_COLOR)
        color = QColorDialog.getColor(initial, self, "Series Color")
        if not color.isValid():
            return
        series.color = color.name()
        self._set_color_swatch(series.color)
        self.changed.emit()

    def _apply_visible(self, checked: bool) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.visible = checked
        self.changed.emit()

    def _apply_width(self, value: float) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.line_width = value
        self.changed.emit()

    def _apply_style(self, _index: int) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.line_style = self.style_combo.currentData()
        self.changed.emit()

    def _apply_marker(self, _index: int) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.marker = self.marker_combo.currentData()
        self.changed.emit()

    def _apply_marker_size(self, value: float) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.marker_size = value
        self.changed.emit()

    def _apply_marker_filled(self, checked: bool) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.marker_filled = checked
        self.changed.emit()

    def _apply_marker_edge_width(self, value: float) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.marker_edge_width = value
        self.changed.emit()

    def _apply_alpha(self, value: float) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.alpha = value
        self.changed.emit()

    def _apply_zorder(self, value: float) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.zorder = value
        self.changed.emit()

    def _apply_offset(self, value: float) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.y_offset = value
        self.changed.emit()

    def _apply_normalize(self, checked: bool) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.normalize_to_max = checked
        self.changed.emit()

    def _apply_bins(self, value: int) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.bins = "auto" if value == 0 else value
        self.changed.emit()

    def _apply_hist_mode(self, _index: int) -> None:
        series = self._current_series()
        if series is None or self._updating:
            return
        series.hist_mode = self.hist_mode_combo.currentData()
        self.changed.emit()

    def _on_remove_clicked(self) -> None:
        series = self._current_series()
        if series is None:
            return
        self._figure.remove_series(series.id)
        self.refresh()
        self.changed.emit()

    def _on_clear_clicked(self) -> None:
        self._figure.clear_series()
        self.refresh()
        self.changed.emit()

    def _on_auto_stack(self) -> None:
        step = self.offset_step_spin.value() or None
        used_step = auto_stack_offsets(self._figure.active_panel, step=step)
        self._updating = True
        self.offset_step_spin.setValue(used_step)
        self._updating = False
        self._on_selection_changed(self.series_list.currentRow())
        self.changed.emit()

    def _on_reset_offsets(self) -> None:
        reset_offsets(self._figure.active_panel)
        self._on_selection_changed(self.series_list.currentRow())
        self.changed.emit()
