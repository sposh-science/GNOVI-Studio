from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.plotting.figure import GnoviFigure

_LEGEND_LOCATIONS = [
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]

_SCALE_OPTIONS = [("Linear", "linear"), ("Log", "log")]
_TICK_DIRECTIONS = [("Out", "out"), ("In", "in"), ("In & out", "inout")]
_GRID_WHICH_OPTIONS = [("Major", "major"), ("Minor", "minor"), ("Both", "both")]

_LIMIT_RANGE = 1e12
_TICK_SPACING_RANGE = 1e9


class FigurePropertiesPanel(QWidget):
    """Title/axis/tick/spine/grid/legend controls for a GnoviFigure's
    currently active Panel.

    Every field here reads/writes `figure.active_panel`, so switching the
    active panel (multi-panel layouts) just requires calling `refresh()`
    again to reload the widgets for the newly active panel -- the panel
    being edited changes, not the widget wiring.

    `changed` is emitted after every mutation so the owner can re-render.
    """

    changed = Signal()

    def __init__(self, figure: GnoviFigure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._updating = False

        # --- Labels ---
        self.title_edit = QLineEdit()
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()

        # --- Limits & scale ---
        self.x_manual_check = QCheckBox("Manual X limits")
        self.x_min_spin = self._make_limit_spin()
        self.x_max_spin = self._make_limit_spin()
        self.x_scale_combo = self._make_option_combo(_SCALE_OPTIONS)
        self.x_invert_check = QCheckBox("Invert X axis")

        self.y_manual_check = QCheckBox("Manual Y limits")
        self.y_min_spin = self._make_limit_spin()
        self.y_max_spin = self._make_limit_spin()
        self.y_scale_combo = self._make_option_combo(_SCALE_OPTIONS)
        self.y_invert_check = QCheckBox("Invert Y axis")

        self.reset_limits_button = QPushButton("Reset / Auto Limits")

        # --- Ticks ---
        self.tick_direction_combo = self._make_option_combo(_TICK_DIRECTIONS)
        self.minor_ticks_check = QCheckBox("Minor ticks")
        self.major_spacing_x_spin = self._make_spacing_spin()
        self.major_spacing_y_spin = self._make_spacing_spin()
        self.minor_spacing_x_spin = self._make_spacing_spin()
        self.minor_spacing_y_spin = self._make_spacing_spin()
        self.sci_notation_x_check = QCheckBox("Scientific notation (X)")
        self.sci_notation_y_check = QCheckBox("Scientific notation (Y)")

        # --- Spines ---
        self.spine_top_check = QCheckBox("Top")
        self.spine_bottom_check = QCheckBox("Bottom")
        self.spine_left_check = QCheckBox("Left")
        self.spine_right_check = QCheckBox("Right")
        self.spine_width_spin = QDoubleSpinBox()
        self.spine_width_spin.setRange(0.1, 10.0)
        self.spine_width_spin.setSingleStep(0.1)

        # --- Grid & legend ---
        self.grid_check = QCheckBox("Show grid")
        self.grid_which_combo = self._make_option_combo(_GRID_WHICH_OPTIONS)
        self.legend_check = QCheckBox("Show legend")
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(_LEGEND_LOCATIONS)
        self.legend_ncol_spin = QSpinBox()
        self.legend_ncol_spin.setRange(1, 10)
        self.legend_frame_check = QCheckBox("Legend frame")
        self.legend_title_edit = QLineEdit()

        labels_group = QGroupBox("Labels")
        labels_form = QFormLayout(labels_group)
        labels_form.addRow("Title", self.title_edit)
        labels_form.addRow("X label", self.xlabel_edit)
        labels_form.addRow("Y label", self.ylabel_edit)

        limits_group = QGroupBox("Axis Limits & Scale")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.addWidget(self.x_manual_check)
        x_row = QHBoxLayout()
        x_row.addWidget(self.x_min_spin)
        x_row.addWidget(self.x_max_spin)
        limits_layout.addLayout(x_row)
        x_scale_row = QHBoxLayout()
        x_scale_row.addWidget(self.x_scale_combo)
        x_scale_row.addWidget(self.x_invert_check)
        limits_layout.addLayout(x_scale_row)
        limits_layout.addWidget(self.y_manual_check)
        y_row = QHBoxLayout()
        y_row.addWidget(self.y_min_spin)
        y_row.addWidget(self.y_max_spin)
        limits_layout.addLayout(y_row)
        y_scale_row = QHBoxLayout()
        y_scale_row.addWidget(self.y_scale_combo)
        y_scale_row.addWidget(self.y_invert_check)
        limits_layout.addLayout(y_scale_row)
        limits_layout.addWidget(self.reset_limits_button)

        ticks_group = QGroupBox("Ticks")
        ticks_form = QFormLayout(ticks_group)
        ticks_form.addRow("Direction", self.tick_direction_combo)
        ticks_form.addRow(self.minor_ticks_check)
        ticks_form.addRow("Major spacing X (0 = auto)", self.major_spacing_x_spin)
        ticks_form.addRow("Major spacing Y (0 = auto)", self.major_spacing_y_spin)
        ticks_form.addRow("Minor spacing X (0 = auto)", self.minor_spacing_x_spin)
        ticks_form.addRow("Minor spacing Y (0 = auto)", self.minor_spacing_y_spin)
        ticks_form.addRow(self.sci_notation_x_check)
        ticks_form.addRow(self.sci_notation_y_check)

        spines_group = QGroupBox("Spines")
        spines_layout = QVBoxLayout(spines_group)
        spines_row = QHBoxLayout()
        spines_row.addWidget(self.spine_top_check)
        spines_row.addWidget(self.spine_bottom_check)
        spines_row.addWidget(self.spine_left_check)
        spines_row.addWidget(self.spine_right_check)
        spines_layout.addLayout(spines_row)
        spine_width_form = QFormLayout()
        spine_width_form.addRow("Line width", self.spine_width_spin)
        spines_layout.addLayout(spine_width_form)

        display_group = QGroupBox("Grid & Legend")
        display_form = QFormLayout(display_group)
        display_form.addRow(self.grid_check)
        display_form.addRow("Grid lines", self.grid_which_combo)
        display_form.addRow(self.legend_check)
        display_form.addRow("Legend location", self.legend_loc_combo)
        display_form.addRow("Legend columns", self.legend_ncol_spin)
        display_form.addRow(self.legend_frame_check)
        display_form.addRow("Legend title", self.legend_title_edit)

        layout = QVBoxLayout(self)
        layout.addWidget(labels_group)
        layout.addWidget(limits_group)
        layout.addWidget(ticks_group)
        layout.addWidget(spines_group)
        layout.addWidget(display_group)
        layout.addStretch(1)

        self.title_edit.editingFinished.connect(self._apply_title)
        self.xlabel_edit.editingFinished.connect(self._apply_xlabel)
        self.ylabel_edit.editingFinished.connect(self._apply_ylabel)
        self.x_manual_check.toggled.connect(self._apply_x_manual)
        self.x_min_spin.valueChanged.connect(self._apply_x_limits)
        self.x_max_spin.valueChanged.connect(self._apply_x_limits)
        self.x_scale_combo.currentIndexChanged.connect(self._apply_x_scale)
        self.x_invert_check.toggled.connect(self._apply_x_invert)
        self.y_manual_check.toggled.connect(self._apply_y_manual)
        self.y_min_spin.valueChanged.connect(self._apply_y_limits)
        self.y_max_spin.valueChanged.connect(self._apply_y_limits)
        self.y_scale_combo.currentIndexChanged.connect(self._apply_y_scale)
        self.y_invert_check.toggled.connect(self._apply_y_invert)
        self.reset_limits_button.clicked.connect(self._on_reset_limits)

        self.tick_direction_combo.currentIndexChanged.connect(self._apply_tick_direction)
        self.minor_ticks_check.toggled.connect(self._apply_minor_ticks)
        self.major_spacing_x_spin.valueChanged.connect(self._apply_major_spacing_x)
        self.major_spacing_y_spin.valueChanged.connect(self._apply_major_spacing_y)
        self.minor_spacing_x_spin.valueChanged.connect(self._apply_minor_spacing_x)
        self.minor_spacing_y_spin.valueChanged.connect(self._apply_minor_spacing_y)
        self.sci_notation_x_check.toggled.connect(self._apply_sci_x)
        self.sci_notation_y_check.toggled.connect(self._apply_sci_y)

        self.spine_top_check.toggled.connect(self._apply_spines)
        self.spine_bottom_check.toggled.connect(self._apply_spines)
        self.spine_left_check.toggled.connect(self._apply_spines)
        self.spine_right_check.toggled.connect(self._apply_spines)
        self.spine_width_spin.valueChanged.connect(self._apply_spines)

        self.grid_check.toggled.connect(self._apply_grid)
        self.grid_which_combo.currentIndexChanged.connect(self._apply_grid_which)
        self.legend_check.toggled.connect(self._apply_legend_visible)
        self.legend_loc_combo.currentTextChanged.connect(self._apply_legend_loc)
        self.legend_ncol_spin.valueChanged.connect(self._apply_legend_ncol)
        self.legend_frame_check.toggled.connect(self._apply_legend_frame)
        self.legend_title_edit.editingFinished.connect(self._apply_legend_title)

        self.refresh()

    @staticmethod
    def _make_limit_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-_LIMIT_RANGE, _LIMIT_RANGE)
        spin.setDecimals(6)
        spin.setEnabled(False)
        return spin

    @staticmethod
    def _make_spacing_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, _TICK_SPACING_RANGE)
        spin.setDecimals(6)
        spin.setSpecialValueText("Auto")
        return spin

    @staticmethod
    def _make_option_combo(options: list[tuple[str, str]]) -> QComboBox:
        combo = QComboBox()
        for text, code in options:
            combo.addItem(text, code)
        return combo

    @property
    def _panel(self):
        return self._figure.active_panel

    def refresh(self) -> None:
        """Reload every field from `figure.active_panel`. Call this after
        switching the active panel, in addition to after construction."""
        self._sync_from_figure()

    def _sync_from_figure(self) -> None:
        panel = self._panel
        self._updating = True
        self.title_edit.setText(panel.title)
        self.xlabel_edit.setText(panel.xlabel)
        self.ylabel_edit.setText(panel.ylabel)

        self.x_manual_check.setChecked(panel.xlim is not None)
        self.x_min_spin.setEnabled(panel.xlim is not None)
        self.x_max_spin.setEnabled(panel.xlim is not None)
        if panel.xlim is not None:
            self.x_min_spin.setValue(panel.xlim[0])
            self.x_max_spin.setValue(panel.xlim[1])
        self.x_scale_combo.setCurrentIndex(max(self.x_scale_combo.findData(panel.xscale), 0))
        self.x_invert_check.setChecked(panel.invert_x)

        self.y_manual_check.setChecked(panel.ylim is not None)
        self.y_min_spin.setEnabled(panel.ylim is not None)
        self.y_max_spin.setEnabled(panel.ylim is not None)
        if panel.ylim is not None:
            self.y_min_spin.setValue(panel.ylim[0])
            self.y_max_spin.setValue(panel.ylim[1])
        self.y_scale_combo.setCurrentIndex(max(self.y_scale_combo.findData(panel.yscale), 0))
        self.y_invert_check.setChecked(panel.invert_y)

        self.tick_direction_combo.setCurrentIndex(
            max(self.tick_direction_combo.findData(panel.tick_direction), 0)
        )
        self.minor_ticks_check.setChecked(panel.minor_ticks)
        self.major_spacing_x_spin.setValue(panel.major_tick_spacing_x or 0.0)
        self.major_spacing_y_spin.setValue(panel.major_tick_spacing_y or 0.0)
        self.minor_spacing_x_spin.setValue(panel.minor_tick_spacing_x or 0.0)
        self.minor_spacing_y_spin.setValue(panel.minor_tick_spacing_y or 0.0)
        self.sci_notation_x_check.setChecked(panel.scientific_notation_x)
        self.sci_notation_y_check.setChecked(panel.scientific_notation_y)

        self.spine_top_check.setChecked(panel.spine_top)
        self.spine_bottom_check.setChecked(panel.spine_bottom)
        self.spine_left_check.setChecked(panel.spine_left)
        self.spine_right_check.setChecked(panel.spine_right)
        self.spine_width_spin.setValue(panel.spine_linewidth)

        self.grid_check.setChecked(panel.grid)
        self.grid_which_combo.setCurrentIndex(max(self.grid_which_combo.findData(panel.grid_which), 0))
        self.legend_check.setChecked(panel.legend_visible)
        self.legend_loc_combo.setCurrentText(panel.legend_loc)
        self.legend_ncol_spin.setValue(panel.legend_ncol)
        self.legend_frame_check.setChecked(panel.legend_frameon)
        self.legend_title_edit.setText(panel.legend_title)
        self._updating = False

    def sync_axes_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        """Reflect the canvas' live (auto-computed) limits into the disabled
        spin boxes, so they show sensible values before the user goes manual."""
        panel = self._panel
        self._updating = True
        if panel.xlim is None:
            self.x_min_spin.setValue(xlim[0])
            self.x_max_spin.setValue(xlim[1])
        if panel.ylim is None:
            self.y_min_spin.setValue(ylim[0])
            self.y_max_spin.setValue(ylim[1])
        self._updating = False

    def _apply_title(self) -> None:
        if self._updating:
            return
        self._panel.title = self.title_edit.text()
        self.changed.emit()

    def _apply_xlabel(self) -> None:
        if self._updating:
            return
        self._panel.xlabel = self.xlabel_edit.text()
        self.changed.emit()

    def _apply_ylabel(self) -> None:
        if self._updating:
            return
        self._panel.ylabel = self.ylabel_edit.text()
        self.changed.emit()

    def _apply_x_manual(self, checked: bool) -> None:
        self.x_min_spin.setEnabled(checked)
        self.x_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._panel.xlim = (self.x_min_spin.value(), self.x_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_x_limits(self, _value: float) -> None:
        if self._updating or not self.x_manual_check.isChecked():
            return
        self._panel.xlim = (self.x_min_spin.value(), self.x_max_spin.value())
        self.changed.emit()

    def _apply_x_scale(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.xscale = self.x_scale_combo.currentData()
        self.changed.emit()

    def _apply_x_invert(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.invert_x = checked
        self.changed.emit()

    def _apply_y_manual(self, checked: bool) -> None:
        self.y_min_spin.setEnabled(checked)
        self.y_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._panel.ylim = (self.y_min_spin.value(), self.y_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_y_limits(self, _value: float) -> None:
        if self._updating or not self.y_manual_check.isChecked():
            return
        self._panel.ylim = (self.y_min_spin.value(), self.y_max_spin.value())
        self.changed.emit()

    def _apply_y_scale(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.yscale = self.y_scale_combo.currentData()
        self.changed.emit()

    def _apply_y_invert(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.invert_y = checked
        self.changed.emit()

    def _on_reset_limits(self) -> None:
        self._panel.reset_limits()
        self._sync_from_figure()
        self.changed.emit()

    def _apply_tick_direction(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.tick_direction = self.tick_direction_combo.currentData()
        self.changed.emit()

    def _apply_minor_ticks(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.minor_ticks = checked
        self.changed.emit()

    def _apply_major_spacing_x(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_spacing_x = value or None
        self.changed.emit()

    def _apply_major_spacing_y(self, value: float) -> None:
        if self._updating:
            return
        self._panel.major_tick_spacing_y = value or None
        self.changed.emit()

    def _apply_minor_spacing_x(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_spacing_x = value or None
        self.changed.emit()

    def _apply_minor_spacing_y(self, value: float) -> None:
        if self._updating:
            return
        self._panel.minor_tick_spacing_y = value or None
        self.changed.emit()

    def _apply_sci_x(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.scientific_notation_x = checked
        self.changed.emit()

    def _apply_sci_y(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.scientific_notation_y = checked
        self.changed.emit()

    def _apply_spines(self, *_args) -> None:
        if self._updating:
            return
        self._panel.spine_top = self.spine_top_check.isChecked()
        self._panel.spine_bottom = self.spine_bottom_check.isChecked()
        self._panel.spine_left = self.spine_left_check.isChecked()
        self._panel.spine_right = self.spine_right_check.isChecked()
        self._panel.spine_linewidth = self.spine_width_spin.value()
        self.changed.emit()

    def _apply_grid(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.grid = checked
        self.changed.emit()

    def _apply_grid_which(self, _index: int) -> None:
        if self._updating:
            return
        self._panel.grid_which = self.grid_which_combo.currentData()
        self.changed.emit()

    def _apply_legend_visible(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.legend_visible = checked
        self.changed.emit()

    def _apply_legend_loc(self, text: str) -> None:
        if self._updating or not text:
            return
        self._panel.legend_loc = text
        self.changed.emit()

    def _apply_legend_ncol(self, value: int) -> None:
        if self._updating:
            return
        self._panel.legend_ncol = value
        self.changed.emit()

    def _apply_legend_frame(self, checked: bool) -> None:
        if self._updating:
            return
        self._panel.legend_frameon = checked
        self.changed.emit()

    def _apply_legend_title(self) -> None:
        if self._updating:
            return
        self._panel.legend_title = self.legend_title_edit.text()
        self.changed.emit()
