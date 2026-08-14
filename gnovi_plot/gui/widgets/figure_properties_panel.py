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

_LIMIT_RANGE = 1e12


class FigurePropertiesPanel(QWidget):
    """Compact title/axis-label/limits/grid/legend controls for a GnoviFigure.

    `changed` is emitted after every mutation so the owner can re-render.
    """

    changed = Signal()

    def __init__(self, figure: GnoviFigure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._updating = False

        self.title_edit = QLineEdit()
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()

        self.x_manual_check = QCheckBox("Manual X limits")
        self.x_min_spin = self._make_limit_spin()
        self.x_max_spin = self._make_limit_spin()
        self.y_manual_check = QCheckBox("Manual Y limits")
        self.y_min_spin = self._make_limit_spin()
        self.y_max_spin = self._make_limit_spin()
        self.reset_limits_button = QPushButton("Reset / Auto Limits")

        self.grid_check = QCheckBox("Show grid")
        self.legend_check = QCheckBox("Show legend")
        self.legend_loc_combo = QComboBox()
        self.legend_loc_combo.addItems(_LEGEND_LOCATIONS)

        labels_group = QGroupBox("Labels")
        labels_form = QFormLayout(labels_group)
        labels_form.addRow("Title", self.title_edit)
        labels_form.addRow("X label", self.xlabel_edit)
        labels_form.addRow("Y label", self.ylabel_edit)

        limits_group = QGroupBox("Axis Limits")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.addWidget(self.x_manual_check)
        x_row = QHBoxLayout()
        x_row.addWidget(self.x_min_spin)
        x_row.addWidget(self.x_max_spin)
        limits_layout.addLayout(x_row)
        limits_layout.addWidget(self.y_manual_check)
        y_row = QHBoxLayout()
        y_row.addWidget(self.y_min_spin)
        y_row.addWidget(self.y_max_spin)
        limits_layout.addLayout(y_row)
        limits_layout.addWidget(self.reset_limits_button)

        display_group = QGroupBox("Grid & Legend")
        display_form = QFormLayout(display_group)
        display_form.addRow(self.grid_check)
        display_form.addRow(self.legend_check)
        display_form.addRow("Legend location", self.legend_loc_combo)

        layout = QVBoxLayout(self)
        layout.addWidget(labels_group)
        layout.addWidget(limits_group)
        layout.addWidget(display_group)
        layout.addStretch(1)

        self.title_edit.editingFinished.connect(self._apply_title)
        self.xlabel_edit.editingFinished.connect(self._apply_xlabel)
        self.ylabel_edit.editingFinished.connect(self._apply_ylabel)
        self.x_manual_check.toggled.connect(self._apply_x_manual)
        self.x_min_spin.valueChanged.connect(self._apply_x_limits)
        self.x_max_spin.valueChanged.connect(self._apply_x_limits)
        self.y_manual_check.toggled.connect(self._apply_y_manual)
        self.y_min_spin.valueChanged.connect(self._apply_y_limits)
        self.y_max_spin.valueChanged.connect(self._apply_y_limits)
        self.reset_limits_button.clicked.connect(self._on_reset_limits)
        self.grid_check.toggled.connect(self._apply_grid)
        self.legend_check.toggled.connect(self._apply_legend_visible)
        self.legend_loc_combo.currentTextChanged.connect(self._apply_legend_loc)

        self._sync_from_figure()

    @staticmethod
    def _make_limit_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-_LIMIT_RANGE, _LIMIT_RANGE)
        spin.setDecimals(6)
        spin.setEnabled(False)
        return spin

    def _sync_from_figure(self) -> None:
        self._updating = True
        self.title_edit.setText(self._figure.title)
        self.xlabel_edit.setText(self._figure.xlabel)
        self.ylabel_edit.setText(self._figure.ylabel)

        self.x_manual_check.setChecked(self._figure.xlim is not None)
        self.x_min_spin.setEnabled(self._figure.xlim is not None)
        self.x_max_spin.setEnabled(self._figure.xlim is not None)
        if self._figure.xlim is not None:
            self.x_min_spin.setValue(self._figure.xlim[0])
            self.x_max_spin.setValue(self._figure.xlim[1])

        self.y_manual_check.setChecked(self._figure.ylim is not None)
        self.y_min_spin.setEnabled(self._figure.ylim is not None)
        self.y_max_spin.setEnabled(self._figure.ylim is not None)
        if self._figure.ylim is not None:
            self.y_min_spin.setValue(self._figure.ylim[0])
            self.y_max_spin.setValue(self._figure.ylim[1])

        self.grid_check.setChecked(self._figure.grid)
        self.legend_check.setChecked(self._figure.legend_visible)
        self.legend_loc_combo.setCurrentText(self._figure.legend_loc)
        self._updating = False

    def sync_axes_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        """Reflect the canvas' live (auto-computed) limits into the disabled
        spin boxes, so they show sensible values before the user goes manual."""
        self._updating = True
        if self._figure.xlim is None:
            self.x_min_spin.setValue(xlim[0])
            self.x_max_spin.setValue(xlim[1])
        if self._figure.ylim is None:
            self.y_min_spin.setValue(ylim[0])
            self.y_max_spin.setValue(ylim[1])
        self._updating = False

    def _apply_title(self) -> None:
        if self._updating:
            return
        self._figure.title = self.title_edit.text()
        self.changed.emit()

    def _apply_xlabel(self) -> None:
        if self._updating:
            return
        self._figure.xlabel = self.xlabel_edit.text()
        self.changed.emit()

    def _apply_ylabel(self) -> None:
        if self._updating:
            return
        self._figure.ylabel = self.ylabel_edit.text()
        self.changed.emit()

    def _apply_x_manual(self, checked: bool) -> None:
        self.x_min_spin.setEnabled(checked)
        self.x_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._figure.xlim = (self.x_min_spin.value(), self.x_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_x_limits(self, _value: float) -> None:
        if self._updating or not self.x_manual_check.isChecked():
            return
        self._figure.xlim = (self.x_min_spin.value(), self.x_max_spin.value())
        self.changed.emit()

    def _apply_y_manual(self, checked: bool) -> None:
        self.y_min_spin.setEnabled(checked)
        self.y_max_spin.setEnabled(checked)
        if self._updating:
            return
        self._figure.ylim = (self.y_min_spin.value(), self.y_max_spin.value()) if checked else None
        self.changed.emit()

    def _apply_y_limits(self, _value: float) -> None:
        if self._updating or not self.y_manual_check.isChecked():
            return
        self._figure.ylim = (self.y_min_spin.value(), self.y_max_spin.value())
        self.changed.emit()

    def _on_reset_limits(self) -> None:
        self._figure.reset_limits()
        self._sync_from_figure()
        self.changed.emit()

    def _apply_grid(self, checked: bool) -> None:
        if self._updating:
            return
        self._figure.grid = checked
        self.changed.emit()

    def _apply_legend_visible(self, checked: bool) -> None:
        if self._updating:
            return
        self._figure.legend_visible = checked
        self.changed.emit()

    def _apply_legend_loc(self, text: str) -> None:
        if self._updating or not text:
            return
        self._figure.legend_loc = text
        self.changed.emit()
