from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.gui.styles import PlotTheme
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.units import ASPECT_RATIO_PRESETS, PUBLICATION_PRESETS_MM, from_inches, to_inches

_UNITS = ["mm", "cm", "in"]

_PLOT_THEME_OPTIONS = [(PlotTheme.LIGHT, "Light"), (PlotTheme.DARK, "Dark")]

# Figure-wide scalar fields this panel edits, in scope for the Apply /
# Cancel / Reset behavior of the dialog that hosts it (see gui.dialogs.
# live_dialog.LiveDialog) -- deliberately excludes `layout`/`panels`/
# `active_panel_index`: those are structural (they add/remove Panel objects
# and can hold series a user added while this dialog stayed open), so
# reverting them on Cancel risks discarding unrelated work. Reset restores
# each of these to a fresh GnoviFigure()'s default, same scope.
_FIGURE_SCALAR_FIELDS = [
    "figure_width_in",
    "figure_height_in",
    "aspect_preset",
    "lock_aspect_ratio",
    "font_family",
    "base_font_size",
    "title_font_size",
    "axis_label_font_size",
    "tick_label_font_size",
    "legend_font_size",
    "panel_labels_visible",
]

# Order matters: existing indices (e.g. index 3 == "2 x 2") are relied on by
# tests and by the Panels menu, so new presets are appended rather than
# inserted.
LAYOUT_PRESETS = [
    ("1 x 1", (1, 1)),
    ("1 x 2", (1, 2)),
    ("2 x 1", (2, 1)),
    ("2 x 2", (2, 2)),
    ("1 x 3", (1, 3)),
    ("2 x 3", (2, 3)),
    ("3 x 2", (3, 2)),
]
_NO_PUBLICATION_PRESET = "(none)"
_SYSTEM_DEFAULT_FONT = "(system default)"


class FigureSizePanel(QWidget):
    """Figure/page geometry (size, aspect-ratio and publication presets),
    multi-panel layout + active-panel selection, and figure-wide typography.

    Figure size here controls export/page geometry only -- it never touches
    Matplotlib's Axes aspect (`ax.set_aspect`) and never resizes the
    interactive canvas widget; the on-screen canvas keeps behaving
    responsively regardless of the configured export size.

    `changed` is emitted after any mutation that should trigger a re-render.
    `panel_switched` is emitted after the active panel or the panel layout
    changes, so the owner can reload panel-scoped widgets (series list,
    figure properties) for the newly active panel.
    """

    changed = Signal()
    panel_switched = Signal()
    theme_change_requested = Signal(object)  # PlotTheme (see `_on_theme_combo_changed`)

    def __init__(self, figure: GnoviFigure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._unit = "in"
        self._updating = False

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(list(ASPECT_RATIO_PRESETS))

        self.publication_combo = QComboBox()
        self.publication_combo.addItems([_NO_PUBLICATION_PRESET] + list(PUBLICATION_PRESETS_MM))

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(_UNITS)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 5000.0)
        self.width_spin.setDecimals(2)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 5000.0)
        self.height_spin.setDecimals(2)

        self.lock_check = QCheckBox("Lock aspect ratio")

        size_group = QGroupBox("Figure Size")
        size_form = QFormLayout(size_group)
        size_form.addRow("Aspect preset", self.aspect_combo)
        size_form.addRow("Publication preset", self.publication_combo)
        size_form.addRow("Unit", self.unit_combo)
        size_form.addRow("Width", self.width_spin)
        size_form.addRow("Height", self.height_spin)
        size_form.addRow(self.lock_check)

        self.layout_combo = QComboBox()
        for text, _dims in LAYOUT_PRESETS:
            self.layout_combo.addItem(text)
        self.panel_combo = QComboBox()
        self.panel_labels_check = QCheckBox("Show panel labels (a), (b), …")

        panels_group = QGroupBox("Panels")
        panels_form = QFormLayout(panels_group)
        panels_form.addRow("Layout", self.layout_combo)
        panels_form.addRow("Active panel", self.panel_combo)
        panels_form.addRow(self.panel_labels_check)

        # Plot Theme lives here (not GnoviFigure -- it's MainWindow-owned,
        # persisted state, see gui.main_window) purely as a display
        # convenience so it's reachable from the Figure page alongside the
        # rest of the figure's look; `theme_change_requested` routes the
        # actual change through MainWindow._on_theme_changed, the same
        # handler the View menu and toolbar already use, and
        # `set_current_theme` below keeps this combo in sync with whichever
        # of those the user changed it from.
        self.theme_combo = QComboBox()
        for mode, label in _PLOT_THEME_OPTIONS:
            self.theme_combo.addItem(label, mode)

        theme_group = QGroupBox("Plot Theme")
        theme_form = QFormLayout(theme_group)
        theme_form.addRow("Theme", self.theme_combo)

        self.font_family_combo = QComboBox()
        self.font_family_combo.addItem(_SYSTEM_DEFAULT_FONT)
        self.font_family_combo.addItems(QFontDatabase.families())

        self.base_font_spin = self._make_font_spin()
        self.title_font_spin = self._make_font_spin()
        self.axis_font_spin = self._make_font_spin()
        self.tick_font_spin = self._make_font_spin()
        self.legend_font_spin = self._make_font_spin()

        font_group = QGroupBox("Typography")
        font_form = QFormLayout(font_group)
        font_form.addRow("Font family", self.font_family_combo)
        font_form.addRow("Base size", self.base_font_spin)
        font_form.addRow("Title size", self.title_font_spin)
        font_form.addRow("Axis label size", self.axis_font_spin)
        font_form.addRow("Tick label size", self.tick_font_spin)
        font_form.addRow("Legend size", self.legend_font_spin)

        # Grid Appearance lives on the Axes page now -- see
        # gui.widgets.figure_properties_panel's "single authoritative
        # location" docstring note; it isn't duplicated here.

        self.reset_button = QPushButton("Reset to Defaults")

        layout = QVBoxLayout(self)
        layout.addWidget(size_group)
        layout.addWidget(panels_group)
        layout.addWidget(theme_group)
        layout.addWidget(font_group)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)

        self.aspect_combo.currentTextChanged.connect(self._apply_aspect_preset)
        self.publication_combo.currentTextChanged.connect(self._apply_publication_preset)
        self.unit_combo.currentTextChanged.connect(self._on_unit_changed)
        self.width_spin.valueChanged.connect(self._on_width_changed)
        self.height_spin.valueChanged.connect(self._on_height_changed)
        self.lock_check.toggled.connect(self._on_lock_toggled)
        self.layout_combo.currentIndexChanged.connect(self._apply_layout)
        self.panel_combo.currentIndexChanged.connect(self._apply_active_panel)
        self.panel_labels_check.toggled.connect(self._apply_panel_labels)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_combo_changed)
        self.font_family_combo.currentTextChanged.connect(self._apply_font_family)
        self.base_font_spin.valueChanged.connect(self._apply_base_font)
        self.title_font_spin.valueChanged.connect(self._apply_title_font)
        self.axis_font_spin.valueChanged.connect(self._apply_axis_font)
        self.tick_font_spin.valueChanged.connect(self._apply_tick_font)
        self.legend_font_spin.valueChanged.connect(self._apply_legend_font)
        self.reset_button.clicked.connect(self.reset_to_defaults)

        self._sync_from_figure()

    @staticmethod
    def _make_font_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(4.0, 48.0)
        spin.setSuffix(" pt")
        return spin

    def _sync_from_figure(self) -> None:
        self._updating = True
        self.aspect_combo.setCurrentText(self._figure.aspect_preset)
        self.publication_combo.setCurrentIndex(0)
        self.unit_combo.setCurrentText(self._unit)
        self.width_spin.setValue(from_inches(self._figure.figure_width_in, self._unit))
        self.height_spin.setValue(from_inches(self._figure.figure_height_in, self._unit))
        self.lock_check.setChecked(self._figure.lock_aspect_ratio)

        rows, cols = self._figure.layout
        for i, (_text, dims) in enumerate(LAYOUT_PRESETS):
            if dims == (rows, cols):
                self.layout_combo.setCurrentIndex(i)
                break
        self._refresh_panel_options()
        self.panel_labels_check.setChecked(self._figure.panel_labels_visible)

        self.font_family_combo.setCurrentText(self._figure.font_family or _SYSTEM_DEFAULT_FONT)
        self.base_font_spin.setValue(self._figure.base_font_size)
        self.title_font_spin.setValue(self._figure.title_font_size)
        self.axis_font_spin.setValue(self._figure.axis_label_font_size)
        self.tick_font_spin.setValue(self._figure.tick_label_font_size)
        self.legend_font_spin.setValue(self._figure.legend_font_size)
        self._updating = False

    def refresh(self) -> None:
        """Reload every field from the live `GnoviFigure` -- call this after
        an external mutation (e.g. Undo/Redo restoring a snapshot), in
        addition to after construction."""
        self._sync_from_figure()

    def _refresh_panel_options(self) -> None:
        self.panel_combo.blockSignals(True)
        self.panel_combo.clear()
        for i in range(len(self._figure.panels)):
            self.panel_combo.addItem(f"Panel {i + 1}")
        self.panel_combo.setCurrentIndex(self._figure.active_panel_index)
        self.panel_combo.blockSignals(False)

    def _apply_aspect_preset(self, text: str) -> None:
        if self._updating or not text:
            return
        self._figure.aspect_preset = text
        ratio = ASPECT_RATIO_PRESETS.get(text)
        if ratio is not None:
            self._figure.lock_aspect_ratio = True
            self._figure.figure_height_in = self._figure.figure_width_in / ratio
        elif text == "Auto / Fit workspace":
            self._figure.lock_aspect_ratio = False
        self._updating = True
        self.lock_check.setChecked(self._figure.lock_aspect_ratio)
        self.height_spin.setValue(from_inches(self._figure.figure_height_in, self._unit))
        self._updating = False
        self.changed.emit()

    def _apply_publication_preset(self, text: str) -> None:
        if self._updating or text == _NO_PUBLICATION_PRESET:
            return
        width_mm, height_mm = PUBLICATION_PRESETS_MM[text]
        self._figure.figure_width_in = to_inches(width_mm, "mm")
        self._figure.figure_height_in = to_inches(height_mm, "mm")
        self._figure.aspect_preset = "Custom"
        self._figure.lock_aspect_ratio = True
        self._unit = "mm"
        self._updating = True
        self.unit_combo.setCurrentText("mm")
        self.aspect_combo.setCurrentText("Custom")
        self.lock_check.setChecked(True)
        self.width_spin.setValue(width_mm)
        self.height_spin.setValue(height_mm)
        self._updating = False
        self.changed.emit()

    def _on_unit_changed(self, unit: str) -> None:
        if self._updating or not unit:
            return
        self._unit = unit
        self._updating = True
        self.width_spin.setValue(from_inches(self._figure.figure_width_in, unit))
        self.height_spin.setValue(from_inches(self._figure.figure_height_in, unit))
        self._updating = False

    def _mark_custom(self) -> None:
        self._figure.aspect_preset = "Custom"
        self._updating = True
        self.aspect_combo.setCurrentText("Custom")
        self._updating = False

    def _on_width_changed(self, value: float) -> None:
        if self._updating:
            return
        old_width_in = self._figure.figure_width_in
        old_height_in = self._figure.figure_height_in
        self._figure.figure_width_in = to_inches(value, self._unit)
        if self._figure.lock_aspect_ratio and old_width_in and old_height_in:
            ratio = old_width_in / old_height_in
            self._figure.figure_height_in = self._figure.figure_width_in / ratio
            self._updating = True
            self.height_spin.setValue(from_inches(self._figure.figure_height_in, self._unit))
            self._updating = False
        self._mark_custom()
        self.changed.emit()

    def _on_height_changed(self, value: float) -> None:
        if self._updating:
            return
        old_width_in = self._figure.figure_width_in
        old_height_in = self._figure.figure_height_in
        self._figure.figure_height_in = to_inches(value, self._unit)
        if self._figure.lock_aspect_ratio and old_width_in and old_height_in:
            ratio = old_width_in / old_height_in
            self._figure.figure_width_in = self._figure.figure_height_in * ratio
            self._updating = True
            self.width_spin.setValue(from_inches(self._figure.figure_width_in, self._unit))
            self._updating = False
        self._mark_custom()
        self.changed.emit()

    def _on_lock_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self._figure.lock_aspect_ratio = checked
        self.changed.emit()

    def _apply_layout(self, index: int) -> None:
        if self._updating or index < 0:
            return
        _text, (rows, cols) = LAYOUT_PRESETS[index]
        self._figure.set_layout(rows, cols)
        self._refresh_panel_options()
        self.changed.emit()
        self.panel_switched.emit()

    def _apply_active_panel(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self._figure.set_active_panel(index)
        self.panel_switched.emit()

    def _apply_panel_labels(self, checked: bool) -> None:
        if self._updating:
            return
        self._figure.panel_labels_visible = checked
        self.changed.emit()

    def _apply_font_family(self, text: str) -> None:
        if self._updating:
            return
        self._figure.font_family = None if text == _SYSTEM_DEFAULT_FONT else text
        self.changed.emit()

    def _apply_base_font(self, value: float) -> None:
        if self._updating:
            return
        self._figure.base_font_size = value
        self.changed.emit()

    def _apply_title_font(self, value: float) -> None:
        if self._updating:
            return
        self._figure.title_font_size = value
        self.changed.emit()

    def _apply_axis_font(self, value: float) -> None:
        if self._updating:
            return
        self._figure.axis_label_font_size = value
        self.changed.emit()

    def _apply_tick_font(self, value: float) -> None:
        if self._updating:
            return
        self._figure.tick_label_font_size = value
        self.changed.emit()

    def _apply_legend_font(self, value: float) -> None:
        if self._updating:
            return
        self._figure.legend_font_size = value
        self.changed.emit()

    # --- Plot Theme (MainWindow-owned state -- see the constructor's
    # theme_group comment) ---------------------------------------------------

    def _on_theme_combo_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self.theme_change_requested.emit(self.theme_combo.itemData(index))

    def set_current_theme(self, mode: PlotTheme) -> None:
        """Reflect the app's current Plot Theme without re-emitting
        `theme_change_requested` -- called by MainWindow whenever the
        toolbar combo or View menu changes it instead."""
        self._updating = True
        self.theme_combo.setCurrentIndex(max(self.theme_combo.findData(mode), 0))
        self._updating = False

    # --- Apply / Cancel / Reset (see gui.dialogs.live_dialog.LiveDialog) ---

    def capture_state(self) -> dict:
        """Snapshot the figure-wide scalar fields this panel edits, for the
        hosting LiveDialog's Cancel to restore on close."""
        return {name: getattr(self._figure, name) for name in _FIGURE_SCALAR_FIELDS}

    def restore_state(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self._figure, name, value)
        self._sync_from_figure()
        self.changed.emit()

    def reset_to_defaults(self) -> None:
        """Reset just the fields in `_FIGURE_SCALAR_FIELDS` to a fresh
        GnoviFigure()'s defaults -- panel layout/content is untouched."""
        defaults = GnoviFigure()
        for name in _FIGURE_SCALAR_FIELDS:
            setattr(self._figure, name, getattr(defaults, name))
        self._sync_from_figure()
        self.changed.emit()
