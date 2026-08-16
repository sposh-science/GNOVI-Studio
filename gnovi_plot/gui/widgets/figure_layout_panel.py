from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gnovi_plot.gui.widgets.active_panel_label import ActivePanelLabel
from gnovi_plot.plotting.backends.matplotlib_backend import compute_tight_layout
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.units import PANEL_ASPECT_RATIO_PRESETS

# Minimum gap enforced between opposing margins (left < right, bottom < top)
# -- each spin box's sibling range is kept updated live so it can never be
# dragged into a combination Matplotlib's `subplots_adjust` would reject.
_MIN_MARGIN_GAP = 0.02

_MARGIN_FIELDS = ["margin_left", "margin_right", "margin_bottom", "margin_top"]
_SPACING_FIELDS = ["panel_wspace", "panel_hspace"]
_ASPECT_FIELDS = ["panel_aspect_preset"]
_LAYOUT_FIELDS = _MARGIN_FIELDS + _SPACING_FIELDS + _ASPECT_FIELDS


class FigureLayoutPanel(QWidget):
    """Outer figure margins, inter-panel spacing, and Panel Aspect Ratio.

    Panel Aspect Ratio (`GnoviFigure.panel_aspect_preset`) is deliberately
    separate from Figure Aspect Ratio (`gui.widgets.figure_size_panel`,
    `GnoviFigure.aspect_preset`/`lock_aspect_ratio`): the former is the
    physical width/height shape of each individual panel's Axes box, the
    latter is the shape of the complete multi-panel page. It's a
    figure/layout-level setting applying uniformly to every panel -- never
    an Active-Panel-only Axes property (see `gui.widgets
    .figure_properties_panel`, which stays untouched by this) -- and it
    never changes numeric X/Y data-unit scaling (see
    `plotting.backends.matplotlib_backend.render_panel`, the only place
    it's actually applied, via `Axes.set_box_aspect`).

    GNOVI Studio's own dedicated route to the same conceptual controls as
    Matplotlib's built-in "Configure Subplots" toolbar dialog -- that
    dialog stays available unchanged (see `gui.main_window`'s toolbar
    setup) for advanced users, and both routes update the exact same
    underlying `GnoviFigure` state (`margin_left`/`margin_right`/
    `margin_bottom`/`margin_top`/`panel_wspace`/`panel_hspace`) that the
    on-screen preview and every exported format (PNG/TIFF/SVG/PDF) render
    from -- see `plotting.backends.matplotlib_backend.apply_figure_layout`.
    There is no GUI-only layout state to fall out of sync.

    Edits apply live (the same convention as `figure_size_panel`/
    `figure_properties_panel`) -- there's no separate modal "Apply" step;
    every change is immediately reflected once the owner re-renders on
    `changed`. "Tight Layout" is a one-off action, not a persistent mode:
    it computes Matplotlib's automatic layout once and bakes the result
    into these same fields, after which the user can keep tuning them
    manually -- it is never re-applied silently on a later render.

    `changed` is emitted after every mutation so the owner can re-render.
    """

    changed = Signal()

    def __init__(self, figure: GnoviFigure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._updating = False

        self.active_panel_label = ActivePanelLabel(figure)

        self.left_spin = self._make_margin_spin()
        self.right_spin = self._make_margin_spin()
        self.bottom_spin = self._make_margin_spin()
        self.top_spin = self._make_margin_spin()

        self.wspace_spin = self._make_spacing_spin()
        self.hspace_spin = self._make_spacing_spin()

        margins_group = QGroupBox("Margins")
        margins_form = QFormLayout(margins_group)
        margins_form.addRow("Left margin", self.left_spin)
        margins_form.addRow("Right margin", self.right_spin)
        margins_form.addRow("Top margin", self.top_spin)
        margins_form.addRow("Bottom margin", self.bottom_spin)

        spacing_group = QGroupBox("Panel Spacing")
        spacing_form = QFormLayout(spacing_group)
        spacing_form.addRow("Horizontal spacing (wspace)", self.wspace_spin)
        spacing_form.addRow("Vertical spacing (hspace)", self.hspace_spin)

        self.panel_aspect_combo = QComboBox()
        self.panel_aspect_combo.addItems(list(PANEL_ASPECT_RATIO_PRESETS))
        # Distinct label/tooltip from Figure Aspect Ratio (Figure Size page)
        # -- this one is each individual graph box's shape only.
        self.panel_aspect_combo.setToolTip(
            "Panel Aspect Ratio: shape of each individual graph box. "
            "Does not change numerical X/Y scaling."
        )
        shape_group = QGroupBox("Panel Shape")
        shape_form = QFormLayout(shape_group)
        shape_form.addRow("Panel Aspect Ratio", self.panel_aspect_combo)

        self.tight_layout_button = QPushButton("Tight Layout")
        self.reset_button = QPushButton("Reset to Defaults")
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.tight_layout_button)
        buttons_row.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.active_panel_label)
        layout.addWidget(margins_group)
        layout.addWidget(spacing_group)
        layout.addWidget(shape_group)
        layout.addLayout(buttons_row)
        layout.addStretch(1)

        self.left_spin.valueChanged.connect(self._apply_left)
        self.right_spin.valueChanged.connect(self._apply_right)
        self.bottom_spin.valueChanged.connect(self._apply_bottom)
        self.top_spin.valueChanged.connect(self._apply_top)
        self.wspace_spin.valueChanged.connect(self._apply_wspace)
        self.hspace_spin.valueChanged.connect(self._apply_hspace)
        self.panel_aspect_combo.currentTextChanged.connect(self._apply_panel_aspect)
        self.tight_layout_button.clicked.connect(self.apply_tight_layout)
        self.reset_button.clicked.connect(self.reset_to_defaults)

        self.refresh()

    @staticmethod
    def _make_margin_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        return spin

    @staticmethod
    def _make_spacing_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 5.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        return spin

    def set_figure(self, figure: GnoviFigure) -> None:
        """Repoint this panel at a different `GnoviFigure` (e.g. after
        Open/New Project swaps the active figure) and reload from it."""
        self._figure = figure
        self.refresh()

    def refresh(self) -> None:
        """Reload every field from the live `GnoviFigure` -- call this after
        an external mutation (e.g. Undo/Redo restoring a snapshot), in
        addition to after construction."""
        self._sync_from_figure()

    def _sync_from_figure(self) -> None:
        self.active_panel_label.refresh(self._figure)
        self._updating = True
        self.left_spin.setValue(self._figure.margin_left)
        self.right_spin.setValue(self._figure.margin_right)
        self.bottom_spin.setValue(self._figure.margin_bottom)
        self.top_spin.setValue(self._figure.margin_top)
        self.wspace_spin.setValue(self._figure.panel_wspace)
        self.hspace_spin.setValue(self._figure.panel_hspace)
        self.panel_aspect_combo.setCurrentText(self._figure.panel_aspect_preset)
        self._sync_margin_bounds()
        self._updating = False

    def _sync_margin_bounds(self) -> None:
        """Keep each margin spin box's range from crossing its opposing
        margin (left < right, bottom < top) -- Matplotlib's
        `subplots_adjust` raises ValueError on an invalid combination, so
        this is enforced interactively rather than only discovered on
        apply."""
        self.left_spin.setMaximum(max(self.right_spin.value() - _MIN_MARGIN_GAP, 0.0))
        self.right_spin.setMinimum(min(self.left_spin.value() + _MIN_MARGIN_GAP, 1.0))
        self.bottom_spin.setMaximum(max(self.top_spin.value() - _MIN_MARGIN_GAP, 0.0))
        self.top_spin.setMinimum(min(self.bottom_spin.value() + _MIN_MARGIN_GAP, 1.0))

    def _apply_left(self, value: float) -> None:
        if self._updating:
            return
        self._figure.margin_left = value
        self._updating = True
        self._sync_margin_bounds()
        self._updating = False
        self.changed.emit()

    def _apply_right(self, value: float) -> None:
        if self._updating:
            return
        self._figure.margin_right = value
        self._updating = True
        self._sync_margin_bounds()
        self._updating = False
        self.changed.emit()

    def _apply_bottom(self, value: float) -> None:
        if self._updating:
            return
        self._figure.margin_bottom = value
        self._updating = True
        self._sync_margin_bounds()
        self._updating = False
        self.changed.emit()

    def _apply_top(self, value: float) -> None:
        if self._updating:
            return
        self._figure.margin_top = value
        self._updating = True
        self._sync_margin_bounds()
        self._updating = False
        self.changed.emit()

    def _apply_wspace(self, value: float) -> None:
        if self._updating:
            return
        self._figure.panel_wspace = value
        self.changed.emit()

    def _apply_hspace(self, value: float) -> None:
        if self._updating:
            return
        self._figure.panel_hspace = value
        self.changed.emit()

    def _apply_panel_aspect(self, text: str) -> None:
        if self._updating or not text:
            return
        self._figure.panel_aspect_preset = text
        self.changed.emit()

    def apply_tight_layout(self) -> None:
        """Compute Matplotlib's automatic "tight" margins/spacing once and
        bake the result into the figure's stored fields -- a one-off
        starting point, not a persistent mode; the user can keep tuning the
        spin boxes afterward (see `compute_tight_layout`)."""
        values = compute_tight_layout(self._figure)
        for name, value in values.items():
            setattr(self._figure, name, value)
        self._sync_from_figure()
        self.changed.emit()

    def reset_to_defaults(self) -> None:
        """Reset just the margin/spacing/Panel-Aspect-Ratio fields to a
        fresh GnoviFigure()'s defaults (Matplotlib's own rcParam values,
        "Auto" for Panel Aspect Ratio) -- panel layout/content is
        untouched."""
        defaults = GnoviFigure()
        for name in _LAYOUT_FIELDS:
            setattr(self._figure, name, getattr(defaults, name))
        self._sync_from_figure()
        self.changed.emit()

    # --- Apply / Cancel / Reset (see gui.dialogs.live_dialog.LiveDialog) ---

    def capture_state(self) -> dict:
        """Snapshot the margin/spacing fields this panel edits, for a
        hosting LiveDialog's Cancel to restore on close."""
        return {name: getattr(self._figure, name) for name in _LAYOUT_FIELDS}

    def restore_state(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self._figure, name, value)
        self._sync_from_figure()
        self.changed.emit()
