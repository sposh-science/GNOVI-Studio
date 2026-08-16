from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel

from gnovi_plot.plotting.figure import GnoviFigure


class ActivePanelLabel(QLabel):
    """Compact, read-only "Active panel: Panel N" line shown near the top of
    every left-drawer page whose controls act on a `GnoviFigure`'s active
    panel (Plot/Series/Figure/Layout/Axes) -- purely a display of
    `GnoviFigure.active_panel_index` (the existing single source of truth
    for "which panel is active", see `GnoviFigure.set_active_panel`), never
    a second one. Callers refresh it via `refresh(figure)` alongside their
    own existing refresh path (panel switch, layout change, project
    load/new, undo/redo) -- it holds no state of its own.
    """

    def __init__(self, figure: GnoviFigure, parent=None) -> None:
        super().__init__(parent)
        font = QFont(self.font())
        font.setBold(True)
        self.setFont(font)
        self.refresh(figure)

    def refresh(self, figure: GnoviFigure) -> None:
        self.setText(f"Active panel: Panel {figure.active_panel_index + 1}")
