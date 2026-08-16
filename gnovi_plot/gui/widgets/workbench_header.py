from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from gnovi_plot.plotting.figure import GnoviFigure


class WorkbenchHeader(QWidget):
    """Small, restrained application-chrome strip docked directly above the
    central plotting/figure work area -- the "Workbench" (GNOVI Studio =
    the application, Workbench = this central plotting/work area, Graph
    Library = reusable saved graphs, Project = complete saved work; see
    `gui.main_window.MainWindow`).

    Deliberately minimal: a "WORKBENCH" label plus the current panel
    layout (e.g. "2 x 2"), never a large title -- plotting space stays the
    priority. Also deliberately NOT a duplicate of the per-panel
    "P1 - ACTIVE" badge (see `gui.widgets.plot_canvas._ActivePanelBadge`,
    unchanged): this header names the WORKSPACE as a whole; the badge names
    which PANEL within it is currently active.

    A plain Qt widget, docked above (never inside/over) the plot canvas and
    never added to the Matplotlib Figure -- like the active-panel badge, it
    is structurally absent from Matplotlib's own toolbar "Save" and from
    every `export.figure_export` format; application chrome only, never
    scientific output. `refresh(figure)` is the only thing that changes it
    -- call it whenever `figure.layout` may have changed (see
    `MainWindow._refresh_active_panel_context`, its one caller).
    """

    def __init__(self, figure: GnoviFigure, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkbenchHeader")

        self.title_label = QLabel("WORKBENCH")
        self.title_label.setObjectName("WorkbenchHeaderLabel")

        self.layout_label = QLabel()
        self.layout_label.setObjectName("WorkbenchHeaderLayoutLabel")
        self.layout_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.addWidget(self.title_label)
        layout.addStretch(1)
        layout.addWidget(self.layout_label)

        self.refresh(figure)

    def refresh(self, figure: GnoviFigure) -> None:
        rows, cols = figure.layout
        self.layout_label.setText(f"{rows} × {cols}")
