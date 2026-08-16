from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from gnovi_plot.plotting.figure import GnoviFigure


class WorkbenchHeader(QWidget):
    """Small, restrained application-chrome strip docked directly above the
    central plotting/figure work area -- the "Workbench" (GNOVI Studio =
    the application, Workbench = one independent plotting workspace/page,
    Graph Library = reusable saved graphs, Project = datasets + Graph
    Library + multiple Workbenches; see `gui.main_window.MainWindow`).

    Deliberately minimal: "WORKBENCH · <name> · <rows> x <cols>", never a
    large title -- plotting space stays the priority. Also deliberately
    NOT a duplicate of the per-panel "P1 - ACTIVE" badge (see
    `gui.widgets.plot_canvas._ActivePanelBadge`, unchanged) or of the
    Workbench tab strip above it (see `gui.widgets.workbench_tabs
    .WorkbenchTabBar`, the actual switcher): this header just names the
    currently active Workbench and its panel layout; it switches nothing
    itself.

    A plain Qt widget, docked above (never inside/over) the plot canvas and
    never added to the Matplotlib Figure -- like the active-panel badge, it
    is structurally absent from Matplotlib's own toolbar "Save" and from
    every `export.figure_export` format; application chrome only, never
    scientific output. `refresh(name, figure)` is the only thing that
    changes it -- call it whenever the active Workbench's name/identity or
    `figure.layout` may have changed (see
    `MainWindow._refresh_active_panel_context`, its one caller).
    """

    def __init__(self, workbench_name: str, figure: GnoviFigure, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkbenchHeader")

        self.label = QLabel()
        self.label.setObjectName("WorkbenchHeaderLabel")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.addWidget(self.label)
        layout.addStretch(1)

        self.refresh(workbench_name, figure)

    def refresh(self, workbench_name: str, figure: GnoviFigure) -> None:
        rows, cols = figure.layout
        self.label.setText(f"WORKBENCH · {workbench_name} · {rows} × {cols}")
