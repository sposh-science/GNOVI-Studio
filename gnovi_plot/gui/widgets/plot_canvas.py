from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gnovi_plot.plotting.backends.matplotlib_backend import render_figure
from gnovi_plot.plotting.figure import GnoviFigure


class PlotCanvas(FigureCanvasQTAgg):
    """Interactive multi-panel Matplotlib canvas embedded in the GUI.

    Keeps the Matplotlib Axes grid in sync with `GnoviFigure.layout`, but
    delegates the actual drawing to `plotting.backends.matplotlib_backend`
    so the exact same code path renders on-screen and into exported files.
    On-screen canvas pixel size never determines export resolution -- export
    always builds its own correctly-sized Figure (see `export.figure_export`).
    """

    def __init__(self, parent=None):
        figure = Figure()
        super().__init__(figure)
        self.setParent(parent)
        self._layout: tuple[int, int] | None = None
        self.axes_list: list = []
        self._ensure_layout((1, 1))

    def _ensure_layout(self, layout: tuple[int, int]) -> None:
        if layout == self._layout:
            return
        rows, cols = layout
        self.figure.clear()
        self.axes_list = list(self.figure.subplots(rows, cols, squeeze=False).flat)
        self._layout = layout

    @property
    def axes(self):
        """Backward-compatible alias for the first panel's Axes."""
        return self.axes_list[0]

    def active_axes(self, figure: GnoviFigure):
        """The Axes for `figure`'s currently active panel."""
        return self.axes_list[figure.active_panel_index]

    def render(self, figure: GnoviFigure) -> None:
        """Fully redraw every panel from `figure`. Reading a series' data
        never mutates the underlying Dataset.dataframe (see
        plotting.backends.matplotlib_backend)."""
        self._ensure_layout(figure.layout)
        render_figure(self.axes_list, figure)
        self.figure.tight_layout()
        self.draw_idle()
