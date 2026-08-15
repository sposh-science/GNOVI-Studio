from __future__ import annotations

from typing import Protocol

from matplotlib.axes import Axes

from gnovi_plot.plotting.figure import GnoviFigure


class PlotBackend(Protocol):
    """Minimal interface a rendering backend must provide: fully redraw a
    sequence of Axes (one per `figure.panels` entry, same order) from a
    GnoviFigure. Kept intentionally small -- only what
    `matplotlib_backend.py` needs today; not designed against a hypothetical
    future (e.g. PyVista) backend before one exists.
    """

    def render_figure(self, axes_list: list[Axes], figure: GnoviFigure) -> None: ...
