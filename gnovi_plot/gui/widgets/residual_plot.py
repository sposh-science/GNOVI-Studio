from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from gnovi_plot.analysis.results import ResidualData

_ZERO_LINE_COLOR = "#8a8f99"
_POINT_COLOR = "#1f77b4"


class ResidualPlotWidget(QWidget):
    """A small, self-contained residual scatter plot: `x` vs. `observed -
    fitted`, discrete points (never a connected line -- residuals are
    independent per-point quantities, not a curve), with a horizontal
    zero-reference line.

    Deliberately independent of `GnoviFigure`/`Panel`/`PlotSeries`/the
    Workbench export path: this is analysis-result visualization only,
    drawn on its own bare Matplotlib `Figure` (the same
    `FigureCanvasQTAgg` primitive `gui.widgets.plot_canvas.PlotCanvas`
    itself is built on), never routed through
    `plotting.backends.matplotlib_backend`. It cannot leak into the
    scientific Workbench figure or its WYSIWYG export because it never
    touches either.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._figure = Figure(figsize=(4, 2.5), tight_layout=True)
        self._axes = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def plot_residuals(self, residual_data: ResidualData, x_label: str = "x", y_label: str = "Residual") -> None:
        """Redraw for `residual_data` -- discrete scatter points plus a
        zero-reference line. `x_label`/`y_label` should be the source
        series' actual column names (e.g. "Potential (V)"), not generic
        placeholders, wherever the caller has them available."""
        self._axes.clear()
        self._axes.axhline(0.0, color=_ZERO_LINE_COLOR, linestyle="--", linewidth=1.0, zorder=1)
        self._axes.scatter(
            residual_data.x, residual_data.residuals, s=18, color=_POINT_COLOR, zorder=2
        )
        self._axes.set_xlabel(x_label)
        self._axes.set_ylabel(y_label)
        self._canvas.draw_idle()

    def clear(self) -> None:
        self._axes.clear()
        self._canvas.draw_idle()
