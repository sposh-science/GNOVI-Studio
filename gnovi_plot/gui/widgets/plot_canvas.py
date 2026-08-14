from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gnovi_plot.data.numeric import numeric_column, numeric_xy
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries, PlotType


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        figure = Figure()
        super().__init__(figure)
        self.setParent(parent)
        self.axes = figure.add_subplot(111)

    def render(self, figure: GnoviFigure) -> None:
        """Fully redraw the axes from a GnoviFigure. Reading a series' data via
        numeric_xy/numeric_column never mutates the underlying Dataset.dataframe.
        """
        self.axes.cla()

        for series in figure.series:
            if series.visible:
                self._draw_series(series)

        self.axes.set_title(figure.title)
        self.axes.set_xlabel(figure.xlabel)
        self.axes.set_ylabel(figure.ylabel)

        if figure.xlim is not None:
            self.axes.set_xlim(*figure.xlim)
        if figure.ylim is not None:
            self.axes.set_ylim(*figure.ylim)
        if figure.xlim is None and figure.ylim is None:
            self.axes.autoscale(enable=True, axis="both")
        elif figure.xlim is None:
            self.axes.autoscale(enable=True, axis="x")
        elif figure.ylim is None:
            self.axes.autoscale(enable=True, axis="y")

        self.axes.grid(figure.grid)

        if figure.legend_visible:
            handles, _labels = self.axes.get_legend_handles_labels()
            if handles:
                self.axes.legend(loc=figure.legend_loc)
        else:
            legend = self.axes.get_legend()
            if legend is not None:
                legend.remove()

        self.draw_idle()

    def _draw_series(self, series: PlotSeries) -> None:
        if series.plot_type == PlotType.LINE:
            x, y = numeric_xy(series.dataframe, series.x_column, series.y_column)
            self.axes.plot(
                x,
                y,
                label=series.label,
                color=series.color,
                linewidth=series.line_width,
                linestyle=series.line_style,
                marker=series.marker or "",
                markersize=series.marker_size,
                alpha=series.alpha,
            )
        elif series.plot_type == PlotType.SCATTER:
            x, y = numeric_xy(series.dataframe, series.x_column, series.y_column)
            self.axes.scatter(
                x,
                y,
                label=series.label,
                color=series.color,
                marker=series.marker or "o",
                s=series.marker_size**2,
                alpha=series.alpha,
            )
        elif series.plot_type == PlotType.HISTOGRAM:
            values = numeric_column(series.dataframe, series.x_column)
            self.axes.hist(
                values,
                bins=series.bins,
                label=series.label,
                color=series.color,
                alpha=series.alpha,
            )
