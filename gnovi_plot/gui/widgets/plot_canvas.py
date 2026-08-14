from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        figure = Figure()
        super().__init__(figure)
        self.setParent(parent)
        self.axes = figure.add_subplot(111)

    def add_line(self, x, y, label: str) -> None:
        """Plot x/y as a new line, retaining any curves already on the axes."""
        self.axes.plot(x, y, label=label)
        self._update_legend()
        self.axes.relim()
        self.axes.autoscale_view()
        self.draw_idle()

    def clear_plot(self) -> None:
        """Remove all plotted curves. Imported datasets are unaffected."""
        self.axes.cla()
        self.draw_idle()

    def _update_legend(self) -> None:
        handles, _ = self.axes.get_legend_handles_labels()
        if handles:
            self.axes.legend()
