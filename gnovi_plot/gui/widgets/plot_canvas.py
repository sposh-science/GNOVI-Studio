from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        figure = Figure()
        super().__init__(figure)
        self.setParent(parent)
        self.axes = figure.add_subplot(111)
