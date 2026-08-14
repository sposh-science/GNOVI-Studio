from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtWidgets import QMainWindow, QMessageBox

from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNOVI PLOT")
        self.resize(1024, 768)

        self.plot_canvas = PlotCanvas(self)
        self.setCentralWidget(self.plot_canvas)

        toolbar = NavigationToolbar2QT(self.plot_canvas, self)
        self.addToolBar(toolbar)

        self._create_menu()

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction("About GNOVI PLOT")
        about_action.triggered.connect(self._show_about)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About GNOVI PLOT",
            "GNOVI PLOT\nScientific Plotting & Analysis",
        )
