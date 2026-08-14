from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMessageBox, QSplitter, QTableView

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.gui.widgets.dataframe_table_model import DataFrameTableModel
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNOVI PLOT")
        self.resize(1024, 768)

        self.dataset_manager = DatasetManager()

        self.plot_canvas = PlotCanvas(self)
        toolbar = NavigationToolbar2QT(self.plot_canvas, self)
        self.addToolBar(toolbar)

        self.preview_table = QTableView()
        self.preview_model = DataFrameTableModel()
        self.preview_table.setModel(self.preview_model)
        self.preview_table.setEditTriggers(QTableView.NoEditTriggers)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.plot_canvas)
        right_splitter.addWidget(self.preview_table)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        self.dataset_panel = DatasetPanel(self.dataset_manager)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.dataset_panel)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        self.setCentralWidget(main_splitter)

        self.dataset_panel.dataset_selected.connect(self._on_dataset_selected)
        self.dataset_panel.add_to_plot_requested.connect(self._on_add_to_plot)
        self.dataset_panel.clear_plot_requested.connect(self.plot_canvas.clear_plot)

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

    def _on_dataset_selected(self, dataset):
        self.preview_model.set_dataframe(dataset.dataframe if dataset is not None else None)

    def _on_add_to_plot(self, dataset, x_col, y_col):
        try:
            x, y = numeric_xy(dataset.dataframe, x_col, y_col)
        except KeyError as exc:
            QMessageBox.critical(self, "Plot Error", f"Column not found: {exc}")
            return
        except InsufficientNumericDataError as exc:
            QMessageBox.critical(self, "Plot Error", str(exc))
            return

        self.plot_canvas.add_line(x, y, label=f"{dataset.name} ({y_col})")
