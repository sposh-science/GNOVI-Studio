from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMessageBox, QSplitter, QTableView

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.dataframe_table_model import DataFrameTableModel
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas
from gnovi_plot.gui.widgets.plot_series_panel import PlotSeriesPanel
from gnovi_plot.plotting.figure import GnoviFigure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNOVI PLOT")
        self.resize(1280, 800)

        self.dataset_manager = DatasetManager()
        self.figure_model = GnoviFigure()

        self.plot_canvas = PlotCanvas(self)
        toolbar = NavigationToolbar2QT(self.plot_canvas, self)
        self.addToolBar(toolbar)

        self.preview_table = QTableView()
        self.preview_model = DataFrameTableModel()
        self.preview_table.setModel(self.preview_model)
        self.preview_table.setEditTriggers(QTableView.NoEditTriggers)

        self.dataset_panel = DatasetPanel(self.dataset_manager, self.preview_table)
        self.series_panel = PlotSeriesPanel(self.figure_model)
        self.properties_panel = FigurePropertiesPanel(self.figure_model)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(self.dataset_panel)
        left_splitter.addWidget(self.series_panel)
        left_splitter.addWidget(self.properties_panel)
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(2, 1)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(self.plot_canvas)
        main_splitter.addWidget(self.preview_table)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 2)
        main_splitter.setSizes([290, 710, 280])
        self.main_splitter = main_splitter

        self.setCentralWidget(main_splitter)

        self.dataset_panel.dataset_selected.connect(self._on_dataset_selected)
        self.dataset_panel.add_to_plot_requested.connect(self._on_add_to_plot)
        self.dataset_panel.clear_plot_requested.connect(self._on_clear_plot)
        self.series_panel.changed.connect(self._rerender)
        self.properties_panel.changed.connect(self._rerender)

        self._create_menu()

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        view_menu = self.menuBar().addMenu("&View")
        self.toggle_preview_action = view_menu.addAction("Data Preview")
        self.toggle_preview_action.setCheckable(True)
        self.toggle_preview_action.setChecked(True)
        self.toggle_preview_action.toggled.connect(self._on_toggle_preview)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction("About GNOVI PLOT")
        about_action.triggered.connect(self._show_about)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About GNOVI PLOT",
            "GNOVI PLOT\nScientific Plotting & Analysis",
        )

    def _on_toggle_preview(self, visible: bool) -> None:
        self.preview_table.setVisible(visible)

    def _on_dataset_selected(self, dataset):
        self.preview_model.set_dataframe(dataset.dataframe if dataset is not None else None)

    def _on_add_to_plot(self, series_list):
        last_id = None
        for series in series_list:
            self.figure_model.add_series(series)
            last_id = series.id
        self.series_panel.refresh(select_id=last_id)
        self._rerender()

    def _on_clear_plot(self):
        self.figure_model.clear_series()
        self.series_panel.refresh()
        self._rerender()

    def _rerender(self):
        self.plot_canvas.render(self.figure_model)
        self.properties_panel.sync_axes_limits(
            self.plot_canvas.axes.get_xlim(), self.plot_canvas.axes.get_ylim()
        )
