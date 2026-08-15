from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QMainWindow, QMessageBox, QScrollArea, QSplitter, QTableView

from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.gui.widgets.data_tools_panel import DataToolsPanel
from gnovi_plot.gui.widgets.dataframe_table_model import DataFrameTableModel
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas
from gnovi_plot.gui.widgets.plot_series_panel import PlotSeriesPanel
from gnovi_plot.plotting.figure import GnoviFigure

# Fraction of the screen's available geometry the main window occupies at
# startup. Centered rather than maximized, and always derived from the
# actual screen -- never a fixed resolution.
_STARTUP_SCREEN_FRACTION = 0.92

_FALLBACK_AVAILABLE_GEOMETRY = QRect(0, 0, 1280, 800)


def compute_initial_geometry(available: QRect, fraction: float = _STARTUP_SCREEN_FRACTION) -> QRect:
    """Return a geometry centered within `available`, scaled by `fraction`.

    Always fits inside `available` for any fraction in (0, 1], regardless
    of the screen's actual resolution.
    """
    fraction = min(max(fraction, 0.1), 1.0)
    width = max(1, int(available.width() * fraction))
    height = max(1, int(available.height() * fraction))
    x = available.x() + (available.width() - width) // 2
    y = available.y() + (available.height() - height) // 2
    return QRect(x, y, width, height)


def _wrap_scrollable(content) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    return scroll


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNOVI PLOT")

        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else _FALLBACK_AVAILABLE_GEOMETRY
        geometry = compute_initial_geometry(available)
        self.setGeometry(geometry)

        self.dataset_manager = DatasetManager()
        self.figure_model = GnoviFigure()

        self.plot_canvas = PlotCanvas(self)
        toolbar = NavigationToolbar2QT(self.plot_canvas, self)
        self.addToolBar(toolbar)

        self.preview_table = QTableView()
        self.preview_model = DataFrameTableModel()
        self.preview_table.setModel(self.preview_model)
        self.preview_table.setEditTriggers(QTableView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)

        self.dataset_panel = DatasetPanel(self.dataset_manager, self.preview_table)
        self.series_panel = PlotSeriesPanel(self.figure_model)
        self.properties_panel = FigurePropertiesPanel(self.figure_model)
        self.data_tools_panel = DataToolsPanel(self.preview_table)
        self.properties_section = CollapsibleSection("Figure Properties", self.properties_panel)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(self.dataset_panel)
        left_splitter.addWidget(self.series_panel)
        left_splitter.addWidget(self.properties_section)
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(2, 1)
        self.left_scroll = _wrap_scrollable(left_splitter)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.preview_table)
        right_splitter.addWidget(self.data_tools_panel)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)
        self.right_scroll = _wrap_scrollable(right_splitter)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.left_scroll)
        main_splitter.addWidget(self.plot_canvas)
        main_splitter.addWidget(self.right_scroll)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 2)
        left_width = int(geometry.width() * 0.24)
        right_width = int(geometry.width() * 0.21)
        center_width = max(geometry.width() - left_width - right_width, 0)
        main_splitter.setSizes([left_width, center_width, right_width])
        self.main_splitter = main_splitter

        self.setCentralWidget(main_splitter)

        self.dataset_panel.dataset_selected.connect(self._on_dataset_selected)
        self.dataset_panel.add_to_plot_requested.connect(self._on_add_to_plot)
        self.dataset_panel.clear_plot_requested.connect(self._on_clear_plot)
        self.series_panel.changed.connect(self._rerender)
        self.properties_panel.changed.connect(self._rerender)
        self.data_tools_panel.transformation_applied.connect(self._on_transformation_applied)

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

        self.toggle_controls_action = view_menu.addAction("Controls")
        self.toggle_controls_action.setCheckable(True)
        self.toggle_controls_action.setChecked(True)
        self.toggle_controls_action.toggled.connect(self._on_toggle_controls)

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
        self.right_scroll.setVisible(visible)

    def _on_toggle_controls(self, visible: bool) -> None:
        self.left_scroll.setVisible(visible)

    def _on_dataset_selected(self, dataset):
        self.preview_model.set_dataframe(dataset.dataframe if dataset is not None else None)
        self.data_tools_panel.set_dataset(dataset)

    def _on_transformation_applied(self, dataset, row_set_changed: bool) -> None:
        self.preview_model.set_dataframe(dataset.dataframe)
        self.dataset_panel.refresh_columns()
        if row_set_changed:
            self.dataset_panel.reset_manual_cycles()

        newly_stale = self.figure_model.invalidate_series_for_dataset(dataset, row_set_changed)
        if newly_stale:
            self.series_panel.refresh()
        self._rerender()

        if newly_stale:
            names = "\n".join(f"- {s.label}" for s in newly_stale)
            QMessageBox.information(
                self,
                "Plot Series Invalidated",
                f"The working data for '{dataset.name}' changed in a way that invalidates "
                f"{len(newly_stale)} plot series (a row range no longer applies, or a "
                f"column it used was removed):\n\n{names}\n\n"
                "These are no longer drawn. Remove and re-add them against the updated "
                "working data.",
            )

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
