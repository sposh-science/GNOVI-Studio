from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTableView,
    QToolBar,
)

from gnovi_plot.analysis.segments import InvalidRowRangeError, contiguous_row_range
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.data.numeric import InsufficientNumericDataError, numeric_xy
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
from gnovi_plot.gui.dialogs.live_dialog import LiveDialog
from gnovi_plot.gui.widgets.data_tools_panel import DataToolsPanel
from gnovi_plot.gui.widgets.dataframe_table_model import DataFrameTableModel
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS, FigureSizePanel
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas
from gnovi_plot.gui.widgets.plot_series_panel import PlotSeriesPanel
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries

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
        nav_toolbar = NavigationToolbar2QT(self.plot_canvas, self)
        self.addToolBar(nav_toolbar)

        self.preview_table = QTableView()
        self.preview_model = DataFrameTableModel()
        self.preview_table.setModel(self.preview_model)
        self.preview_table.setEditTriggers(QTableView.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)

        self.dataset_panel = DatasetPanel(self.dataset_manager, self.preview_table)
        self.series_panel = PlotSeriesPanel(self.figure_model)
        self.properties_panel = FigurePropertiesPanel(self.figure_model)
        self.figure_size_panel = FigureSizePanel(self.figure_model)
        self.data_tools_panel = DataToolsPanel(self.preview_table)

        # Universal figure-wide controls live in non-modal dialogs reachable
        # from the Figure/Panels menus and toolbar, not permanently-visible
        # sidebar blocks -- see the Figure/Panels menu handlers below. Each
        # dialog wraps a single persistent panel instance, so reopening it
        # always shows current, live state.
        self.axes_dialog = LiveDialog("Axes, Ticks & Legend", self.properties_panel, self)
        self.figure_size_dialog = LiveDialog("Figure Size, Panels & Typography", self.figure_size_panel, self)

        # Contextual, always-relevant controls stay directly in the sidebars.
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(self.dataset_panel)
        left_splitter.addWidget(self.series_panel)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
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
        self.dataset_panel.axis_preset_requested.connect(self._on_axis_preset_requested)
        self.series_panel.changed.connect(self._rerender)
        self.properties_panel.changed.connect(self._rerender)
        self.figure_size_panel.changed.connect(self._rerender)
        self.figure_size_panel.panel_switched.connect(self._on_panel_switched)
        self.data_tools_panel.transformation_applied.connect(self._on_transformation_applied)
        self.data_tools_panel.plot_selected_rows_requested.connect(self._on_plot_selected_rows)
        self.figure_size_panel.panel_labels_check.toggled.connect(self._sync_panel_labels_action)

        self._create_menu()
        self._create_toolbar()

    # --- Menus -----------------------------------------------------------

    def _create_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        data_menu = self.menuBar().addMenu("&Data")
        import_action = data_menu.addAction("Import Data…")
        import_action.triggered.connect(self._on_import_data)
        save_working_action = data_menu.addAction("Save Working Data…")
        save_working_action.triggered.connect(self._on_save_working_data)

        plot_menu = self.menuBar().addMenu("&Plot")
        add_to_plot_action = plot_menu.addAction("Add to Plot")
        add_to_plot_action.triggered.connect(lambda: self.dataset_panel.add_to_plot_button.click())
        clear_plot_action = plot_menu.addAction("Clear Plot")
        clear_plot_action.triggered.connect(lambda: self.dataset_panel.clear_plot_button.click())

        figure_menu = self.menuBar().addMenu("&Figure")
        figure_size_action = figure_menu.addAction("Figure Size && Ratio…")
        figure_size_action.triggered.connect(self._show_figure_size_dialog)
        publication_action = figure_menu.addAction("Publication Presets…")
        publication_action.triggered.connect(self._show_figure_size_dialog)
        typography_action = figure_menu.addAction("Typography…")
        typography_action.triggered.connect(self._show_figure_size_dialog)
        figure_menu.addSeparator()
        axes_action = figure_menu.addAction("Axes && Ticks…")
        axes_action.triggered.connect(self._show_axes_dialog)
        legend_action = figure_menu.addAction("Legend…")
        legend_action.triggered.connect(self._show_axes_dialog)
        figure_menu.addSeparator()
        export_action = figure_menu.addAction("Export Figure…")
        export_action.triggered.connect(self._on_export_figure)

        self.panels_menu = self.menuBar().addMenu("&Panels")
        self.layout_menu = self.panels_menu.addMenu("Layout")
        self.layout_menu.aboutToShow.connect(self._rebuild_layout_menu)
        self.active_panel_menu = self.panels_menu.addMenu("Active Panel…")
        self.active_panel_menu.aboutToShow.connect(self._rebuild_active_panel_menu)
        self.panels_menu.addSeparator()
        copy_style_action = self.panels_menu.addAction("Copy Active Panel Style to All Panels")
        copy_style_action.triggered.connect(self._on_copy_style_to_all_panels)
        self.panels_menu.addSeparator()
        self.panel_labels_action = self.panels_menu.addAction("Panel Labels On/Off")
        self.panel_labels_action.setCheckable(True)
        self.panel_labels_action.toggled.connect(self._on_toggle_panel_labels)

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

    def _rebuild_layout_menu(self) -> None:
        self.layout_menu.clear()
        current_index = self.figure_size_panel.layout_combo.currentIndex()
        for i, (text, _dims) in enumerate(LAYOUT_PRESETS):
            action = self.layout_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(i == current_index)
            action.triggered.connect(lambda _checked=False, index=i: self._set_layout(index))

    def _rebuild_active_panel_menu(self) -> None:
        self.active_panel_menu.clear()
        current_index = self.figure_model.active_panel_index
        for i in range(len(self.figure_model.panels)):
            action = self.active_panel_menu.addAction(f"Panel {i + 1}")
            action.setCheckable(True)
            action.setChecked(i == current_index)
            action.triggered.connect(lambda _checked=False, index=i: self._set_active_panel(index))

    def _show_about(self):
        QMessageBox.about(
            self,
            "About GNOVI PLOT",
            "GNOVI PLOT\nScientific Plotting & Analysis",
        )

    # --- Toolbar -----------------------------------------------------------

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        self.addToolBar(toolbar)

        import_action = toolbar.addAction("Import Data")
        import_action.triggered.connect(self._on_import_data)

        save_working_action = toolbar.addAction("Save Working Data")
        save_working_action.triggered.connect(self._on_save_working_data)

        export_action = toolbar.addAction("Export Figure")
        export_action.triggered.connect(self._on_export_figure)

        toolbar.addSeparator()

        self.toolbar_layout_combo = QComboBox()
        self.toolbar_layout_combo.addItems([text for text, _dims in LAYOUT_PRESETS])
        self.toolbar_layout_combo.setToolTip("Panel Layout")
        self.toolbar_layout_combo.currentIndexChanged.connect(self._on_toolbar_layout_changed)
        toolbar.addWidget(self.toolbar_layout_combo)

        self.toolbar_panel_combo = QComboBox()
        self.toolbar_panel_combo.setToolTip("Active Panel")
        self.toolbar_panel_combo.currentIndexChanged.connect(self._on_toolbar_panel_changed)
        toolbar.addWidget(self.toolbar_panel_combo)

        self._sync_toolbar_panel_controls()

    def _sync_toolbar_panel_controls(self) -> None:
        self.toolbar_layout_combo.blockSignals(True)
        self.toolbar_layout_combo.setCurrentIndex(self.figure_size_panel.layout_combo.currentIndex())
        self.toolbar_layout_combo.blockSignals(False)

        self.toolbar_panel_combo.blockSignals(True)
        self.toolbar_panel_combo.clear()
        for i in range(len(self.figure_model.panels)):
            self.toolbar_panel_combo.addItem(f"Panel {i + 1}")
        self.toolbar_panel_combo.setCurrentIndex(self.figure_model.active_panel_index)
        self.toolbar_panel_combo.blockSignals(False)

    def _on_toolbar_layout_changed(self, index: int) -> None:
        if index < 0 or index == self.figure_size_panel.layout_combo.currentIndex():
            return
        self._set_layout(index)

    def _on_toolbar_panel_changed(self, index: int) -> None:
        if index < 0 or index == self.figure_model.active_panel_index:
            return
        self._set_active_panel(index)

    # --- Shared handlers (menu, toolbar, and sidebar controls all call these) --

    def _on_import_data(self) -> None:
        self.dataset_panel.import_button.click()

    def _on_save_working_data(self) -> None:
        dataset = self.dataset_panel.current_dataset
        if dataset is None:
            QMessageBox.information(self, "Save Working Data", "Select a dataset first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Working Data", f"{dataset.name}.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            dataset.dataframe.to_csv(path, index=False)
        except OSError as exc:
            QMessageBox.critical(self, "Save Working Data", str(exc))

    def _show_figure_size_dialog(self) -> None:
        self.figure_size_dialog.show_raised()

    def _show_axes_dialog(self) -> None:
        self.axes_dialog.show_raised()

    def _set_layout(self, index: int) -> None:
        self.figure_size_panel.layout_combo.setCurrentIndex(index)

    def _set_active_panel(self, index: int) -> None:
        self.figure_size_panel.panel_combo.setCurrentIndex(index)

    def _on_copy_style_to_all_panels(self) -> None:
        self.figure_model.copy_active_panel_style_to_all()
        self._rerender()

    def _on_toggle_panel_labels(self, checked: bool) -> None:
        self.figure_size_panel.panel_labels_check.setChecked(checked)

    def _sync_panel_labels_action(self, checked: bool) -> None:
        self.panel_labels_action.blockSignals(True)
        self.panel_labels_action.setChecked(checked)
        self.panel_labels_action.blockSignals(False)

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

    def _on_plot_selected_rows(self, positions: list[int]) -> None:
        """Add a new PlotSeries scoped to the selected Data Preview rows to
        the active panel, without touching the dataset's raw or working
        data. Deliberately separate from Working Data transformations
        (Exclude/Keep Selection): no `Dataset` method is called here, so
        nothing is added to the transformation history and no existing
        series can be invalidated.
        """
        dataset = self.dataset_panel.current_dataset
        if dataset is None:
            return

        x_col = self.dataset_panel.x_combo.currentText()
        y_col = self.dataset_panel.y_combo.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "Plot Selected Rows", "Select X and Y columns to plot.")
            return

        try:
            row_range = contiguous_row_range(positions)
        except InvalidRowRangeError as exc:
            QMessageBox.warning(self, "Plot Selected Rows", str(exc))
            return

        start, end = row_range
        try:
            numeric_xy(dataset.dataframe.iloc[start:end], x_col, y_col)
        except (KeyError, InsufficientNumericDataError) as exc:
            QMessageBox.critical(self, "Plot Selected Rows", str(exc))
            return

        series = PlotSeries.line(
            dataset,
            x_col,
            y_col,
            label=f"{dataset.name} — rows {start}–{end - 1}",
            row_range=row_range,
        )
        self._on_add_to_plot([series])

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

    def _on_axis_preset_requested(self, preset: dict) -> None:
        panel = self.figure_model.active_panel
        panel.xlabel = preset.get("xlabel", panel.xlabel)
        panel.ylabel = preset.get("ylabel", panel.ylabel)
        self.properties_panel.refresh()
        self._rerender()

    def _on_panel_switched(self):
        self.series_panel.refresh()
        self.properties_panel.refresh()
        self._sync_toolbar_panel_controls()
        self._rerender()

    def _on_export_figure(self):
        dialog = ExportFigureDialog(self.figure_model, self)
        dialog.exec()

    def _rerender(self):
        self.plot_canvas.render(self.figure_model)
        active_axes = self.plot_canvas.active_axes(self.figure_model)
        self.properties_panel.sync_axes_limits(active_axes.get_xlim(), active_axes.get_ylim())
