import pandas as pd
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QItemSelection, QItemSelectionModel, QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QTableView, QToolBar, QVBoxLayout, QWidget

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.gui.main_window import MainWindow, compute_initial_geometry
from gnovi_plot.gui.styles import PlotTheme
from gnovi_plot.gui.widgets.collapsible_section import CollapsibleSection
from gnovi_plot.gui.widgets.dataset_panel import DatasetPanel
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS
from gnovi_plot.gui.widgets.plot_canvas import ReferenceCursorMode
from gnovi_plot.plotting.series import PlotSeries, PlotType


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


def _select_preview_rows(window, start, end_inclusive):
    model = window.preview_table.model()
    selection_model = window.preview_table.selectionModel()
    top_left = model.index(start, 0)
    bottom_right = model.index(end_inclusive, model.columnCount() - 1)
    selection_model.select(QItemSelection(top_left, bottom_right), QItemSelectionModel.ClearAndSelect)


# --- compute_initial_geometry -------------------------------------------------


def test_initial_geometry_fits_within_available_for_small_screen():
    available = QRect(0, 0, 1366, 768)
    geometry = compute_initial_geometry(available)

    assert geometry.left() >= available.left()
    assert geometry.top() >= available.top()
    assert geometry.right() <= available.right()
    assert geometry.bottom() <= available.bottom()


def test_initial_geometry_fits_within_available_for_large_screen():
    available = QRect(0, 0, 3840, 2160)
    geometry = compute_initial_geometry(available)

    assert geometry.width() <= available.width()
    assert geometry.height() <= available.height()
    assert geometry.right() <= available.right()
    assert geometry.bottom() <= available.bottom()


def test_initial_geometry_is_centered_and_not_fullscreen():
    available = QRect(0, 0, 1920, 1080)
    geometry = compute_initial_geometry(available)

    assert geometry.width() < available.width()
    assert geometry.height() < available.height()
    assert 0.9 <= geometry.width() / available.width() <= 0.95
    assert 0.9 <= geometry.height() / available.height() <= 0.95
    left_margin = geometry.left() - available.left()
    right_margin = available.right() - geometry.right()
    assert abs(left_margin - right_margin) <= 1


def test_initial_geometry_respects_nonzero_screen_origin():
    available = QRect(100, 50, 1366, 768)
    geometry = compute_initial_geometry(available)

    assert geometry.left() >= available.left()
    assert geometry.top() >= available.top()
    assert geometry.right() <= available.right()
    assert geometry.bottom() <= available.bottom()


def test_main_window_startup_geometry_never_exceeds_screen(qapp):
    window = MainWindow()
    screen = QGuiApplication.primaryScreen()
    available = screen.availableGeometry()

    assert window.geometry().right() <= available.right()
    assert window.geometry().bottom() <= available.bottom()
    assert window.geometry().left() >= available.left()
    assert window.geometry().top() >= available.top()


# --- View menu toggles ---------------------------------------------------------


def test_view_controls_action_toggles_the_left_tool_drawer(qapp):
    window = MainWindow()
    window.show()

    assert window.tool_drawer.isVisible() is True
    window.toggle_controls_action.setChecked(False)
    assert window.tool_drawer.isVisible() is False
    window.toggle_controls_action.setChecked(True)
    assert window.tool_drawer.isVisible() is True
    window.close()


def test_view_working_data_action_toggles_the_right_working_drawer(qapp):
    window = MainWindow()
    window.show()

    assert window.working_drawer.isVisible() is True
    window.toggle_working_data_action.setChecked(False)
    assert window.working_drawer.isVisible() is False
    window.toggle_working_data_action.setChecked(True)
    assert window.working_drawer.isVisible() is True
    window.close()


# --- LEFT tool strip / drawer: Data / Plot / Series (DSO-style) -----------------


def test_drawer_opens_the_data_page_by_default_on_startup(qapp):
    window = MainWindow()
    window.show()

    assert window.tool_drawer.active_key == "data"
    window.close()


def test_data_button_opens_the_data_page(qapp):
    window = MainWindow()
    window.show()
    window.tool_drawer.collapse()

    window.tool_drawer._buttons["data"].click()

    assert window.tool_drawer.active_key == "data"
    window.close()


def test_plot_button_opens_the_plot_page(qapp):
    window = MainWindow()
    window.show()

    window.tool_drawer._buttons["plot"].click()

    assert window.tool_drawer.active_key == "plot"
    window.close()


def test_series_button_opens_the_series_page(qapp):
    window = MainWindow()
    window.show()

    window.tool_drawer._buttons["series"].click()

    assert window.tool_drawer.active_key == "series"
    window.close()


def test_figure_button_opens_the_figure_page(qapp):
    window = MainWindow()
    window.show()

    window.tool_drawer._buttons["figure"].click()

    assert window.tool_drawer.active_key == "figure"
    assert window.tool_drawer._pages["figure"].isVisible() is True
    window.close()


def test_layout_button_opens_the_layout_page(qapp):
    window = MainWindow()
    window.show()

    window.tool_drawer._buttons["layout"].click()

    assert window.tool_drawer.active_key == "layout"
    assert window.tool_drawer._pages["layout"].isVisible() is True
    window.close()


def test_axes_button_opens_the_axes_page(qapp):
    window = MainWindow()
    window.show()

    window.tool_drawer._buttons["axes"].click()

    assert window.tool_drawer.active_key == "axes"
    assert window.tool_drawer._pages["axes"].isVisible() is True
    window.close()


def test_figure_menu_shortcuts_open_the_figure_and_axes_drawer_pages(qapp):
    """Figure Size/Publication/Typography and Axes/Legend menu entries used
    to open their own dialogs -- they're now shortcuts to the one place
    those controls live (see MainWindow._open_drawer_page)."""
    window = MainWindow()
    window.show()
    window.toggle_controls_action.setChecked(False)
    assert window.tool_drawer.isVisible() is False

    window._show_figure_size_dialog()

    assert window.tool_drawer.isVisible() is True
    assert window.tool_drawer.active_key == "figure"

    window._show_axes_dialog()

    assert window.tool_drawer.active_key == "axes"
    window.close()


def test_working_is_not_a_page_in_the_left_tool_drawer(qapp):
    """Working Data moved to its own RIGHT drawer -- the LEFT drawer only
    covers "what data/series/figure/layout/axes do I want to
    plot/configure?" (Data/Plot/Series/Figure/Layout/Axes)."""
    window = MainWindow()
    window.show()

    assert set(window.tool_drawer._buttons.keys()) == {
        "data",
        "plot",
        "series",
        "figure",
        "layout",
        "axes",
    }
    assert "working" not in window.tool_drawer._pages
    window.close()


def test_only_one_left_drawer_page_is_visible_at_a_time(qapp):
    window = MainWindow()
    window.show()
    drawer = window.tool_drawer
    drawer.collapse()

    for key in ("data", "plot", "series", "figure", "layout", "axes"):
        drawer._buttons[key].click()
        visible_pages = [k for k, page in drawer._pages.items() if page.isVisible()]
        assert visible_pages == [key]
        for other_key, button in drawer._buttons.items():
            assert button.isChecked() == (other_key == key)
    window.close()


def test_clicking_the_active_left_strip_button_collapses_the_drawer(qapp):
    window = MainWindow()
    window.show()
    drawer = window.tool_drawer
    drawer._buttons["series"].click()
    assert drawer.is_collapsed is False

    drawer._buttons["series"].click()

    assert drawer.is_collapsed is True
    assert drawer._stack.isVisible() is False
    assert drawer._buttons["series"].isChecked() is False
    window.close()


def test_switching_pages_preserves_dataset_panel_workflow_state(qapp):
    """Splitting DatasetPanel's Datasets/Add to Plot sections across two
    drawer pages must not disturb the panel's own column selectors -- the
    same DatasetPanel instance still backs both pages."""
    window = MainWindow()
    window.show()
    manager = window.dataset_manager
    dataset = _make_dataset()
    manager.add(dataset)
    window.dataset_panel._refresh_list(select_id=dataset.id)

    window.tool_drawer._buttons["plot"].click()
    assert window.dataset_panel.x_combo.count() == 2

    window.tool_drawer._buttons["data"].click()
    window.tool_drawer._buttons["plot"].click()
    assert window.dataset_panel.x_combo.count() == 2
    window.close()


# --- RIGHT Working Data drawer ---------------------------------------------------


def test_right_working_data_drawer_exists_with_a_single_working_page(qapp):
    window = MainWindow()
    window.show()

    assert set(window.working_drawer._buttons.keys()) == {"working"}
    assert window.working_drawer.active_key == "working"
    window.close()


def test_working_drawer_hosts_the_data_tools_panel(qapp):
    window = MainWindow()
    window.show()

    assert window.data_tools_panel.isVisible() is True
    assert window.working_drawer._pages["working"].isVisible() is True
    window.close()


def test_clicking_the_active_working_button_collapses_the_right_drawer(qapp):
    window = MainWindow()
    window.show()
    drawer = window.working_drawer
    assert drawer.is_collapsed is False  # open by default

    drawer._buttons["working"].click()

    assert drawer.is_collapsed is True
    assert drawer._buttons["working"].isChecked() is False
    window.close()


def test_working_drawer_collapse_and_reopen_preserves_dataset_selection_state(qapp):
    """Working Data status (e.g. the selected dataset's row-count readout)
    must survive a collapse/reopen cycle -- the DataToolsPanel widget is
    hidden, not destroyed."""
    window = MainWindow()
    window.show()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.dataset_panel._refresh_list(select_id=dataset.id)
    assert "Working Data: 3 / 3 rows" in window.data_tools_panel.row_count_label.text()

    window.working_drawer._buttons["working"].click()  # collapse
    window.working_drawer._buttons["working"].click()  # reopen

    assert "Working Data: 3 / 3 rows" in window.data_tools_panel.row_count_label.text()
    window.close()


# --- Independent left/right collapse and canvas-space reclaiming ---------------


def test_graph_expands_when_the_left_drawer_collapses(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    left_before, center_before, _right_before = window.main_splitter.sizes()

    window.tool_drawer._buttons["data"].click()  # collapse (already active)

    left_after, center_after, right_after = window.main_splitter.sizes()
    assert left_after == window.tool_drawer.strip_width
    assert center_after > center_before
    assert right_after == _right_before
    window.close()


def test_graph_expands_when_the_right_drawer_collapses(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    left_before, center_before, _right_before = window.main_splitter.sizes()

    window.working_drawer._buttons["working"].click()  # collapse (already active)

    left_after, center_after, right_after = window.main_splitter.sizes()
    assert right_after == window.working_drawer.strip_width
    assert center_after > center_before
    assert left_after == left_before
    window.close()


def test_left_and_right_drawers_collapse_independently(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)

    # Left open + Right open (default) -> collapse only the left.
    window.tool_drawer._buttons["data"].click()
    left, _center, right = window.main_splitter.sizes()
    assert left == window.tool_drawer.strip_width
    assert right != window.working_drawer.strip_width

    # Left closed + Right open -> also collapse the right.
    window.working_drawer._buttons["working"].click()
    left, _center, right = window.main_splitter.sizes()
    assert left == window.tool_drawer.strip_width
    assert right == window.working_drawer.strip_width

    # Left closed + Right closed -> reopen just the left.
    window.tool_drawer._buttons["data"].click()
    left, _center, right = window.main_splitter.sizes()
    assert left != window.tool_drawer.strip_width
    assert right == window.working_drawer.strip_width
    window.close()


def test_graph_occupies_almost_all_width_when_both_drawers_are_closed(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)

    window.tool_drawer._buttons["data"].click()
    window.working_drawer._buttons["working"].click()

    left, center, right = window.main_splitter.sizes()
    assert left == window.tool_drawer.strip_width
    assert right == window.working_drawer.strip_width
    assert center / window.width() > 0.8
    window.close()


def test_reopening_the_left_drawer_restores_the_previous_width_and_page_state(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    drawer = window.tool_drawer
    drawer._buttons["series"].click()
    open_sizes = window.main_splitter.sizes()

    drawer._buttons["series"].click()  # collapse
    drawer._buttons["series"].click()  # reopen -- same button, restores the page

    assert window.main_splitter.sizes() == open_sizes
    assert drawer.active_key == "series"
    assert drawer._pages["series"].isVisible() is True
    window.close()


def test_reopening_the_right_drawer_restores_the_previous_width(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    drawer = window.working_drawer
    open_sizes = window.main_splitter.sizes()

    drawer._buttons["working"].click()  # collapse
    drawer._buttons["working"].click()  # reopen

    assert window.main_splitter.sizes() == open_sizes
    assert drawer.active_key == "working"
    window.close()


def test_left_drawer_pages_fit_without_clipping_at_default_expanded_width(qapp):
    """Regression test: the Axes page (the widest left-drawer page, with its
    Top/Bottom/Left/Right spine checkboxes and many spin boxes) used to be
    narrower than its own minimumSizeHint at the drawer's default width,
    squeezing controls -- spin-box arrows, the "Right" spine label -- against
    the vertical scrollbar. The default width must clear every left-drawer
    page's minimum content width, not just Axes, with room left over for the
    scrollbar."""
    window = MainWindow()
    window.show()
    window.resize(1400, 900)

    for key in ("data", "plot", "series", "figure", "layout", "axes"):
        window.tool_drawer.show_page(key)
        scroll = window.tool_drawer._pages[key]
        content = scroll.widget()
        assert scroll.viewport().width() >= content.minimumSizeHint().width(), (
            f"'{key}' page content is clipped at the default drawer width"
        )
    window.close()


# --- Working Data workflows still function from the RIGHT drawer ---------------


def test_calculated_column_workflow_still_works_from_the_working_drawer(qapp, monkeypatch):
    window = MainWindow()
    window.show()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.dataset_panel._refresh_list(select_id=dataset.id)

    class _FakeDialog:
        Accepted = 1
        name = "z"
        formula = "x + y"

        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            return self.Accepted

    monkeypatch.setattr(
        "gnovi_plot.gui.widgets.data_tools_panel.CalculatedColumnDialog", _FakeDialog
    )

    window.data_tools_panel.calculated_column_button.click()

    assert "z" in dataset.calculated_columns
    assert window.dataset_panel.x_combo.findText("z") >= 0
    window.close()


def test_plot_selected_rows_still_works_from_the_working_drawer(qapp):
    window = MainWindow()
    window.show()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.dataset_panel._refresh_list(select_id=dataset.id)
    _select_preview_rows(window, 0, 2)

    window.data_tools_panel.plot_selected_rows_button.click()

    assert len(window.figure_model.series) == 1
    window.close()


# --- CollapsibleSection ---------------------------------------------------------


def test_collapsible_section_starts_expanded_by_default(qapp):
    content = QLabel("hello")
    section = CollapsibleSection("Title", content)
    section.show()

    assert section.is_expanded() is True
    assert content.isVisible() is True
    section.close()


def test_collapsible_section_collapse_hides_content_without_destroying_it(qapp):
    content = QWidget()
    layout = QVBoxLayout(content)
    inner_label = QLabel("inner")
    layout.addWidget(inner_label)
    section = CollapsibleSection("Title", content)
    section.show()

    section.set_expanded(False)

    assert section.is_expanded() is False
    assert content.isVisible() is False
    # widget still exists and is still the same object -- not destroyed
    assert section.content is content
    assert inner_label.parent() is content
    section.close()


def test_collapsible_section_expand_restores_content(qapp):
    content = QLabel("hello")
    section = CollapsibleSection("Title", content, expanded=False)
    section.show()

    assert content.isVisible() is False
    section.set_expanded(True)
    assert content.isVisible() is True
    section.close()


def test_collapsible_section_preserves_internal_dynamic_visibility(qapp):
    """Collapsing/expanding a section must not clobber visibility state a
    child widget manages itself (e.g. plot-type dependent controls in
    DatasetPanel)."""
    manager = DatasetManager()
    dataset = _make_dataset()
    manager.add(dataset)
    preview_table = QTableView()
    panel = DatasetPanel(manager, preview_table)
    panel.show()

    hist_index = panel.plot_type_combo.findData(PlotType.HISTOGRAM)
    panel.plot_type_combo.setCurrentIndex(hist_index)
    assert panel.y_combo.isVisible() is False

    panel.plot_section.set_expanded(False)
    panel.plot_section.set_expanded(True)

    # Y column selector must still be hidden -- collapsing/expanding the
    # section must not have reset it to visible.
    assert panel.y_combo.isVisible() is False
    # Widgets that were never plot-type-hidden remain reachable/visible.
    assert panel.import_button.isVisible() is True
    panel.close()


# --- Bottom panel (Data / Transformations / Results / Messages) ----------------


def test_bottom_panel_has_the_five_expected_tabs(qapp):
    window = MainWindow()
    assert [window.bottom_panel.tabText(i) for i in range(window.bottom_panel.count())] == [
        "Data",
        "Graphs",
        "Transformations",
        "Results",
        "Messages",
    ]
    window.close()


def test_bottom_panel_hosts_the_data_preview_table_and_transformation_history(qapp):
    window = MainWindow()
    assert window.preview_table.parent() is window.bottom_panel._data_tab
    assert window.data_tools_panel.history_group.parent() is window.bottom_panel._transformations_tab
    window.close()


def test_results_tab_is_an_inert_placeholder(qapp):
    """No fabricated analysis output -- just confirms the tab exists as a
    plain, non-interactive placeholder widget (no buttons/inputs)."""
    window = MainWindow()
    results_tab = window.bottom_panel.widget(3)
    assert isinstance(results_tab, QLabel)
    assert results_tab.findChildren(QWidget) == []
    window.close()


def test_toggle_bottom_panel_action_shows_and_hides_it(qapp):
    window = MainWindow()
    window.show()

    assert window.bottom_panel.isVisible() is True
    window.toggle_bottom_panel_action.setChecked(False)
    assert window.bottom_panel.isVisible() is False
    window.toggle_bottom_panel_action.setChecked(True)
    assert window.bottom_panel.isVisible() is True
    window.close()


def test_hiding_and_restoring_bottom_panel_does_not_change_the_figure_size(qapp):
    window = MainWindow()
    window.show()
    width_before = window.figure_model.figure_width_in
    height_before = window.figure_model.figure_height_in

    window.toggle_bottom_panel_action.setChecked(False)
    window.toggle_bottom_panel_action.setChecked(True)

    assert window.figure_model.figure_width_in == width_before
    assert window.figure_model.figure_height_in == height_before
    window.close()


def test_hiding_bottom_panel_remembers_the_split_for_restoring(qapp):
    window = MainWindow()
    window.show()
    sizes_before = window.center_splitter.sizes()

    window.toggle_bottom_panel_action.setChecked(False)
    window.toggle_bottom_panel_action.setChecked(True)

    assert window.center_splitter.sizes() == sizes_before
    window.close()


def test_plot_canvas_remains_the_dominant_center_widget(qapp):
    """The bottom panel starts noticeably smaller than the canvas -- the
    graph stays the dominant workspace."""
    window = MainWindow()
    window.show()
    canvas_height, bottom_height = window.center_splitter.sizes()

    assert canvas_height > bottom_height
    window.close()


# --- Toolbar stability / coordinate readout -------------------------------------


def test_navigation_and_main_toolbars_are_on_separate_rows(qapp):
    window = MainWindow()
    window.show()
    toolbars = window.findChildren(QToolBar)

    assert len(toolbars) == 2
    rows = {tb.geometry().y() for tb in toolbars}
    assert len(rows) == 2, "toolbars must not share a row -- see addToolBarBreak()"
    window.close()


def test_navigation_toolbar_has_no_builtin_coordinate_label(qapp):
    """Matplotlib's own toolbar coordinate label uses an Expanding size
    policy that reflows neighboring controls as digits change width --
    disabled via `coordinates=False` in favor of the fixed-width status-bar
    readout."""
    window = MainWindow()
    nav_toolbar = window.findChildren(QToolBar)[0]
    assert not hasattr(nav_toolbar, "locLabel")
    window.close()


def test_coord_label_lives_in_the_status_bar_with_a_stable_minimum_width(qapp):
    window = MainWindow()
    assert window.coord_label in window.statusBar().findChildren(QLabel)
    assert window.coord_label.minimumWidth() > 0
    window.close()


def test_mouse_move_over_axes_updates_coord_label_and_leave_clears_it(qapp):
    window = MainWindow()
    window.show()
    ax = window.plot_canvas.active_axes(window.figure_model)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    event.xdata, event.ydata = 5.0, 5.0
    window._on_mouse_move(event)
    assert window.coord_label.text() == "x = 5, y = 5"

    window._on_mouse_leave(event)
    assert window.coord_label.text() == ""
    window.close()


def test_mouse_move_outside_axes_clears_coord_label(qapp):
    window = MainWindow()
    window.show()
    event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
    event.inaxes = None
    window._on_mouse_move(event)
    assert window.coord_label.text() == ""
    window.close()


def test_mouse_move_updates_coord_label_for_every_panel_in_a_multi_panel_layout(qapp):
    """The status-bar coordinate readout must track whichever panel the
    pointer is actually over, regardless of which panel is the *active*
    one (the panel Figure/Series edits target) -- see main_window's
    `_on_mouse_move`, which keys off Matplotlib's own `event.inaxes` rather
    than `figure.active_panel_index`."""
    window = MainWindow()
    window.show()
    layout_index = next(i for i, (text, _dims) in enumerate(LAYOUT_PRESETS) if text == "2 x 2")
    window.figure_size_panel.layout_combo.setCurrentIndex(layout_index)
    assert window.figure_model.active_panel_index == 0  # panel 1 stays active throughout

    for i, ax in enumerate(window.plot_canvas.axes_list):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
        event.inaxes = ax
        event.xdata, event.ydata = float(i), float(i)
        window._on_mouse_move(event)
        assert window.coord_label.text() == f"x = {i}, y = {i}"

    window.close()


def test_mouse_leave_clears_coord_label_in_a_multi_panel_layout(qapp):
    window = MainWindow()
    window.show()
    layout_index = next(i for i, (text, _dims) in enumerate(LAYOUT_PRESETS) if text == "2 x 2")
    window.figure_size_panel.layout_combo.setCurrentIndex(layout_index)

    ax = window.plot_canvas.axes_list[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    event.xdata, event.ydata = 3.0, 3.0
    window._on_mouse_move(event)
    assert window.coord_label.text() != ""

    window._on_mouse_leave(event)
    assert window.coord_label.text() == ""
    window.close()


def test_toolbar_controls_are_unaffected_by_mouse_move(qapp):
    """Moving the mouse over the graph must never shift the fixed toolbar
    controls -- only the status-bar coord_label text changes."""
    window = MainWindow()
    window.show()
    toolbar_widgets = [
        window.toolbar_layout_combo,
        window.toolbar_panel_combo,
        window.toolbar_theme_combo,
    ]
    geometries_before = [w.geometry() for w in toolbar_widgets]

    ax = window.plot_canvas.active_axes(window.figure_model)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    event.xdata, event.ydata = 1234.5678, -9876.5432
    window._on_mouse_move(event)

    geometries_after = [w.geometry() for w in toolbar_widgets]
    assert geometries_after == geometries_before
    window.close()


# --- Undo/Redo (figure content only) --------------------------------------------


def test_undo_redo_actions_start_disabled(qapp):
    window = MainWindow()
    assert window.undo_action.isEnabled() is False
    assert window.redo_action.isEnabled() is False
    window.close()


def test_a_figure_content_change_enables_undo_and_clears_redo(qapp):
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    series = PlotSeries.line(dataset, "x", "y")

    window._on_add_to_plot([series])

    assert window.undo_action.isEnabled() is True
    assert window.redo_action.isEnabled() is False
    window.close()


def test_undo_reverts_the_last_series_addition(qapp):
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    assert len(window.figure_model.series) == 1

    window._on_undo()

    assert len(window.figure_model.series) == 0
    assert window.redo_action.isEnabled() is True


def test_redo_reapplies_an_undone_change(qapp):
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    window._on_undo()

    window._on_redo()

    assert len(window.figure_model.series) == 1


def test_undo_reverts_a_panel_property_edit(qapp):
    window = MainWindow()
    window.properties_panel.title_edit.setText("Temperature")
    window.properties_panel._apply_title()
    assert window.figure_model.active_panel.title == "Temperature"

    window._on_undo()

    assert window.figure_model.active_panel.title == ""


def test_undo_reverts_a_layout_change(qapp):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    assert window.figure_model.layout == (2, 2)

    window._on_undo()

    assert window.figure_model.layout == (1, 1)


def test_switching_the_active_panel_is_not_an_undoable_step(qapp):
    """Pure navigation (which panel is active) must never show up as a
    spurious undo entry."""
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window._on_undo()  # revert the layout change itself
    assert window.undo_action.isEnabled() is False

    window.figure_size_panel.panel_combo.setCurrentIndex(0)

    assert window.undo_action.isEnabled() is False
    window.close()


def test_dataset_transformations_are_not_undoable_via_the_figure_undo_stack(qapp):
    """Working Data mutations keep their own Reset Working Data recovery
    path and Transformation History -- they must never push an entry onto
    the figure Undo stack (see gui.undo_manager's scoping rationale)."""
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.dataset_panel._refresh_list(select_id=dataset.id)

    dataset.add_calculated_column("z", "x + y")
    window._on_transformation_applied(dataset, False)

    assert window.undo_action.isEnabled() is False
    window.close()


def test_undo_history_is_bounded_by_a_snapshot_cap(qapp):
    window = MainWindow()
    for i in range(60):
        window.properties_panel.title_edit.setText(f"t{i}")
        window.properties_panel._apply_title()

    undo_count = 0
    while window.undo_action.isEnabled():
        window._on_undo()
        undo_count += 1

    assert undo_count <= 50  # UndoManager's default max_entries
    window.close()


# --- Click subplot to make it active --------------------------------------------


def test_clicking_inside_a_panel_makes_it_the_active_panel(qapp):
    window = MainWindow()
    window.show()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    assert window.figure_model.active_panel_index == 0
    ax = window.plot_canvas.axes_list[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    event = MouseEvent("button_press_event", window.plot_canvas, 1, 1)
    event.inaxes = ax

    window._on_canvas_click(event)

    assert window.figure_model.active_panel_index == 1
    window.close()


def test_clicking_outside_any_axes_does_nothing(qapp):
    window = MainWindow()
    window.show()
    event = MouseEvent("button_press_event", window.plot_canvas, 1, 1)
    event.inaxes = None

    window._on_canvas_click(event)  # must not raise

    assert window.figure_model.active_panel_index == 0
    window.close()


def test_clicking_a_panel_is_not_an_undoable_step(qapp):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2" -> 1 undo entry
    ax = window.plot_canvas.axes_list[1]
    event = MouseEvent("button_press_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    undo_enabled_before_click = window.undo_action.isEnabled()
    assert undo_enabled_before_click is True

    window._on_canvas_click(event)

    assert window.undo_action.isEnabled() == undo_enabled_before_click
    window._on_undo()
    assert window.undo_action.isEnabled() is False  # exactly one entry, from the layout change alone
    window.close()


# --- Reference cursor (Off / X line / Y line / Crosshair) ----------------------


def test_reference_cursor_defaults_to_off_and_menu_reflects_it(qapp):
    window = MainWindow()
    assert window._cursor_mode == ReferenceCursorMode.OFF
    assert window._cursor_actions[ReferenceCursorMode.OFF].isChecked() is True
    window.close()


def test_changing_cursor_mode_updates_canvas_menu_and_toolbar_together(qapp):
    window = MainWindow()

    window._on_cursor_mode_changed(ReferenceCursorMode.CROSSHAIR)

    assert window.plot_canvas._cursor_mode == ReferenceCursorMode.CROSSHAIR
    assert window._cursor_actions[ReferenceCursorMode.CROSSHAIR].isChecked() is True
    assert window.toolbar_cursor_combo.currentData() == ReferenceCursorMode.CROSSHAIR
    window.close()


def test_mouse_move_draws_the_reference_cursor_without_disturbing_the_coord_label(qapp):
    window = MainWindow()
    window.show()
    window._on_cursor_mode_changed(ReferenceCursorMode.CROSSHAIR)
    ax = window.plot_canvas.active_axes(window.figure_model)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    event.xdata, event.ydata = 5.0, 5.0

    window._on_mouse_move(event)

    assert window.coord_label.text() == "x = 5, y = 5"
    assert len(window.plot_canvas._cursor_artists) == 2
    window.close()


def test_mouse_leave_clears_both_coord_label_and_reference_cursor(qapp):
    window = MainWindow()
    window.show()
    window._on_cursor_mode_changed(ReferenceCursorMode.X_LINE)
    ax = window.plot_canvas.active_axes(window.figure_model)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    event.xdata, event.ydata = 5.0, 5.0
    window._on_mouse_move(event)
    assert len(window.plot_canvas._cursor_artists) == 1

    window._on_mouse_leave(event)

    assert window.coord_label.text() == ""
    assert window.plot_canvas._cursor_artists == []
    window.close()


# --- Enum-vs-string boundary: Plot Theme / Reference Cursor handlers -----------
#
# Regression coverage for a real crash: QComboBox.itemData() round-trips a
# str-subclassed Enum through QVariant and hands back a plain `str`, not the
# original Enum member (confirmed directly against PySide6, not assumed) --
# so the toolbar Plot Theme / Reference Cursor combos were handing
# `_on_theme_changed`/`_on_cursor_mode_changed` bare strings, and
# `mode.value` raised AttributeError. Both handlers now normalize incoming
# values to their enum type (see main_window._on_theme_changed's
# docstring). These tests exercise both the direct-string path (the exact
# failure mode) and the real combo-selection path end-to-end.


def test_theme_changed_accepts_a_plain_string_for_light(qapp):
    window = MainWindow()
    window._on_theme_changed("dark")  # start from a genuine non-default so "light" below is a real change
    window._on_theme_changed("light")
    assert window.figure_model.plot_theme == PlotTheme.LIGHT
    assert window._settings.value("plot_theme") == "light"
    window.close()


def test_theme_changed_accepts_a_plain_string_for_dark(qapp):
    window = MainWindow()
    window._on_theme_changed("dark")
    assert window.figure_model.plot_theme == PlotTheme.DARK
    assert window._settings.value("plot_theme") == "dark"
    window.close()


def test_theme_light_dark_light_round_trip_via_plain_strings(qapp):
    window = MainWindow()
    for value, expected in [("dark", PlotTheme.DARK), ("light", PlotTheme.LIGHT), ("dark", PlotTheme.DARK)]:
        window._on_theme_changed(value)
        assert window.figure_model.plot_theme == expected
        assert window._settings.value("plot_theme") == expected.value
    window.close()


def test_cursor_mode_changed_accepts_a_plain_string_for_off(qapp):
    window = MainWindow()
    window._on_cursor_mode_changed(ReferenceCursorMode.CROSSHAIR)  # start non-default

    window._on_cursor_mode_changed("off")

    assert window._cursor_mode == ReferenceCursorMode.OFF
    assert window.plot_canvas._cursor_mode == ReferenceCursorMode.OFF
    assert window._settings.value("reference_cursor") == "off"
    window.close()


def test_cursor_mode_changed_accepts_a_plain_string_for_x_line(qapp):
    window = MainWindow()
    window._on_cursor_mode_changed("x")
    assert window._cursor_mode == ReferenceCursorMode.X_LINE
    assert window._settings.value("reference_cursor") == "x"
    window.close()


def test_cursor_mode_changed_accepts_a_plain_string_for_y_line(qapp):
    window = MainWindow()
    window._on_cursor_mode_changed("y")
    assert window._cursor_mode == ReferenceCursorMode.Y_LINE
    assert window._settings.value("reference_cursor") == "y"
    window.close()


def test_cursor_mode_changed_accepts_a_plain_string_for_crosshair(qapp):
    window = MainWindow()
    window._on_cursor_mode_changed("crosshair")
    assert window._cursor_mode == ReferenceCursorMode.CROSSHAIR
    assert window._settings.value("reference_cursor") == "crosshair"
    window.close()


def test_cursor_crosshair_then_off_via_plain_strings(qapp):
    window = MainWindow()
    window._on_cursor_mode_changed("crosshair")
    assert window._cursor_mode == ReferenceCursorMode.CROSSHAIR

    window._on_cursor_mode_changed("off")

    assert window._cursor_mode == ReferenceCursorMode.OFF
    assert window.plot_canvas._cursor_mode == ReferenceCursorMode.OFF
    window.close()


def test_toolbar_theme_combo_selection_does_not_raise_and_updates_state(qapp):
    """Exercises the real bug path end-to-end: selecting an item in the
    toolbar combo fires `currentIndexChanged` -> `_on_toolbar_theme_changed`
    -> `itemData()`, which is exactly where Qt hands back a plain string
    instead of the `PlotTheme` that was stored via `addItem(label, mode)`."""
    window = MainWindow()
    dark_index = window.toolbar_theme_combo.findText("Dark")

    window.toolbar_theme_combo.setCurrentIndex(dark_index)  # must not raise

    assert window.figure_model.plot_theme == PlotTheme.DARK
    assert window._settings.value("plot_theme") == "dark"

    light_index = window.toolbar_theme_combo.findText("Light")
    window.toolbar_theme_combo.setCurrentIndex(light_index)

    assert window.figure_model.plot_theme == PlotTheme.LIGHT
    window.close()


def test_toolbar_cursor_combo_selection_does_not_raise_and_updates_state(qapp):
    window = MainWindow()
    crosshair_index = window.toolbar_cursor_combo.findText("Cursor: Crosshair")

    window.toolbar_cursor_combo.setCurrentIndex(crosshair_index)  # must not raise

    assert window._cursor_mode == ReferenceCursorMode.CROSSHAIR
    assert window._settings.value("reference_cursor") == "crosshair"
    window.close()


def test_toolbar_cursor_combo_labels_are_unambiguous(qapp):
    """See main_window._REFERENCE_CURSOR_TOOLBAR_LABELS -- a lone "Off" in
    the toolbar (with no adjacent title, unlike the View menu submenu) reads
    as ambiguous, so every toolbar item spells out "Cursor: "."""
    window = MainWindow()

    labels = [window.toolbar_cursor_combo.itemText(i) for i in range(window.toolbar_cursor_combo.count())]

    assert labels == ["Cursor: Off", "Cursor: X Line", "Cursor: Y Line", "Cursor: Crosshair"]
    window.close()


def test_theme_switching_while_crosshair_enabled_leaves_cursor_mode_untouched(qapp):
    window = MainWindow()
    window._on_cursor_mode_changed(ReferenceCursorMode.CROSSHAIR)

    window._on_theme_changed("dark")

    assert window.figure_model.plot_theme == PlotTheme.DARK
    assert window._cursor_mode == ReferenceCursorMode.CROSSHAIR
    assert window.plot_canvas._cursor_mode == ReferenceCursorMode.CROSSHAIR
    window.close()


def test_qsettings_receives_plain_serializable_strings_not_enum_objects(qapp):
    window = MainWindow()
    window._on_theme_changed(PlotTheme.DARK)
    window._on_cursor_mode_changed(ReferenceCursorMode.CROSSHAIR)

    assert type(window._settings.value("plot_theme")) is str
    assert type(window._settings.value("reference_cursor")) is str
    window.close()


def test_restoring_saved_settings_reconstructs_the_correct_enum_state(qapp):
    window = MainWindow()
    window._on_theme_changed(PlotTheme.DARK)
    window._on_cursor_mode_changed(ReferenceCursorMode.X_LINE)
    window.close()

    restored = MainWindow()

    assert restored.figure_model.plot_theme == PlotTheme.DARK
    assert isinstance(restored.figure_model.plot_theme, PlotTheme)
    assert restored._cursor_mode == ReferenceCursorMode.X_LINE
    assert isinstance(restored._cursor_mode, ReferenceCursorMode)
    restored.close()
