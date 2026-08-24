"""PR "Sidebar Navigation & 2D/3D Workflow Polish": visually grouped
DATA/PLOT/FORMAT/ANALYZE tool-strip sections, the Plot -> 2D user-facing
rename, adaptive Series/Axes page headings, and the removal of the
redundant read-only series list from the 3D creation page.

Pure information-architecture/GUI tests -- no model, renderer, or
serialization behavior changes in this milestone, so nothing here touches
`Panel3D`/`Series3D` fields or project persistence beyond confirming both
are unaffected.
"""

import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.styles import build_stylesheet
from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.gui.widgets.plot3d_panel import Plot3DPanel
from gnovi_plot.gui.widgets.plot_series_panel import PlotSeriesPanel
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Series3D


def _make_2d_dataset(name="d2"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


def _make_3d_dataset(name="d3"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0]})
    return Dataset(name=name, dataframe=df)


def _make_2d_panel():
    from gnovi_plot.plotting.figure import Panel

    panel = Panel()
    panel.add_series(PlotSeries.line(_make_2d_dataset(), "x", "y", label="2D series"))
    return panel


def _make_3d_panel():
    panel = Panel3D()
    panel.add_series(Series3D(dataset=_make_3d_dataset(), x_column="x", y_column="y", z_column="z", label="3D series"))
    return panel


def _make_mixed_figure():
    """Panel 1 = 2D, Panel 2 = 3D, Panel 3 = 2D -- the exact mixed
    2D/3D/2D scenario the PR spec calls for (its own Part 13)."""
    return GnoviFigure(panels=[_make_2d_panel(), _make_3d_panel(), _make_2d_panel()])


# --- SIDEBAR: exact page order, section headers, non-interactivity, terminology --


def test_left_drawer_page_order_matches_the_data_plot_format_analyze_grouping(qapp):
    window = MainWindow()

    assert list(window.tool_drawer._buttons.keys()) == [
        "data",
        "plot",
        "3d",
        "series",
        "axes",
        "figure",
        "layout",
        "analysis",
    ]
    window.close()


def test_left_drawer_strip_order_interleaves_section_headers_correctly(qapp):
    """The strip's exact top-to-bottom visual order: a DATA heading before
    "data", a PLOT heading before "2D"/"3D", a FORMAT heading before
    "series"/"axes"/"figure"/"layout", an ANALYZE heading before
    "analysis" -- section headers are pure visual grouping, not a nested
    tree, so every page button still appears exactly once, flat."""
    window = MainWindow()

    assert window.tool_drawer.strip_order == [
        ("section", "DATA"),
        ("page", "data"),
        ("section", "PLOT"),
        ("page", "plot"),
        ("page", "3d"),
        ("section", "FORMAT"),
        ("page", "series"),
        ("page", "axes"),
        ("page", "figure"),
        ("page", "layout"),
        ("section", "ANALYZE"),
        ("page", "analysis"),
    ]
    window.close()


def test_section_headers_are_plain_labels_not_buttons(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QToolButton

    window = MainWindow()
    strip = window.tool_drawer._strip

    headings = [w for w in strip.findChildren(QLabel) if w.objectName() == "ToolStripSectionLeft"]
    assert [h.text() for h in headings] == ["DATA", "PLOT", "FORMAT", "ANALYZE"]
    for heading in headings:
        assert not isinstance(heading, QToolButton)
        assert heading.focusPolicy() == Qt.NoFocus
    window.close()


def test_section_headers_never_become_active_pages_or_buttons(qapp):
    """Headers are never added to `_buttons`/`_pages` -- only the eight
    real page keys exist there, so `show_page`/the click handler can never
    target a section heading, and `show_page` on a heading's own text is
    silently a no-op (not in `_pages`)."""
    window = MainWindow()
    window.show()

    expected_keys = {"data", "plot", "3d", "series", "axes", "figure", "layout", "analysis"}
    assert set(window.tool_drawer._pages.keys()) == expected_keys
    assert set(window.tool_drawer._buttons.keys()) == expected_keys
    for heading_text in ("DATA", "PLOT", "FORMAT", "ANALYZE"):
        assert heading_text not in window.tool_drawer._pages
        assert heading_text not in window.tool_drawer._buttons

    active_before = window.tool_drawer.active_key
    window.tool_drawer.show_page("PLOT")  # not a real key -- must be a no-op
    assert window.tool_drawer.active_key == active_before
    window.close()


def test_plot_page_user_facing_label_is_now_2d(qapp):
    window = MainWindow()

    assert window.tool_drawer._buttons["plot"].text() == "2D"
    window.close()


def test_3d_page_user_facing_label_remains_3d(qapp):
    window = MainWindow()

    assert window.tool_drawer._buttons["3d"].text() == "3D"
    window.close()


def test_plot_key_and_page_still_work_unchanged_under_the_new_label(qapp):
    """Internal key/behavior is untouched by the rename -- only the
    user-facing button text changed (see PR spec's own "internal classes/
    files/functions may stay named Plot*" note)."""
    window = MainWindow()
    window.show()

    window.tool_drawer._buttons["plot"].click()

    assert window.tool_drawer.active_key == "plot"
    window.close()


# --- ADAPTIVE SERIES: 2D/3D heading, switching, no stale controls ----------------


def test_series_page_heading_reads_2d_series_for_a_2d_panel(qapp):
    figure = GnoviFigure(panels=[_make_2d_panel()])
    panel = PlotSeriesPanel(figure)

    assert "2D Series" in panel.list_section.toggle_button.text()
    assert panel._stack.currentWidget() is panel._page_2d


def test_series_page_heading_reads_3d_series_for_a_3d_panel(qapp):
    figure = GnoviFigure(panels=[_make_3d_panel()])
    panel = PlotSeriesPanel(figure)

    assert "3D Series" in panel.list_section_3d.toggle_button.text()
    assert panel._stack.currentWidget() is panel._page_3d


def test_series_page_switching_2d_3d_2d_shows_no_stale_controls(qapp):
    figure = _make_mixed_figure()
    panel = PlotSeriesPanel(figure)

    figure.set_active_panel(0)
    panel.refresh()
    assert panel._stack.currentWidget() is panel._page_2d
    assert panel._page_3d.isVisible() is False

    figure.set_active_panel(1)
    panel.refresh()
    assert panel._stack.currentWidget() is panel._page_3d
    assert panel._page_2d.isVisible() is False
    assert panel.series3d_list.count() == 1

    figure.set_active_panel(2)
    panel.refresh()
    assert panel._stack.currentWidget() is panel._page_2d
    assert panel._page_3d.isVisible() is False
    assert panel.series_list.count() == 1


# --- ADAPTIVE AXES: 2D/3D Axes heading, mixed switching, no stale controls -------


def test_axes_page_heading_reads_2d_axes_for_a_2d_panel(qapp):
    figure = GnoviFigure(panels=[_make_2d_panel()])
    panel = FigurePropertiesPanel(figure)

    assert panel.page_heading_2d.text() == "2D Axes"
    assert panel._stack.currentWidget() is panel._page_2d


def test_axes_page_heading_reads_3d_axes_and_view_for_a_3d_panel(qapp):
    figure = GnoviFigure(panels=[_make_3d_panel()])
    panel = FigurePropertiesPanel(figure)

    assert panel.page_heading_3d.text() == "3D Axes & View"
    assert panel._stack.currentWidget() is panel._page_3d


def test_axes_page_mixed_panel_switching_shows_the_correct_page_each_time(qapp):
    figure = _make_mixed_figure()
    panel = FigurePropertiesPanel(figure)

    figure.set_active_panel(0)
    panel.refresh()
    assert panel._stack.currentWidget() is panel._page_2d
    assert panel._page_3d.isVisible() is False

    figure.set_active_panel(1)
    panel.refresh()
    assert panel._stack.currentWidget() is panel._page_3d
    assert panel._page_2d.isVisible() is False

    figure.set_active_panel(2)
    panel.refresh()
    assert panel._stack.currentWidget() is panel._page_2d
    assert panel._page_3d.isVisible() is False


def test_mixed_2d_3d_2d_figure_series_and_axes_headings_follow_the_active_panel(qapp):
    """The full Part 13 scenario: Panel 1 = 2D, Panel 2 = 3D, Panel 3 = 2D,
    Series/Axes headings tracking the active panel each time -- the
    sidebar destinations themselves (Series, Axes) never change, only
    their page content/heading."""
    figure = _make_mixed_figure()
    series_panel = PlotSeriesPanel(figure)
    axes_panel = FigurePropertiesPanel(figure)

    figure.set_active_panel(0)
    series_panel.refresh()
    axes_panel.refresh()
    assert series_panel._stack.currentWidget() is series_panel._page_2d
    assert axes_panel.page_heading_2d.text() == "2D Axes"
    assert axes_panel._stack.currentWidget() is axes_panel._page_2d

    figure.set_active_panel(1)
    series_panel.refresh()
    axes_panel.refresh()
    assert series_panel._stack.currentWidget() is series_panel._page_3d
    assert axes_panel.page_heading_3d.text() == "3D Axes & View"
    assert axes_panel._stack.currentWidget() is axes_panel._page_3d

    figure.set_active_panel(2)
    series_panel.refresh()
    axes_panel.refresh()
    assert series_panel._stack.currentWidget() is series_panel._page_2d
    assert axes_panel._stack.currentWidget() is axes_panel._page_2d


# --- 3D CREATION: redundant series list removed, Clear 3D Plot kept -------------


def test_3d_creation_page_no_longer_exposes_a_series_list(qapp):
    """See `Plot3DPanel`'s own docstring: the read-only summary list was
    removed once the adaptive Series page was confirmed to fully cover
    selecting/renaming/styling/removing/clearing every Series3D it
    creates -- it offered no capability the Series page doesn't."""
    from gnovi_plot.data.dataset_manager import DatasetManager

    figure = GnoviFigure(panels=[_make_3d_panel()])
    panel = Plot3DPanel(DatasetManager(), figure)

    assert not hasattr(panel, "series_list")
    assert not hasattr(panel, "list_section")


def test_clear_3d_plot_button_still_exists_and_reflects_active_panel_series(qapp):
    from gnovi_plot.data.dataset_manager import DatasetManager

    empty_figure = GnoviFigure(panels=[_make_2d_panel()])
    panel = Plot3DPanel(DatasetManager(), empty_figure)
    assert panel.clear_button.text() == "Clear 3D Plot"
    assert panel.clear_button.isEnabled() is False

    populated_figure = GnoviFigure(panels=[_make_3d_panel()])
    panel.set_figure(populated_figure)
    panel.refresh()
    assert panel.clear_button.isEnabled() is True


def test_clear_3d_plot_still_emits_the_clear_signal(qapp):
    from gnovi_plot.data.dataset_manager import DatasetManager

    figure = GnoviFigure(panels=[_make_3d_panel()])
    panel = Plot3DPanel(DatasetManager(), figure)
    received = []
    panel.clear_3d_plot_requested.connect(lambda: received.append(True))

    panel.clear_button.click()

    assert received == [True]


def test_grouped_3d_add_still_works_and_series_page_immediately_manages_it(qapp, monkeypatch):
    """No workflow available only through the removed list: a grouped Add
    still creates every Series3D correctly, and they're immediately
    selectable/manageable on the Series page -- the one place that now
    owns 3D series management."""
    from PySide6.QtWidgets import QMessageBox

    from gnovi_plot.data.dataset_manager import DatasetManager

    df = pd.DataFrame(
        {
            "x": [0.1, 0.1, 0.2, 0.2],
            "group": [25.0, 35.0, 25.0, 35.0],
            "z": [1.0, 1.5, 2.0, 2.5],
        }
    )
    dataset = Dataset(name="grouped", dataframe=df)
    manager = DatasetManager()
    manager.add(dataset)
    figure = GnoviFigure(panels=[_make_2d_panel()])
    plot3d = Plot3DPanel(manager, figure)
    plot3d.set_manager(manager)
    series_panel = PlotSeriesPanel(figure)

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    plot3d.dataset_combo.setCurrentIndex(plot3d.dataset_combo.findData(dataset.id))
    plot3d.x_combo.setCurrentText("x")
    plot3d.y_combo.setCurrentText("group")
    plot3d.z_combo.setCurrentText("z")
    plot3d.group_by_combo.setCurrentIndex(plot3d.group_by_combo.findData("group"))

    added = []
    plot3d.add_3d_series_requested.connect(lambda series_list: added.append(series_list))
    plot3d.add_button.click()

    assert len(added) == 1
    assert len(added[0]) == 2  # one Series3D per "group" value (25, 35)

    figure.panels[0] = Panel3D()
    for series in added[0]:
        figure.panels[0].add_series(series)
    series_panel.set_figure(figure)

    assert series_panel.series3d_list.count() == 2
    series_panel.series3d_list.setCurrentRow(0)
    assert series_panel.d3_label_edit.text() != ""
    series_panel.remove_3d_button.click()
    assert len(figure.active_panel.series) == 1


# --- WORKBENCH / FOCUS / EXTRACT: adaptive headings survive navigation ----------


def test_focus_3d_panel_keeps_3d_series_and_axes_headings(qapp):
    window = MainWindow()
    window.figure_model.panels[0] = _make_3d_panel()
    window._on_panel_switched()
    assert window.series_panel._stack.currentWidget() is window.series_panel._page_3d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_3d

    window._focus_panel(window.figure_model.active_panel)

    assert window.series_panel._stack.currentWidget() is window.series_panel._page_3d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_3d

    window._restore_multi_panel_view()

    assert window.series_panel._stack.currentWidget() is window.series_panel._page_3d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_3d
    window.close()


def test_focus_2d_panel_keeps_2d_series_and_axes_headings(qapp):
    window = MainWindow()
    assert window.series_panel._stack.currentWidget() is window.series_panel._page_2d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_2d

    window._focus_panel(window.figure_model.active_panel)

    assert window.series_panel._stack.currentWidget() is window.series_panel._page_2d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_2d
    window.close()


def test_extract_3d_panel_new_workbench_shows_3d_adaptive_content_immediately(qapp):
    window = MainWindow()
    window.figure_model.panels[0] = _make_3d_panel()
    window._on_panel_switched()

    window._on_extract_panel_requested()

    assert isinstance(window.figure_model.active_panel, Panel3D)
    assert window.series_panel._stack.currentWidget() is window.series_panel._page_3d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_3d
    window.close()


def test_extract_2d_panel_new_workbench_shows_2d_adaptive_content(qapp):
    window = MainWindow()

    window._on_extract_panel_requested()

    assert window.series_panel._stack.currentWidget() is window.series_panel._page_2d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_2d
    window.close()


def test_workbench_switching_between_a_2d_and_a_3d_workbench_updates_headings(qapp):
    window = MainWindow()
    original_workbench_id = window._current_workbench_id

    window._on_new_workbench_requested()
    window.figure_model.panels[0] = _make_3d_panel()
    window._on_panel_switched()
    assert window.series_panel._stack.currentWidget() is window.series_panel._page_3d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_3d

    window._on_workbench_tab_selected(original_workbench_id)

    assert window.series_panel._stack.currentWidget() is window.series_panel._page_2d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_2d
    window.close()


# --- THEME: section-header styling is defined and readable ---------------------


def test_stylesheet_defines_the_tool_strip_section_heading_selectors():
    qss = build_stylesheet()

    assert 'QLabel#ToolStripSectionLeft, QLabel#ToolStripSectionRight {' in qss
    section_rule = qss.split('QLabel#ToolStripSectionLeft, QLabel#ToolStripSectionRight {')[1].split("}")[0]
    assert "color:" in section_rule
    # Never a checked/selected-button look -- no background/border-left
    # accent stripe like `QToolButton#ToolStripButtonLeft:checked`.
    assert "background-color" not in section_rule


def test_stylesheet_defines_the_adaptive_page_heading_selector():
    qss = build_stylesheet()

    assert 'QLabel[pageHeading="true"] {' in qss


# --- PERSISTENCE / VERSION: navigation-only change, no format bump -------------


def test_project_format_version_is_unaffected_by_navigation_changes():
    from gnovi_plot.core.project_io import PROJECT_FORMAT_VERSION

    assert PROJECT_FORMAT_VERSION == 3
