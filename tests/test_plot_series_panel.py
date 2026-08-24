import pandas as pd
from PySide6.QtGui import QColor

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets import plot_series_panel as plot_series_panel_module
from gnovi_plot.gui.widgets.plot_series_panel import PlotSeriesPanel
from gnovi_plot.plotting.figure import GnoviFigure, Panel3D
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.series3d import Plot3DType, Series3D


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


def _make_3d_dataset(name="d3"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0]})
    return Dataset(name=name, dataframe=df)


def _make_3d_figure():
    panel3d = Panel3D()
    panel3d.add_series(Series3D(dataset=_make_3d_dataset(), x_column="x", y_column="y", z_column="z"))
    return GnoviFigure(panels=[panel3d])


# --- Theme-aware contrast warning (manual colors only) -----------------------


def test_update_contrast_warnings_flags_a_low_contrast_manual_color(qapp):
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y", color="#fafafa")
    series.color_is_manual = True
    figure.add_series(series)
    panel = PlotSeriesPanel(figure)
    panel.show()

    panel.update_contrast_warnings(dark_mode=False)

    assert panel.contrast_warning_label.isVisible() is True
    assert panel.optimize_colors_button.isVisible() is True
    assert "1 series has low contrast" in panel.contrast_warning_label.text()


def test_update_contrast_warnings_ignores_automatic_colors(qapp):
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y", color="#fafafa")
    # color_is_manual left False -- auto-assigned colors are never flagged,
    # they're already picked from a theme-appropriate cycle.
    figure.add_series(series)
    panel = PlotSeriesPanel(figure)
    panel.show()

    panel.update_contrast_warnings(dark_mode=False)

    assert panel.contrast_warning_label.isVisible() is False


def test_update_contrast_warnings_hides_banner_for_a_readable_manual_color(qapp):
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y", color="#1f77b4")
    series.color_is_manual = True
    figure.add_series(series)
    panel = PlotSeriesPanel(figure)
    panel.show()

    panel.update_contrast_warnings(dark_mode=False)

    assert panel.contrast_warning_label.isVisible() is False
    assert panel.optimize_colors_button.isVisible() is False


def test_optimize_colors_reassigns_flagged_series_and_clears_the_manual_flag(qapp):
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y", color="#fafafa")
    series.color_is_manual = True
    figure.add_series(series)
    panel = PlotSeriesPanel(figure)
    panel.show()
    panel.update_contrast_warnings(dark_mode=False)

    panel.optimize_colors_button.click()

    assert series.color != "#fafafa"
    assert series.color_is_manual is False
    assert panel.contrast_warning_label.isVisible() is False


def test_picking_a_color_marks_the_series_as_manual_and_never_silently_changes_again(qapp, monkeypatch):
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y")
    figure.add_series(series)
    panel = PlotSeriesPanel(figure)

    monkeypatch.setattr(
        plot_series_panel_module.QColorDialog, "getColor", lambda *args, **kwargs: QColor("#123456")
    )
    panel._pick_color()

    assert series.color == "#123456"
    assert series.color_is_manual is True


# --- Adaptive 3D page (Panel3D / Series3D) ------------------------------------


def test_a_panel3d_active_panel_shows_the_3d_stack_page(qapp):
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)

    assert panel._stack.currentWidget() is panel._page_3d


def test_a_2d_panel_active_panel_shows_the_2d_stack_page(qapp):
    figure = GnoviFigure()
    panel = PlotSeriesPanel(figure)

    assert panel._stack.currentWidget() is panel._page_2d


def test_3d_series_list_is_populated_from_the_active_panel3d(qapp):
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)

    assert panel.series3d_list.count() == 1


def test_3d_series_editors_edit_the_selected_series3d(qapp):
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)
    panel.series3d_list.setCurrentRow(0)

    panel.d3_label_edit.setText("mat scatter")
    panel._apply_3d_label()
    panel.d3_marker_size_spin.setValue(11.0)
    panel._apply_3d_marker_size(11.0)
    panel.d3_alpha_spin.setValue(0.4)
    panel._apply_3d_alpha(0.4)
    panel.d3_visible_check.setChecked(False)
    panel._apply_3d_visible(False)

    series = figure.active_panel.series[0]
    assert series.label == "mat scatter"
    assert series.marker_size == 11.0
    assert series.alpha == 0.4
    assert series.visible is False


def test_3d_remove_series_removes_only_from_the_panel3d(qapp):
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)
    panel.series3d_list.setCurrentRow(0)

    panel.remove_3d_button.click()

    assert figure.active_panel.series == []


def test_3d_clear_all_removes_every_series3d(qapp):
    figure = _make_3d_figure()
    figure.active_panel.add_series(
        Series3D(dataset=_make_3d_dataset(), x_column="x", y_column="y", z_column="z")
    )
    panel = PlotSeriesPanel(figure)
    assert len(figure.active_panel.series) == 2

    panel.clear_3d_button.click()

    assert figure.active_panel.series == []


def test_3d_marker_options_exclude_none(qapp):
    """A 3D scatter point with no marker at all would render nothing --
    unlike a 2D line plot, where marker="" just means "no marker dots on
    the line" (the line itself still visible) -- so "None" is intentionally
    absent from the 3D marker combo (see `_MARKER_OPTIONS_3D`)."""
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)

    codes = [panel.d3_marker_combo.itemData(i) for i in range(panel.d3_marker_combo.count())]

    assert "" not in codes


def test_switching_from_2d_to_3d_panel_swaps_the_stack_page_on_refresh(qapp):
    figure = GnoviFigure()
    panel = PlotSeriesPanel(figure)
    assert panel._stack.currentWidget() is panel._page_2d

    figure.panels.append(_make_3d_figure().panels[0])
    figure.set_active_panel(1)
    panel.refresh()

    assert panel._stack.currentWidget() is panel._page_3d


# --- 3D plot type: editable, conditional marker/line-style-width visibility -----------


def test_3d_plot_type_defaults_to_scatter_in_the_combo(qapp):
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)
    panel.series3d_list.setCurrentRow(0)

    assert panel.d3_plot_type_combo.currentData() == Plot3DType.SCATTER


def test_scatter_shows_marker_controls_and_hides_line_controls(qapp):
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)
    panel.show()
    panel.series3d_list.setCurrentRow(0)

    assert panel.d3_marker_combo.isVisible() is True
    assert panel.d3_marker_size_spin.isVisible() is True
    assert panel.d3_line_style_combo.isVisible() is False
    assert panel.d3_line_width_spin.isVisible() is False
    panel.close()


def test_line_hides_marker_controls_and_shows_line_controls(qapp):
    figure = _make_3d_figure()
    figure.active_panel.series[0].plot_type = Plot3DType.LINE
    panel = PlotSeriesPanel(figure)
    panel.show()
    panel.series3d_list.setCurrentRow(0)

    assert panel.d3_marker_combo.isVisible() is False
    assert panel.d3_marker_size_spin.isVisible() is False
    assert panel.d3_line_style_combo.isVisible() is True
    assert panel.d3_line_width_spin.isVisible() is True
    panel.close()


def test_line_marker_shows_both_marker_and_line_controls(qapp):
    figure = _make_3d_figure()
    figure.active_panel.series[0].plot_type = Plot3DType.LINE_MARKER
    panel = PlotSeriesPanel(figure)
    panel.show()
    panel.series3d_list.setCurrentRow(0)

    assert panel.d3_marker_combo.isVisible() is True
    assert panel.d3_line_style_combo.isVisible() is True
    panel.close()


def test_changing_plot_type_via_the_combo_updates_the_series_and_visibility(qapp):
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)
    panel.show()
    panel.series3d_list.setCurrentRow(0)

    panel.d3_plot_type_combo.setCurrentIndex(panel.d3_plot_type_combo.findData(Plot3DType.LINE))

    assert figure.active_panel.series[0].plot_type == Plot3DType.LINE
    assert panel.d3_line_style_combo.isVisible() is True
    assert panel.d3_marker_combo.isVisible() is False
    panel.close()


def test_changing_plot_type_via_the_combo_stores_a_genuine_enum_member(qapp):
    """Regression test: `QComboBox.currentData()` can hand back a plain
    `str` that merely `==`-compares equal to the right `Plot3DType` member
    (a Qt/PySide `str`-subclassed-Enum marshalling quirk) -- must be
    normalized back to a real enum member or `Series3D.to_dict()` crashes
    on `.value` at save time. Caught via manual GUI validation."""
    figure = _make_3d_figure()
    panel = PlotSeriesPanel(figure)
    panel.series3d_list.setCurrentRow(0)

    panel.d3_plot_type_combo.setCurrentIndex(panel.d3_plot_type_combo.findData(Plot3DType.LINE))

    series = figure.active_panel.series[0]
    assert isinstance(series.plot_type, Plot3DType)
    series.to_dict()  # must not raise AttributeError
    panel.close()


def test_editing_line_style_and_width_updates_the_series(qapp):
    figure = _make_3d_figure()
    figure.active_panel.series[0].plot_type = Plot3DType.LINE
    panel = PlotSeriesPanel(figure)
    panel.series3d_list.setCurrentRow(0)

    panel.d3_line_style_combo.setCurrentIndex(panel.d3_line_style_combo.findData("--"))
    panel.d3_line_width_spin.setValue(3.0)

    assert figure.active_panel.series[0].line_style == "--"
    assert figure.active_panel.series[0].line_width == 3.0
    panel.close()
