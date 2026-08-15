import pandas as pd
from PySide6.QtGui import QColor

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets import plot_series_panel as plot_series_panel_module
from gnovi_plot.gui.widgets.plot_series_panel import PlotSeriesPanel
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


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
