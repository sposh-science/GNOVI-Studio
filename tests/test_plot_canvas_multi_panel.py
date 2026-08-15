import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


def test_canvas_axes_alias_matches_single_panel_layout(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes_list) == 1
    assert canvas.axes is canvas.axes_list[0]
    assert len(canvas.axes.lines) == 1


def test_canvas_rebuilds_axes_grid_on_layout_change(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    assert len(canvas.axes_list) == 1

    figure.set_layout(2, 2)
    canvas.render(figure)

    assert len(canvas.axes_list) == 4


def test_series_added_to_different_panels_render_independently(qapp):
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.set_active_panel(0)
    figure.add_series(PlotSeries.line(_make_dataset("a"), "x", "y"))
    figure.set_active_panel(1)
    figure.add_series(PlotSeries.line(_make_dataset("b"), "x", "y"))
    figure.add_series(PlotSeries.line(_make_dataset("c"), "x", "y"))

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes_list[0].lines) == 1
    assert len(canvas.axes_list[1].lines) == 2


def test_active_axes_tracks_the_active_panel_index(qapp):
    figure = GnoviFigure()
    figure.set_layout(2, 1)
    canvas = PlotCanvas()
    canvas.render(figure)

    assert canvas.active_axes(figure) is canvas.axes_list[0]
    figure.set_active_panel(1)
    assert canvas.active_axes(figure) is canvas.axes_list[1]


def test_shrinking_layout_after_growing_rebuilds_axes_again(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)

    figure.set_layout(1, 2)
    canvas.render(figure)
    assert len(canvas.axes_list) == 2

    figure.set_layout(1, 1)
    canvas.render(figure)
    assert len(canvas.axes_list) == 1
