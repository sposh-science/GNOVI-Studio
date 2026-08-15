import pandas as pd
import pytest
from matplotlib.figure import Figure

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.backends.matplotlib_backend import render_figure, render_panel
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d", x=(1.0, 2.0, 3.0, 4.0), y=(1.0, 4.0, 9.0, 16.0)):
    df = pd.DataFrame({"x": list(x), "y": list(y)})
    return Dataset(name=name, dataframe=df)


def _axes_for(figure: GnoviFigure):
    rows, cols = figure.layout
    mpl_figure = Figure()
    return mpl_figure, list(mpl_figure.subplots(rows, cols, squeeze=False).flat)


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_render_figure_draws_into_one_axes_per_panel(rows, cols):
    figure = GnoviFigure()
    figure.set_layout(rows, cols)
    for i, panel in enumerate(figure.panels):
        panel.add_series(PlotSeries.line(_make_dataset(f"d{i}"), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)

    assert len(axes_list) == rows * cols
    for ax in axes_list:
        assert len(ax.lines) == 1


def test_series_in_different_panels_do_not_leak_into_each_other():
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.set_active_panel(0)
    figure.add_series(PlotSeries.line(_make_dataset("a"), "x", "y"))
    # Panel 1 (index 1) gets nothing.

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)

    assert len(axes_list[0].lines) == 1
    assert len(axes_list[1].lines) == 0


def test_panel_labels_drawn_only_when_visible():
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    _mpl_figure, axes_list = _axes_for(figure)

    render_figure(axes_list, figure)
    assert all(len(ax.texts) == 0 for ax in axes_list)

    figure.panel_labels_visible = True
    render_figure(axes_list, figure)
    assert [ax.texts[0].get_text() for ax in axes_list] == ["(a)", "(b)"]


def test_log_scale_and_invert_axes_are_applied():
    figure = GnoviFigure()
    panel = figure.active_panel
    panel.xscale = "log"
    panel.yscale = "log"
    panel.xlim = (1.0, 10.0)
    panel.invert_x = True
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    ax = axes_list[0]

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_xlim() == (10.0, 1.0)  # inverted


def test_spines_visibility_and_linewidth():
    figure = GnoviFigure()
    panel = figure.active_panel
    panel.spine_top = False
    panel.spine_right = False
    panel.spine_linewidth = 2.5
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    ax = axes_list[0]

    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.spines["bottom"].get_visible() is True
    assert ax.spines["bottom"].get_linewidth() == pytest.approx(2.5)


def test_legend_columns_and_frame_are_applied():
    figure = GnoviFigure()
    panel = figure.active_panel
    panel.legend_ncol = 2
    panel.legend_frameon = False
    figure.add_series(PlotSeries.line(_make_dataset("a"), "x", "y"))
    figure.add_series(PlotSeries.line(_make_dataset("b"), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    ax = axes_list[0]

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_frame_on() is False


def test_normalize_to_max_does_not_mutate_source_dataframe():
    dataset = _make_dataset(y=(1.0, 4.0, 9.0, 16.0))
    original = dataset.dataframe.copy(deep=True)
    series = PlotSeries.line(dataset, "x", "y", normalize_to_max=True)

    figure = GnoviFigure()
    figure.add_series(series)
    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)

    pd.testing.assert_frame_equal(dataset.dataframe, original)
    line = axes_list[0].lines[0]
    assert max(line.get_ydata()) == pytest.approx(1.0)


def test_y_offset_shifts_drawn_values_without_mutating_source():
    dataset = _make_dataset(y=(1.0, 2.0, 3.0, 4.0))
    original = dataset.dataframe.copy(deep=True)
    series = PlotSeries.line(dataset, "x", "y", y_offset=100.0)

    figure = GnoviFigure()
    figure.add_series(series)
    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)

    pd.testing.assert_frame_equal(dataset.dataframe, original)
    line = axes_list[0].lines[0]
    assert list(line.get_ydata()) == [101.0, 102.0, 103.0, 104.0]


def test_histogram_percentage_mode_sums_to_one_hundred():
    df = pd.DataFrame({"v": [1, 1, 2, 2, 3, 3, 3, 4]})
    dataset = Dataset(name="hist", dataframe=df)
    series = PlotSeries.histogram(dataset, "v", hist_mode="percentage", bins=4)

    figure = GnoviFigure()
    figure.add_series(series)
    _mpl_figure, axes_list = _axes_for(figure)
    render_panel(axes_list[0], figure.active_panel, figure)

    heights = [patch.get_height() for patch in axes_list[0].patches]
    assert sum(heights) == pytest.approx(100.0)


def test_histogram_cumulative_mode_is_monotonically_nondecreasing():
    df = pd.DataFrame({"v": [1, 2, 2, 3, 3, 3, 4, 5]})
    dataset = Dataset(name="hist", dataframe=df)
    series = PlotSeries.histogram(dataset, "v", hist_mode="cumulative", bins=5)

    figure = GnoviFigure()
    figure.add_series(series)
    _mpl_figure, axes_list = _axes_for(figure)
    render_panel(axes_list[0], figure.active_panel, figure)

    heights = [patch.get_height() for patch in axes_list[0].patches]
    assert heights == sorted(heights)
    assert heights[-1] == pytest.approx(8.0)


def test_xrd_style_line_series_has_no_marker_by_default():
    dataset = _make_dataset()
    series = PlotSeries.line(dataset, "x", "y")
    assert series.marker == ""
