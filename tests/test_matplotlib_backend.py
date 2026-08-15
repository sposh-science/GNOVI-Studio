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


# --- dark_mode theming -------------------------------------------------------


def test_dark_mode_recolors_figure_and_axes_background():
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    mpl_figure, axes_list = _axes_for(figure)

    render_figure(axes_list, figure, dark_mode=True)

    assert mpl_figure.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
    assert axes_list[0].get_facecolor() != (1.0, 1.0, 1.0, 1.0)


def test_light_mode_is_the_default_and_uses_a_white_background():
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    mpl_figure, axes_list = _axes_for(figure)

    render_figure(axes_list, figure)

    assert mpl_figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert axes_list[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_dark_mode_is_fully_reversible_on_the_same_axes():
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    mpl_figure, axes_list = _axes_for(figure)

    render_figure(axes_list, figure, dark_mode=True)
    render_figure(axes_list, figure, dark_mode=False)

    assert mpl_figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert axes_list[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_dark_mode_does_not_change_series_color():
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y")
    figure.add_series(series)
    original_color = series.color

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure, dark_mode=True)

    assert series.color == original_color
    assert axes_list[0].lines[0].get_color() == original_color


# --- Regression: log scale + scientific notation must not crash --------------
#
# `ax.ticklabel_format(style="scientific")` only works with Matplotlib's
# ScalarFormatter; a log-scale axis uses LogFormatterSciNotation instead and
# raises AttributeError if asked for scientific-notation styling. Toggling
# Log scale and Scientific notation together in the Axes & Ticks panel is an
# ordinary, unremarkable user action, and previously reached this uncaught,
# surfacing as the app's generic "Unexpected Error" dialog.


def test_log_scale_with_scientific_notation_x_does_not_raise():
    figure = GnoviFigure()
    panel = figure.active_panel
    panel.xscale = "log"
    panel.scientific_notation_x = True
    figure.add_series(PlotSeries.line(_make_dataset(x=(1.0, 2.0, 3.0, 4.0)), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)  # must not raise

    assert axes_list[0].get_xscale() == "log"


def test_log_scale_with_scientific_notation_y_does_not_raise():
    figure = GnoviFigure()
    panel = figure.active_panel
    panel.yscale = "log"
    panel.scientific_notation_y = True
    figure.add_series(PlotSeries.line(_make_dataset(y=(1.0, 2.0, 3.0, 4.0)), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)  # must not raise

    assert axes_list[0].get_yscale() == "log"


def test_linear_scale_with_scientific_notation_still_applies():
    """The fix must not disable scientific notation for the (default)
    linear-scale case it actually works for."""
    figure = GnoviFigure()
    panel = figure.active_panel
    panel.scientific_notation_x = True
    figure.add_series(PlotSeries.line(_make_dataset(x=(100000.0, 200000.0, 300000.0, 400000.0)), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    ax = axes_list[0]

    assert ax.xaxis.get_major_formatter()._scientific is True


# --- Universal Grid Appearance (figure-wide) ----------------------------------


def test_grid_appearance_style_width_and_alpha_are_applied():
    figure = GnoviFigure()
    figure.active_panel.grid = True
    figure.grid_linestyle = ":"
    figure.grid_linewidth = 2.5
    figure.grid_alpha = 0.3
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    ax = axes_list[0]

    gridlines = ax.xaxis.get_gridlines()
    assert gridlines
    assert gridlines[0].get_linestyle() == ":"
    assert gridlines[0].get_linewidth() == pytest.approx(2.5)
    assert gridlines[0].get_alpha() == pytest.approx(0.3)


def test_grid_appearance_is_uniform_across_panels():
    """Grid appearance is figure-wide -- unlike `Panel.grid`/`grid_which`
    (on/off, per panel), every panel that has its grid on renders it with
    the same style/width/opacity/color."""
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.grid_linewidth = 3.0
    for panel in figure.panels:
        panel.grid = True

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)

    widths = [ax.xaxis.get_gridlines()[0].get_linewidth() for ax in axes_list]
    assert widths == [pytest.approx(3.0), pytest.approx(3.0)]


def test_custom_grid_color_overrides_the_theme_default():
    figure = GnoviFigure()
    figure.active_panel.grid = True
    figure.grid_color = "#ff00ff"
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure, dark_mode=True)
    ax = axes_list[0]

    from matplotlib.colors import to_hex

    assert to_hex(ax.xaxis.get_gridlines()[0].get_color()) == "#ff00ff"


def test_grid_color_none_falls_back_to_the_theme_default():
    figure = GnoviFigure()
    figure.active_panel.grid = True
    assert figure.grid_color is None
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure, dark_mode=False)
    render_figure(axes_list, figure, dark_mode=True)  # must not raise either theme


def test_dark_mode_with_grid_disabled_does_not_warn(recwarn):
    figure = GnoviFigure()
    figure.active_panel.grid = False
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    _mpl_figure, axes_list = _axes_for(figure)

    render_figure(axes_list, figure, dark_mode=True)

    assert not any("grid" in str(w.message).lower() for w in recwarn.list)


# --- Tick length/width (major/minor) ------------------------------------------


def test_major_and_minor_tick_length_and_width_are_applied():
    figure = GnoviFigure()
    figure.active_panel.minor_ticks = True
    figure.active_panel.major_tick_length = 7.0
    figure.active_panel.major_tick_width = 2.0
    figure.active_panel.minor_tick_length = 3.0
    figure.active_panel.minor_tick_width = 0.4
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    ax = axes_list[0]

    major_tick = ax.xaxis.get_major_ticks()[0]
    minor_tick = ax.xaxis.get_minor_ticks()[0]
    assert major_tick.tick1line.get_markersize() == pytest.approx(7.0)
    assert major_tick.tick1line.get_markeredgewidth() == pytest.approx(2.0)
    assert minor_tick.tick1line.get_markersize() == pytest.approx(3.0)
    assert minor_tick.tick1line.get_markeredgewidth() == pytest.approx(0.4)


# --- Legend: Outside Right / Outside Bottom -----------------------------------


def test_outside_right_legend_uses_a_bbox_to_anchor_outside_the_axes():
    figure = GnoviFigure()
    figure.active_panel.legend_loc = "outside right"
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    legend = axes_list[0].get_legend()

    assert legend is not None
    assert legend._loc == 6  # "center left" -- the anchor corner for "outside right"


def test_outside_bottom_legend_uses_a_bbox_to_anchor_outside_the_axes():
    figure = GnoviFigure()
    figure.active_panel.legend_loc = "outside bottom"
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    _mpl_figure, axes_list = _axes_for(figure)
    render_figure(axes_list, figure)
    legend = axes_list[0].get_legend()

    assert legend is not None
    assert legend._loc == 9  # "upper center" -- the anchor corner for "outside bottom"


# --- Theme-aware contrast checking (manual series colors only) ---------------


def test_contrast_ratio_of_a_color_against_itself_is_one():
    from gnovi_plot.plotting.backends.matplotlib_backend import contrast_ratio

    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0)


def test_contrast_ratio_black_vs_white_is_maximal():
    from gnovi_plot.plotting.backends.matplotlib_backend import contrast_ratio

    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, rel=1e-3)


def test_is_low_contrast_flags_a_near_background_color_on_light_theme():
    from gnovi_plot.plotting.backends.matplotlib_backend import is_low_contrast

    # Near-white on the light theme's white axes background.
    assert is_low_contrast("#fafafa", dark_mode=False) is True


def test_is_low_contrast_flags_a_near_background_color_on_dark_theme():
    from gnovi_plot.plotting.backends.matplotlib_backend import is_low_contrast

    # Near the dark theme's dark axes background.
    assert is_low_contrast("#26272e", dark_mode=True) is True


def test_is_low_contrast_is_false_for_a_clearly_readable_color():
    from gnovi_plot.plotting.backends.matplotlib_backend import is_low_contrast

    assert is_low_contrast("#1f77b4", dark_mode=False) is False
    assert is_low_contrast("#1f77b4", dark_mode=True) is False
