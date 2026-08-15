import pandas as pd
import pytest
from matplotlib.figure import Figure

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.backends.matplotlib_backend import apply_figure_layout, compute_tight_layout, render_figure
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_figure(rows=1, cols=1):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    dataset = Dataset(name="d", dataframe=df)
    figure = GnoviFigure()
    figure.set_layout(rows, cols)
    for panel in figure.panels:
        panel.title = "Panel Title"
        panel.xlabel = "X"
        panel.ylabel = "Y"
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    return figure


def test_default_margins_match_matplotlib_rc_defaults():
    figure = GnoviFigure()
    assert figure.margin_left == pytest.approx(0.125)
    assert figure.margin_right == pytest.approx(0.9)
    assert figure.margin_bottom == pytest.approx(0.11)
    assert figure.margin_top == pytest.approx(0.88)
    assert figure.panel_wspace == pytest.approx(0.2)
    assert figure.panel_hspace == pytest.approx(0.2)


def test_apply_figure_layout_uses_stored_margins_and_spacing():
    figure = _make_figure(2, 2)
    figure.margin_left = 0.2
    figure.margin_right = 0.8
    figure.margin_bottom = 0.15
    figure.margin_top = 0.85
    figure.panel_wspace = 0.4
    figure.panel_hspace = 0.5

    mpl_figure = Figure()
    axes_list = list(mpl_figure.subplots(2, 2, squeeze=False).flat)
    render_figure(axes_list, figure)
    apply_figure_layout(mpl_figure, figure)

    subplot_params = mpl_figure.subplotpars
    assert subplot_params.left == pytest.approx(0.2)
    assert subplot_params.right == pytest.approx(0.8)
    assert subplot_params.bottom == pytest.approx(0.15)
    assert subplot_params.top == pytest.approx(0.85)
    assert subplot_params.wspace == pytest.approx(0.4)
    assert subplot_params.hspace == pytest.approx(0.5)


def test_apply_figure_layout_reflects_the_stored_value_exactly():
    """Regression guard: the applied margin is exactly the stored value,
    never Matplotlib's automatically-recomputed "tight" layout -- the two
    would otherwise silently diverge (e.g. after a title/label change)
    unless margins are baked in explicitly (see `compute_tight_layout`)."""
    figure = _make_figure()
    figure.margin_left = 0.3

    mpl_figure = Figure()
    axes_list = list(mpl_figure.subplots(1, 1, squeeze=False).flat)
    render_figure(axes_list, figure)
    apply_figure_layout(mpl_figure, figure)

    assert mpl_figure.subplotpars.left == pytest.approx(0.3)


def test_apply_figure_layout_scales_margins_within_a_letterbox_rect():
    figure = GnoviFigure()
    mpl_figure = Figure()
    axes_list = list(mpl_figure.subplots(1, 1, squeeze=False).flat)
    render_figure(axes_list, figure)

    apply_figure_layout(mpl_figure, figure, rect=(0.1, 0.1, 0.9, 0.9))

    subplot_params = mpl_figure.subplotpars
    width_frac = 0.8
    height_frac = 0.8
    assert subplot_params.left == pytest.approx(0.1 + figure.margin_left * width_frac)
    assert subplot_params.right == pytest.approx(0.1 + figure.margin_right * width_frac)
    assert subplot_params.bottom == pytest.approx(0.1 + figure.margin_bottom * height_frac)
    assert subplot_params.top == pytest.approx(0.1 + figure.margin_top * height_frac)


def test_compute_tight_layout_returns_all_six_fields_without_mutating_figure():
    figure = _make_figure()
    original = (figure.margin_left, figure.margin_right, figure.margin_bottom, figure.margin_top)

    result = compute_tight_layout(figure)

    assert set(result) == {
        "margin_left",
        "margin_right",
        "margin_bottom",
        "margin_top",
        "panel_wspace",
        "panel_hspace",
    }
    assert (figure.margin_left, figure.margin_right, figure.margin_bottom, figure.margin_top) == original


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 2)])
def test_apply_figure_layout_works_for_every_supported_panel_grid(rows, cols):
    figure = _make_figure(rows, cols)
    mpl_figure = Figure()
    axes_list = list(mpl_figure.subplots(rows, cols, squeeze=False).flat)
    render_figure(axes_list, figure)

    apply_figure_layout(mpl_figure, figure)  # must not raise

    assert mpl_figure.subplotpars.left == pytest.approx(figure.margin_left)
    assert mpl_figure.subplotpars.wspace == pytest.approx(figure.panel_wspace)
    assert mpl_figure.subplotpars.hspace == pytest.approx(figure.panel_hspace)


def test_layout_margins_do_not_change_axis_limits_or_series_data():
    figure = _make_figure()
    panel = figure.active_panel
    panel.xlim = (0.0, 5.0)
    panel.ylim = (0.0, 20.0)
    series_before = [(s.x_column, s.y_column) for s in panel.series]

    figure.margin_left = 0.3
    figure.margin_right = 0.7
    figure.panel_wspace = 1.0
    figure.panel_hspace = 1.0

    mpl_figure = Figure()
    axes_list = list(mpl_figure.subplots(1, 1, squeeze=False).flat)
    render_figure(axes_list, figure)
    apply_figure_layout(mpl_figure, figure)

    assert panel.xlim == (0.0, 5.0)
    assert panel.ylim == (0.0, 20.0)
    assert [(s.x_column, s.y_column) for s in panel.series] == series_before
    assert axes_list[0].get_xlim() == (0.0, 5.0)
    assert axes_list[0].get_ylim() == (0.0, 20.0)


def test_preview_and_export_paths_produce_identical_subplot_params(qapp):
    """PlotCanvas (on-screen preview) and export.figure_export both call
    `apply_figure_layout` with the figure's stored margins -- this checks
    they actually agree for the same GnoviFigure, not just that each calls
    *some* layout function."""
    from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas

    figure = _make_figure(2, 2)
    figure.margin_left = 0.22
    figure.margin_right = 0.77
    figure.margin_bottom = 0.18
    figure.margin_top = 0.82
    figure.panel_wspace = 0.5
    figure.panel_hspace = 0.6
    figure.lock_aspect_ratio = False  # full-bleed rect, matching export

    canvas = PlotCanvas()
    canvas.resize(400, 300)
    canvas.render(figure)
    preview_params = canvas.figure.subplotpars

    export_mpl_figure = Figure(figsize=(figure.figure_width_in, figure.figure_height_in))
    export_axes = list(export_mpl_figure.subplots(2, 2, squeeze=False).flat)
    render_figure(export_axes, figure)
    apply_figure_layout(export_mpl_figure, figure)
    export_params = export_mpl_figure.subplotpars

    assert preview_params.left == pytest.approx(export_params.left)
    assert preview_params.right == pytest.approx(export_params.right)
    assert preview_params.bottom == pytest.approx(export_params.bottom)
    assert preview_params.top == pytest.approx(export_params.top)
    assert preview_params.wspace == pytest.approx(export_params.wspace)
    assert preview_params.hspace == pytest.approx(export_params.hspace)
