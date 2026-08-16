"""Preview-only legend fitting (`matplotlib_backend.fit_panel_legends_to_axes`)
-- backend-level, Qt-free (uses `FigureCanvasAgg` directly, the same rendering
stack `gui.widgets.plot_canvas.PlotCanvas` sits on top of). See
`tests/test_figure_aspect_ratio_gui.py`-style MainWindow-driven smoke tests
in `test_gui_responsiveness.py` for the end-to-end (real Qt canvas) coverage.
"""

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.backends.matplotlib_backend import (
    fit_panel_legends_to_axes,
    render_figure,
)
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries

_LONG_LABELS = [
    "Extremely long legend label describing series number one in detail",
    "Extremely long legend label describing series number two in detail",
    "Extremely long legend label describing series number three in detail",
]


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


def _canvas_and_axes(figure: GnoviFigure, figsize=(6.4, 4.8)):
    rows, cols = figure.layout
    mpl_figure = Figure(figsize=figsize)
    FigureCanvasAgg(mpl_figure)  # attaches itself as mpl_figure.canvas
    axes_list = list(mpl_figure.subplots(rows, cols, squeeze=False).flat)
    return mpl_figure, axes_list


def _crowded_figure(*, layout=(1, 1), legend_loc="best", legend_fontsize=9.0) -> GnoviFigure:
    figure = GnoviFigure()
    figure.legend_font_size = legend_fontsize
    figure.set_layout(*layout)
    dataset = _make_dataset()
    for panel in figure.panels:
        for label in _LONG_LABELS:
            panel.add_series(PlotSeries.line(dataset, "x", "y", label=label))
        panel.legend_visible = True
        panel.legend_loc = legend_loc
    return figure


def _fontsize(ax) -> float:
    return ax.get_legend().get_texts()[0].get_fontsize()


def _ax_bbox(ax, renderer):
    return ax.get_window_extent(renderer=renderer)


def _legend_bbox(ax, renderer):
    return ax.get_legend().get_window_extent(renderer=renderer)


def _contained(ax, renderer, tolerance=2.0) -> bool:
    ax_bbox = _ax_bbox(ax, renderer)
    leg_bbox = _legend_bbox(ax, renderer)
    return (
        leg_bbox.x0 >= ax_bbox.x0 - tolerance
        and leg_bbox.x1 <= ax_bbox.x1 + tolerance
        and leg_bbox.y0 >= ax_bbox.y0 - tolerance
        and leg_bbox.y1 <= ax_bbox.y1 + tolerance
    )


# --- Configured value is never mutated ------------------------------------------


def test_fit_pass_never_mutates_the_configured_panel_legend_fontsize():
    figure = _crowded_figure(layout=(2, 2))
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(3.0, 2.4))  # tiny -- forces a shrink

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)

    assert figure.legend_font_size == 9.0
    assert all(panel.legend_fontsize is None for panel in figure.panels)


def test_fit_pass_never_mutates_an_explicit_panel_level_fontsize():
    figure = _crowded_figure(layout=(2, 2))
    figure.panels[0].legend_fontsize = 11.0
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(3.0, 2.4))

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)

    assert figure.panels[0].legend_fontsize == 11.0


# --- Small axes shrink, large axes don't ----------------------------------------


def test_a_small_crowded_panel_uses_a_smaller_effective_fontsize():
    figure = _crowded_figure(layout=(2, 2), legend_fontsize=12.0)
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(3.0, 2.4))

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)

    for ax in axes_list:
        assert _fontsize(ax) < 12.0
        assert _fontsize(ax) >= 6.0  # never below the readability floor


def test_a_large_uncrowded_panel_keeps_the_configured_fontsize():
    figure = _crowded_figure(layout=(1, 1), legend_fontsize=9.0)
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(12.0, 9.0))

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)

    assert _fontsize(axes_list[0]) == pytest.approx(9.0)


def test_fit_pass_is_a_no_op_when_nothing_overflows():
    """A legend that already fits at the configured size must be left
    completely alone -- same artist, not recreated."""
    figure = _crowded_figure(layout=(1, 1), legend_fontsize=9.0)
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(12.0, 9.0))
    render_figure(axes_list, figure)
    original_legend = axes_list[0].get_legend()

    fit_panel_legends_to_axes(axes_list, figure)

    assert axes_list[0].get_legend() is original_legend


# --- Containment -----------------------------------------------------------------


def test_fit_pass_contains_the_legend_within_its_own_axes_when_reasonably_possible():
    # A moderately (not extremely) tight panel: 3 long-ish labels should fit
    # once shrunk, so containment is actually achievable here.
    figure = GnoviFigure()
    dataset = _make_dataset()
    figure.add_series(PlotSeries.line(dataset, "x", "y", label="Series alpha"))
    figure.add_series(PlotSeries.line(dataset, "x", "y", label="Series beta"))
    figure.active_panel.legend_visible = True
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(2.2, 1.8))

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)
    canvas = mpl_figure.canvas
    canvas.draw()
    renderer = canvas.get_renderer()

    assert _contained(axes_list[0], renderer)


def test_fit_pass_leaves_an_ungraceful_case_at_the_floor_size_without_raising():
    """Even when containment genuinely can't be achieved (too many long
    labels for a tiny panel), the fit pass must complete gracefully at the
    floor size rather than raise or loop forever."""
    figure = _crowded_figure(layout=(2, 2), legend_fontsize=9.0)
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(1.4, 1.1))  # very tiny

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)  # must not raise

    for ax in axes_list:
        assert _fontsize(ax) == pytest.approx(6.0)


# --- Growing back restores the configured size -----------------------------------


def test_shrinking_then_growing_the_panel_restores_the_configured_fontsize():
    figure = _crowded_figure(layout=(1, 1), legend_fontsize=9.0)
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(2.0, 1.6))
    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)
    assert _fontsize(axes_list[0]) < 9.0  # confirm it actually shrank

    mpl_figure.set_size_inches(12.0, 9.0, forward=True)
    fit_panel_legends_to_axes(axes_list, figure)

    assert _fontsize(axes_list[0]) == pytest.approx(9.0)
    assert figure.legend_font_size == 9.0  # model was never touched throughout


# --- Outside placements are never pulled back inside ------------------------------


@pytest.mark.parametrize("outside_loc", ["outside right", "outside bottom"])
def test_outside_legend_placements_are_never_shrunk_or_repositioned(outside_loc):
    figure = _crowded_figure(layout=(2, 2), legend_loc=outside_loc, legend_fontsize=9.0)
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(3.0, 2.4))

    render_figure(axes_list, figure)
    original_legends = [ax.get_legend() for ax in axes_list]
    fit_panel_legends_to_axes(axes_list, figure)

    for ax, original in zip(axes_list, original_legends):
        assert ax.get_legend() is original  # completely untouched
        assert _fontsize(ax) == pytest.approx(9.0)


# --- Multiple layouts --------------------------------------------------------------


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_fit_pass_runs_cleanly_across_supported_layouts(layout):
    figure = _crowded_figure(layout=layout, legend_fontsize=9.0)
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(4.0, 3.0))

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)  # must not raise for any layout

    for ax in axes_list:
        assert _fontsize(ax) >= 6.0


# --- No legend / legend hidden are safely skipped ----------------------------------


def test_fit_pass_is_a_no_op_when_legend_is_hidden():
    figure = GnoviFigure()
    dataset = _make_dataset()
    figure.add_series(PlotSeries.line(dataset, "x", "y", label="a"))
    figure.active_panel.legend_visible = False
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(2.0, 1.6))

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)  # must not raise

    assert axes_list[0].get_legend() is None


def test_fit_pass_is_a_no_op_when_there_are_no_series():
    figure = GnoviFigure()
    figure.active_panel.legend_visible = True
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(2.0, 1.6))

    render_figure(axes_list, figure)
    fit_panel_legends_to_axes(axes_list, figure)  # must not raise

    assert axes_list[0].get_legend() is None
