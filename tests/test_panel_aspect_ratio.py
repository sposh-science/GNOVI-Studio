"""Panel Aspect Ratio (`GnoviFigure.panel_aspect_preset`) -- backend-level,
Qt-free. Confirms it's genuinely independent of Figure Aspect Ratio, never
touches numeric data limits/data-unit scaling (`Axes.set_box_aspect`, never
`set_aspect("equal")`), applies uniformly across every panel, and that
preview and export render identically (both go through `render_panel`/
`render_figure` -- see `plotting.backends.matplotlib_backend`).
"""

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.backends.matplotlib_backend import (
    apply_figure_layout,
    render_figure,
    render_panel,
)
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries
from gnovi_plot.plotting.units import PANEL_ASPECT_RATIO_PRESETS, panel_box_aspect


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [-0.2, 0.0, 0.6], "y": [-5e-5, 0.0, 5e-5]})
    return Dataset(name=name, dataframe=df)


def _canvas_and_axes(figure: GnoviFigure, figsize=(8.0, 6.0)):
    rows, cols = figure.layout
    mpl_figure = Figure(figsize=figsize)
    FigureCanvasAgg(mpl_figure)
    axes_list = list(mpl_figure.subplots(rows, cols, squeeze=False).flat)
    return mpl_figure, axes_list


def _figure_with_series(*, layout=(1, 1), panel_aspect="Auto") -> GnoviFigure:
    figure = GnoviFigure()
    figure.panel_aspect_preset = panel_aspect
    figure.set_layout(*layout)
    dataset = _make_dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    return figure


def _box_ratio(ax, canvas_w_in, canvas_h_in) -> float:
    pos = ax.get_position()
    return (pos.width * canvas_w_in) / (pos.height * canvas_h_in)


# --- Model representation ---------------------------------------------------------


def test_default_panel_aspect_preset_is_auto():
    assert GnoviFigure().panel_aspect_preset == "Auto"


def test_panel_aspect_preset_is_a_plain_constructor_field():
    figure = GnoviFigure(panel_aspect_preset="1:1")
    assert figure.panel_aspect_preset == "1:1"


def test_panel_aspect_ratio_presets_match_figure_aspect_ratio_values():
    """Same width/height ratio values as the Figure Aspect Ratio presets
    (deliberately -- "1:1" must mean the same physical shape wherever it's
    picked), minus the figure-only "Auto / Fit workspace"/"Custom" entries."""
    from gnovi_plot.plotting.units import ASPECT_RATIO_PRESETS

    assert set(PANEL_ASPECT_RATIO_PRESETS) == {"Auto", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9"}
    for name in ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9"):
        assert PANEL_ASPECT_RATIO_PRESETS[name] == ASPECT_RATIO_PRESETS[name]


@pytest.mark.parametrize(
    "preset,expected_box_aspect",
    [("Auto", None), ("1:1", 1.0), ("4:3", 0.75), ("3:4", 4 / 3), ("16:9", 9 / 16)],
)
def test_panel_box_aspect_is_the_reciprocal_of_the_width_height_ratio(preset, expected_box_aspect):
    result = panel_box_aspect(preset)
    if expected_box_aspect is None:
        assert result is None
    else:
        assert result == pytest.approx(expected_box_aspect)


# --- Serialization -----------------------------------------------------------------


def test_panel_aspect_preset_round_trips_through_to_dict_from_dict():
    figure = GnoviFigure()
    figure.panel_aspect_preset = "1:1"

    restored = GnoviFigure.from_dict(figure.to_dict(), {})

    assert restored.panel_aspect_preset == "1:1"


def test_missing_panel_aspect_preset_in_saved_data_defaults_to_auto():
    """Loading an older project (saved before this feature existed) must
    not crash and must fall back to "Auto"."""
    figure = GnoviFigure()
    data = figure.to_dict()
    del data["panel_aspect_preset"]

    restored = GnoviFigure.from_dict(data, {})

    assert restored.panel_aspect_preset == "Auto"


def test_figure_and_panel_aspect_are_independent_fields():
    figure = GnoviFigure()
    figure.aspect_preset = "16:9"
    figure.lock_aspect_ratio = True
    figure.panel_aspect_preset = "1:1"

    restored = GnoviFigure.from_dict(figure.to_dict(), {})

    assert restored.aspect_preset == "16:9"
    assert restored.lock_aspect_ratio is True
    assert restored.panel_aspect_preset == "1:1"


# --- Rendering: box_aspect applied, data untouched ----------------------------------


def test_auto_preset_leaves_box_aspect_unset_matching_prior_behavior():
    figure = _figure_with_series(panel_aspect="Auto")
    _mpl_figure, axes_list = _canvas_and_axes(figure)

    render_figure(axes_list, figure)

    assert axes_list[0].get_box_aspect() is None


@pytest.mark.parametrize("preset", ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9"])
def test_each_preset_sets_the_expected_box_aspect(preset):
    figure = _figure_with_series(panel_aspect=preset)
    _mpl_figure, axes_list = _canvas_and_axes(figure)

    render_figure(axes_list, figure)

    assert axes_list[0].get_box_aspect() == pytest.approx(panel_box_aspect(preset))


def test_panel_aspect_never_uses_data_aspect_equal():
    """The graphical-box-only requirement: `Axes.get_aspect()` (data-unit
    scaling) must stay "auto" regardless of Panel Aspect Ratio."""
    figure = _figure_with_series(panel_aspect="1:1")
    _mpl_figure, axes_list = _canvas_and_axes(figure)

    render_figure(axes_list, figure)

    assert axes_list[0].get_aspect() == "auto"


def test_panel_aspect_does_not_alter_autoscaled_data_limits():
    figure_auto = _figure_with_series(panel_aspect="Auto")
    figure_square = _figure_with_series(panel_aspect="1:1")
    _f1, axes_auto = _canvas_and_axes(figure_auto)
    _f2, axes_square = _canvas_and_axes(figure_square)

    render_figure(axes_auto, figure_auto)
    render_figure(axes_square, figure_square)

    assert axes_auto[0].get_xlim() == axes_square[0].get_xlim()
    assert axes_auto[0].get_ylim() == axes_square[0].get_ylim()


def test_panel_aspect_does_not_alter_manually_set_axis_limits():
    figure = _figure_with_series(panel_aspect="1:1")
    figure.active_panel.xlim = (0.0, 100.0)
    figure.active_panel.ylim = (-1.0, 1.0)
    _mpl_figure, axes_list = _canvas_and_axes(figure)

    render_figure(axes_list, figure)

    assert axes_list[0].get_xlim() == (0.0, 100.0)
    assert axes_list[0].get_ylim() == (-1.0, 1.0)


@pytest.mark.parametrize("layout", [(1, 2), (2, 2), (2, 3)])
def test_panel_aspect_applies_uniformly_to_every_panel_in_the_layout(layout):
    figure = _figure_with_series(layout=layout, panel_aspect="1:1")
    _mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(10.0, 6.0))

    render_figure(axes_list, figure)

    assert len(axes_list) == layout[0] * layout[1]
    for ax in axes_list:
        assert ax.get_box_aspect() == pytest.approx(1.0)


def test_2x3_with_4_3_panel_aspect_gives_every_panel_the_same_box_ratio():
    figure = _figure_with_series(layout=(2, 3), panel_aspect="4:3")
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(12.0, 6.0))

    render_figure(axes_list, figure)
    apply_figure_layout(mpl_figure, figure)
    mpl_figure.canvas.draw()

    ratios = [_box_ratio(ax, 12.0, 6.0) for ax in axes_list]
    for ratio in ratios:
        assert ratio == pytest.approx(4 / 3, rel=1e-6)


# --- Preview/export consistency: single algorithm, not duplicated ------------------


def test_preview_and_export_style_rendering_produce_the_same_box_ratio():
    """`render_panel` is the one place box_aspect is applied -- both the
    interactive preview (via `render_figure`) and `export.figure_export`
    (also via `render_figure`) call it, so there is no second, export-only
    panel-aspect algorithm to drift out of sync."""
    figure = _figure_with_series(layout=(2, 3), panel_aspect="1:1")

    preview_figure, preview_axes = _canvas_and_axes(figure, figsize=(9.6, 5.4))  # a different on-screen size
    render_figure(preview_axes, figure)
    apply_figure_layout(preview_figure, figure)
    preview_figure.canvas.draw()

    export_figure_obj = Figure(figsize=(16.0, 9.0), dpi=150)
    FigureCanvasAgg(export_figure_obj)
    export_axes = list(export_figure_obj.subplots(2, 3, squeeze=False).flat)
    render_figure(export_axes, figure)
    apply_figure_layout(export_figure_obj, figure)
    export_figure_obj.canvas.draw()

    preview_ratio = _box_ratio(preview_axes[0], 9.6, 5.4)
    export_ratio = _box_ratio(export_axes[0], 16.0, 9.0)
    assert preview_ratio == pytest.approx(1.0, rel=1e-6)
    assert export_ratio == pytest.approx(1.0, rel=1e-6)


def test_render_panel_accepts_no_figure_context_without_raising():
    """A bare single-panel `render_panel(ax, panel)` call (no `figure`
    argument) must still work -- box_aspect just stays unset (None), same
    as every other figure-supplied fallback in this function."""
    figure = _figure_with_series()
    panel = figure.active_panel
    mpl_figure = Figure()
    FigureCanvasAgg(mpl_figure)
    ax = mpl_figure.subplots()

    render_panel(ax, panel)  # must not raise

    assert ax.get_box_aspect() is None


def test_a_constrained_combination_renders_gracefully_without_crashing():
    """Tiny figure + wide panel spacing + a wide Panel Aspect Ratio on a
    2x3 grid: physically can't give every panel much room. Must never crash
    or distort -- Matplotlib's own box_aspect layout just shrinks each
    panel within its cell, which is graceful degradation, not a bug."""
    figure = _figure_with_series(layout=(2, 3), panel_aspect="16:9")
    figure.figure_width_in = 1.0
    figure.figure_height_in = 1.0
    figure.margin_left, figure.margin_right = 0.4, 0.6
    figure.margin_bottom, figure.margin_top = 0.4, 0.6
    figure.panel_wspace = figure.panel_hspace = 2.0
    mpl_figure, axes_list = _canvas_and_axes(figure, figsize=(1.0, 1.0))

    render_figure(axes_list, figure)
    apply_figure_layout(mpl_figure, figure)
    mpl_figure.canvas.draw()  # must not raise

    for ax in axes_list:
        pos = ax.get_position()
        assert pos.width > 0
        assert pos.height > 0
        assert ax.get_box_aspect() == pytest.approx(panel_box_aspect("16:9"))


def test_switching_from_a_preset_back_to_auto_clears_the_box_aspect():
    figure = _figure_with_series(panel_aspect="1:1")
    _mpl_figure, axes_list = _canvas_and_axes(figure)
    render_figure(axes_list, figure)
    assert axes_list[0].get_box_aspect() == pytest.approx(1.0)

    figure.panel_aspect_preset = "Auto"
    render_figure(axes_list, figure)

    assert axes_list[0].get_box_aspect() is None
