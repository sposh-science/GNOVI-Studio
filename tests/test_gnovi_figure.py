import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure, PlotTheme
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


def test_add_series_assigns_a_default_color_when_none_given():
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y")
    assert series.color is None

    figure.add_series(series)
    assert series.color is not None


def test_add_series_respects_an_explicit_color():
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y", color="#ff00ff")
    figure.add_series(series)
    assert series.color == "#ff00ff"


def test_auto_assigned_color_is_not_marked_manual():
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y")

    figure.add_series(series)

    assert series.color_is_manual is False


def test_add_series_uses_the_dark_theme_cycle_when_dark_mode_is_true():
    from gnovi_plot.plotting.figure import theme_color_cycle

    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y")

    figure.add_series(series, dark_mode=True)

    assert series.color == theme_color_cycle(dark_mode=True)[0]
    assert series.color != theme_color_cycle(dark_mode=False)[0]


def test_add_series_uses_the_light_theme_cycle_by_default():
    from gnovi_plot.plotting.figure import theme_color_cycle

    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y")

    figure.add_series(series)

    assert series.color == theme_color_cycle(dark_mode=False)[0]


def test_default_colors_are_distinct_and_stable_across_hide():
    figure = GnoviFigure()
    a = PlotSeries.line(_make_dataset("a"), "x", "y")
    b = PlotSeries.line(_make_dataset("b"), "x", "y")
    figure.add_series(a)
    figure.add_series(b)

    assert a.color != b.color

    a.visible = False
    color_a, color_b = a.color, b.color

    c = PlotSeries.line(_make_dataset("c"), "x", "y")
    figure.add_series(c)

    assert a.color == color_a
    assert b.color == color_b


def test_get_series_returns_none_for_unknown_id():
    figure = GnoviFigure()
    assert figure.get_series("does-not-exist") is None


def test_remove_series_only_removes_the_target():
    figure = GnoviFigure()
    a = PlotSeries.line(_make_dataset("a"), "x", "y")
    b = PlotSeries.line(_make_dataset("b"), "x", "y")
    figure.add_series(a)
    figure.add_series(b)

    figure.remove_series(a.id)

    assert [s.id for s in figure.series] == [b.id]


def test_clear_series_empties_the_list_and_resets_color_cycle():
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset("a"), "x", "y"))
    figure.clear_series()

    assert figure.series == []

    fresh = PlotSeries.line(_make_dataset("b"), "x", "y")
    figure.add_series(fresh)
    assert fresh.color is not None


def test_reset_limits_clears_manual_xlim_and_ylim():
    figure = GnoviFigure(xlim=(0, 1), ylim=(0, 1))
    figure.reset_limits()
    assert figure.xlim is None
    assert figure.ylim is None


# --- Project-save serialization -----------------------------------------------


def _elaborate_figure():
    """A 2x2 figure exercising every category of Panel/GnoviFigure state:
    layout, series assignment, axes/labels/limits/scales, ticks, grid,
    legend, typography, figure size/ratio, margins/spacing, panel labels."""
    ds_a = _make_dataset("a")
    ds_b = _make_dataset("b")
    figure = GnoviFigure(
        name="Overlay Figure",
        figure_width_in=8.0,
        figure_height_in=5.0,
        aspect_preset="Custom",
        lock_aspect_ratio=True,
        panel_aspect_preset="1:1",
        font_family="DejaVu Sans",
        base_font_size=11.0,
        title_font_size=14.0,
        axis_label_font_size=12.0,
        tick_label_font_size=10.0,
        legend_font_size=10.0,
        grid_linestyle=":",
        grid_linewidth=1.2,
        grid_alpha=0.4,
        grid_color="#888888",
        margin_left=0.15,
        margin_right=0.85,
        margin_bottom=0.15,
        margin_top=0.85,
        panel_wspace=0.3,
        panel_hspace=0.35,
        panel_labels_visible=True,
    )
    figure.set_layout(2, 2)
    figure.set_active_panel(0)
    figure.active_panel.title = "Panel A"
    figure.active_panel.xlabel = "X (a.u.)"
    figure.active_panel.ylabel = "Y (a.u.)"
    figure.active_panel.xlim = (-1.0, 5.0)
    figure.active_panel.ylim = (0.0, 10.0)
    figure.active_panel.xscale = "log"
    figure.active_panel.yscale = "linear"
    figure.active_panel.invert_x = True
    figure.active_panel.grid = True
    figure.active_panel.grid_which = "both"
    figure.active_panel.legend_visible = True
    figure.active_panel.legend_loc = "upper right"
    figure.active_panel.legend_ncol = 2
    figure.active_panel.legend_title = "Series"
    figure.active_panel.tick_direction = "inout"
    figure.active_panel.minor_ticks = True
    figure.active_panel.major_tick_spacing_x = 1.0
    figure.active_panel.minor_tick_spacing_y = 0.5
    figure.active_panel.scientific_notation_y = True
    figure.active_panel.spine_right = False
    figure.active_panel.spine_linewidth = 2.0
    figure.add_series(PlotSeries.line(ds_a, "x", "y", color="#123456"))
    figure.add_series(PlotSeries.scatter(ds_b, "x", "y", row_range=(0, 2)))

    figure.set_active_panel(1)
    figure.active_panel.panel_label = "(b)"
    figure.add_series(PlotSeries.histogram(ds_a, "y", bins=5))

    return figure, ds_a, ds_b


def test_gnovi_figure_to_dict_round_trip_preserves_layout_and_active_panel():
    figure, ds_a, ds_b = _elaborate_figure()
    data = figure.to_dict()
    lookup = {ds_a.id: ds_a, ds_b.id: ds_b}
    restored = GnoviFigure.from_dict(data, lookup)

    assert restored.name == "Overlay Figure"
    assert restored.layout == (2, 2)
    assert len(restored.panels) == 4
    assert [p.series for p in restored.panels[2:]] == [[], []]  # empty panels stay empty


def test_gnovi_figure_round_trip_preserves_series_and_shared_dataset_identity():
    figure, ds_a, ds_b = _elaborate_figure()
    lookup = {ds_a.id: ds_a, ds_b.id: ds_b}
    restored = GnoviFigure.from_dict(figure.to_dict(), lookup)

    panel0 = restored.panels[0]
    assert len(panel0.series) == 2
    assert panel0.series[0].dataset is ds_a
    assert panel0.series[0].color == "#123456"
    assert panel0.series[1].dataset is ds_b
    assert panel0.series[1].row_range == (0, 2)

    panel1 = restored.panels[1]
    assert len(panel1.series) == 1
    assert panel1.series[0].plot_type.value == "histogram"
    assert panel1.series[0].bins == 5


def test_gnovi_figure_round_trip_preserves_axes_labels_limits_and_scales():
    figure, ds_a, ds_b = _elaborate_figure()
    lookup = {ds_a.id: ds_a, ds_b.id: ds_b}
    restored = GnoviFigure.from_dict(figure.to_dict(), lookup)
    panel0 = restored.panels[0]

    assert panel0.title == "Panel A"
    assert panel0.xlabel == "X (a.u.)"
    assert panel0.ylabel == "Y (a.u.)"
    assert panel0.xlim == (-1.0, 5.0)
    assert isinstance(panel0.xlim, tuple)
    assert panel0.ylim == (0.0, 10.0)
    assert panel0.xscale == "log"
    assert panel0.yscale == "linear"
    assert panel0.invert_x is True


def test_gnovi_figure_round_trip_preserves_ticks_grid_and_legend():
    figure, ds_a, ds_b = _elaborate_figure()
    lookup = {ds_a.id: ds_a, ds_b.id: ds_b}
    restored = GnoviFigure.from_dict(figure.to_dict(), lookup)
    panel0 = restored.panels[0]

    assert panel0.grid is True
    assert panel0.grid_which == "both"
    assert panel0.legend_visible is True
    assert panel0.legend_loc == "upper right"
    assert panel0.legend_ncol == 2
    assert panel0.legend_title == "Series"
    assert panel0.tick_direction == "inout"
    assert panel0.minor_ticks is True
    assert panel0.major_tick_spacing_x == 1.0
    assert panel0.minor_tick_spacing_y == 0.5
    assert panel0.scientific_notation_y is True
    assert panel0.spine_right is False
    assert panel0.spine_linewidth == 2.0


def test_gnovi_figure_round_trip_preserves_typography_size_ratio_and_margins():
    figure, ds_a, ds_b = _elaborate_figure()
    lookup = {ds_a.id: ds_a, ds_b.id: ds_b}
    restored = GnoviFigure.from_dict(figure.to_dict(), lookup)

    assert restored.figure_width_in == 8.0
    assert restored.figure_height_in == 5.0
    assert restored.aspect_preset == "Custom"
    assert restored.lock_aspect_ratio is True
    assert restored.panel_aspect_preset == "1:1"
    assert restored.font_family == "DejaVu Sans"
    assert restored.base_font_size == 11.0
    assert restored.title_font_size == 14.0
    assert restored.axis_label_font_size == 12.0
    assert restored.tick_label_font_size == 10.0
    assert restored.legend_font_size == 10.0
    assert restored.grid_linestyle == ":"
    assert restored.grid_linewidth == 1.2
    assert restored.grid_alpha == 0.4
    assert restored.grid_color == "#888888"
    assert restored.margin_left == 0.15
    assert restored.margin_right == 0.85
    assert restored.margin_bottom == 0.15
    assert restored.margin_top == 0.85
    assert restored.panel_wspace == 0.3
    assert restored.panel_hspace == 0.35


def test_gnovi_figure_round_trip_preserves_panel_labels_visibility_and_text():
    figure, ds_a, ds_b = _elaborate_figure()
    lookup = {ds_a.id: ds_a, ds_b.id: ds_b}
    restored = GnoviFigure.from_dict(figure.to_dict(), lookup)

    assert restored.panel_labels_visible is True
    # panel_label is deterministically re-derived from position on
    # construction (`_renumber_panel_labels`), so it comes back as "(a)"
    # regardless of what was explicitly set before save.
    assert restored.panels[0].panel_label == "(a)"
    assert restored.panels[1].panel_label == "(b)"


def test_gnovi_figure_from_dict_drops_series_with_unresolvable_dataset_id():
    figure, ds_a, ds_b = _elaborate_figure()
    data = figure.to_dict()
    # Only ds_a is available on load -- any series referencing ds_b must be
    # dropped, not raise, and the rest of the panel must load fine.
    restored = GnoviFigure.from_dict(data, {ds_a.id: ds_a})

    panel0 = restored.panels[0]
    assert len(panel0.series) == 1
    assert panel0.series[0].dataset is ds_a


def test_gnovi_figure_from_dict_clamps_out_of_range_active_panel_index():
    figure, ds_a, ds_b = _elaborate_figure()
    data = figure.to_dict()
    data["active_panel_index"] = 99
    restored = GnoviFigure.from_dict(data, {ds_a.id: ds_a, ds_b.id: ds_b})
    assert restored.active_panel_index == len(restored.panels) - 1
    assert restored.active_panel is restored.panels[-1]  # doesn't raise


def test_gnovi_figure_from_dict_falls_back_to_a_default_panel_when_panels_list_is_empty():
    data = GnoviFigure().to_dict()
    data["panels"] = []
    restored = GnoviFigure.from_dict(data, {})
    assert len(restored.panels) == 1
    assert restored.active_panel is restored.panels[0]  # doesn't raise


# --- Plot Theme: declarative GnoviFigure state --------------------------------


def test_plot_theme_defaults_to_light():
    figure = GnoviFigure()
    assert figure.plot_theme == PlotTheme.LIGHT


def test_plot_theme_light_figure_round_trips_through_to_dict_from_dict():
    figure = GnoviFigure(plot_theme=PlotTheme.LIGHT)
    data = figure.to_dict()
    assert data["plot_theme"] == "light"

    restored = GnoviFigure.from_dict(data, {})
    assert restored.plot_theme == PlotTheme.LIGHT
    assert restored.plot_theme is PlotTheme.LIGHT  # the actual enum member, not a lookalike string


def test_plot_theme_dark_figure_round_trips_through_to_dict_from_dict():
    figure = GnoviFigure(plot_theme=PlotTheme.DARK)
    data = figure.to_dict()
    assert data["plot_theme"] == "dark"

    restored = GnoviFigure.from_dict(data, {})
    assert restored.plot_theme == PlotTheme.DARK
    assert restored.plot_theme is PlotTheme.DARK


def test_plot_theme_missing_from_saved_data_falls_back_to_light():
    data = GnoviFigure().to_dict()
    del data["plot_theme"]
    restored = GnoviFigure.from_dict(data, {})
    assert restored.plot_theme == PlotTheme.LIGHT


def test_plot_theme_invalid_value_in_saved_data_falls_back_to_light_instead_of_raising():
    data = GnoviFigure().to_dict()
    data["plot_theme"] = "not-a-real-theme"
    restored = GnoviFigure.from_dict(data, {})
    assert restored.plot_theme == PlotTheme.LIGHT
