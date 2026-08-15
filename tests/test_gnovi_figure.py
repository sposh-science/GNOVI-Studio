import pandas as pd

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.plotting.figure import GnoviFigure
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
