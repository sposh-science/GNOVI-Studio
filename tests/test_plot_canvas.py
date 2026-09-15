import pandas as pd

from gnovi_plot.analysis.cycles import detect_cycles
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas
from gnovi_plot.plotting.figure import GnoviFigure, Panel
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


def test_render_line_series_draws_a_line(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes.lines) == 1


def test_render_scatter_series_draws_a_collection(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.scatter(_make_dataset(), "x", "y"))

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes.collections) == 1


def test_render_histogram_series_draws_patches(qapp):
    df = pd.DataFrame({"current": [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]})
    dataset = Dataset(name="hist", dataframe=df)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.histogram(dataset, "current"))

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes.patches) > 0


def test_render_skips_hidden_series(qapp):
    figure = GnoviFigure()
    series = PlotSeries.line(_make_dataset(), "x", "y")
    series.visible = False
    figure.add_series(series)

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes.lines) == 0


def test_render_applies_manual_axis_limits(qapp):
    figure = GnoviFigure(xlim=(0.0, 10.0), ylim=(-5.0, 5.0))
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))

    canvas = PlotCanvas()
    canvas.render(figure)

    assert canvas.axes.get_xlim() == (0.0, 10.0)
    assert canvas.axes.get_ylim() == (-5.0, 5.0)


def test_render_multiple_series_overlay_on_same_axes(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset("a"), "x", "y"))
    figure.add_series(PlotSeries.line(_make_dataset("b"), "x", "y"))

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes.lines) == 2


def test_render_draws_each_detected_cycle_as_a_separate_line_with_distinct_colors(qapp):
    leg = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0]
    x = leg + leg[1:] + leg[1:]
    y = [float(i) for i in range(len(x))]
    df = pd.DataFrame({"Potential/V": x, "Current/A": y})
    dataset = Dataset(name="cv", dataframe=df)

    cycles = detect_cycles(dataset.dataframe, "Potential/V")
    assert len(cycles) == 3

    figure = GnoviFigure()
    for i, row_range in enumerate(cycles):
        figure.add_series(
            PlotSeries.line(
                dataset,
                "Potential/V",
                "Current/A",
                label=f"cv — Cycle {i + 1}",
                row_range=row_range,
            )
        )

    canvas = PlotCanvas()
    canvas.render(figure)

    assert len(canvas.axes.lines) == 3
    assert len({series.color for series in figure.series}) == 3


def test_render_does_not_mutate_dataset_dataframe(qapp):
    df = pd.DataFrame({"x": ["1", "bad", "3"], "y": ["4", "5", "6"]})
    dataset = Dataset(name="messy", dataframe=df)
    original = dataset.dataframe.copy(deep=True)

    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    canvas = PlotCanvas()
    canvas.render(figure)

    pd.testing.assert_frame_equal(dataset.dataframe, original)


# --- Review-fix regression: set_analysis_overlay's own Axes resolution
# (code-review finding #5) ---------------------------------------------------
#
# Compared directly against active_axes(): equivalent for every in-bounds/
# focused case (both resolve the identical Axes), but deliberately NOT
# reused, because active_axes() has no bounds guard and would raise/
# misbehave for an out-of-range active_panel_index -- a case
# set_analysis_overlay must tolerate as a safe no-op, since an XRD overlay
# refresh can be triggered independently of a matching render(). See
# set_analysis_overlay's own docstring for the full reasoning.


def _two_panel_figure_and_canvas():
    figure = GnoviFigure(panels=[Panel(), Panel()], layout=(1, 2))
    ds = _make_dataset()
    figure.panels[0].add_series(PlotSeries.line(ds, "x", "y"))
    figure.panels[1].add_series(PlotSeries.line(ds, "x", "y"))
    canvas = PlotCanvas()
    canvas.render(figure)
    return figure, canvas


def test_set_analysis_overlay_targets_the_same_axes_as_active_axes(qapp):
    figure, canvas = _two_panel_figure_and_canvas()
    figure.set_active_panel(1)
    canvas.set_analysis_overlay(figure, peak_points=[(1.0, 2.0, "")], preview_xy=None)
    assert canvas._analysis_overlay_artists
    (artist,) = canvas._analysis_overlay_artists
    assert artist.axes is canvas.active_axes(figure)
    assert canvas.active_axes(figure) is canvas.axes_list[1]


def test_set_analysis_overlay_is_a_safe_noop_when_axes_list_is_stale(qapp):
    # The realistic version of "active_panel_index out of range for
    # axes_list": figure.panels/active_panel_index stay self-consistent
    # (GnoviFigure enforces that, see set_active_panel/remove_series), but
    # a THIRD panel was just added to the model and made active before
    # this canvas has re-rendered to match -- axes_list (still length 2)
    # is what's actually stale, exactly the "independent of any render"
    # case set_analysis_overlay's own docstring describes.
    figure, canvas = _two_panel_figure_and_canvas()
    figure.panels.append(Panel())
    figure.panels[2].add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    figure.set_active_panel(2)  # valid for figure.panels; NOT valid for canvas.axes_list yet
    # Must not raise (active_axes() itself would IndexError here) and must
    # leave no overlay artists behind.
    canvas.set_analysis_overlay(figure, peak_points=[(1.0, 2.0, "")], preview_xy=None)
    assert canvas._analysis_overlay_artists == []


def test_set_analysis_overlay_is_a_safe_noop_for_a_negative_active_panel_index(qapp):
    figure, canvas = _two_panel_figure_and_canvas()
    figure.active_panel_index = -1  # active_axes() would silently wrap to axes_list[-1]
    canvas.set_analysis_overlay(figure, peak_points=[(1.0, 2.0, "")], preview_xy=None)
    assert canvas._analysis_overlay_artists == []


def test_set_analysis_overlay_in_focus_mode_matches_active_axes(qapp):
    figure, canvas = _two_panel_figure_and_canvas()
    canvas.render(figure, focused_panel=figure.panels[1])
    assert canvas.is_focused
    canvas.set_analysis_overlay(figure, peak_points=[(1.0, 2.0, "")], preview_xy=None)
    assert canvas._analysis_overlay_artists
    (artist,) = canvas._analysis_overlay_artists
    assert artist.axes is canvas.active_axes(figure)
    assert canvas.active_axes(figure) is canvas.axes_list[0]


# --- set_analysis_overlay: viewport stability -------------------------------
#
# Transient overlay artists (background/smoothing preview, peak markers,
# fit-window span, fit curves) must never move the viewport the researcher
# is currently looking at -- see set_analysis_overlay's own docstring for
# why: Matplotlib's autoscale participates in every ax.plot()/scatter()/
# axvspan() call by default, and clear_analysis_overlay()'s artist.remove()
# does not retract a removed artist's earlier ax.dataLim contribution, so
# left alone that accumulates across repeated refreshes. Every scenario
# below intentionally uses overlay data OUTSIDE the rendered series' own
# range, so a regression (the fix silently dropped) would show up as a
# genuine limit change here, not pass by coincidence.


def _one_panel_figure_and_canvas():
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    canvas = PlotCanvas()
    canvas.render(figure)
    return figure, canvas


def test_background_preview_overlay_does_not_change_the_viewport(qapp):
    figure, canvas = _one_panel_figure_and_canvas()
    xlim, ylim = canvas.axes.get_xlim(), canvas.axes.get_ylim()

    # Well outside _make_dataset()'s x in [1, 4] / y in [1, 16].
    preview_xy = ([-50.0, 50.0], [-500.0, -500.0])
    canvas.set_analysis_overlay(figure, peak_points=None, preview_xy=preview_xy)

    assert canvas.axes.get_xlim() == xlim
    assert canvas.axes.get_ylim() == ylim


def test_peak_marker_overlay_does_not_change_the_viewport(qapp):
    figure, canvas = _one_panel_figure_and_canvas()
    xlim, ylim = canvas.axes.get_xlim(), canvas.axes.get_ylim()

    canvas.set_analysis_overlay(
        figure, peak_points=[(-50.0, 500.0, "far outside")], preview_xy=None
    )

    assert canvas.axes.get_xlim() == xlim
    assert canvas.axes.get_ylim() == ylim


def test_fit_window_and_fit_curve_overlays_do_not_change_the_viewport(qapp):
    figure, canvas = _one_panel_figure_and_canvas()
    xlim, ylim = canvas.axes.get_xlim(), canvas.axes.get_ylim()

    canvas.set_analysis_overlay(
        figure,
        peak_points=None,
        preview_xy=None,
        fit_window=(-40.0, -30.0),
        fit_curves={
            "baseline_xy": ([-40.0, -30.0], [-200.0, -200.0]),
            "total_xy": ([-40.0, -30.0], [300.0, 300.0]),
        },
    )

    assert canvas.axes.get_xlim() == xlim
    assert canvas.axes.get_ylim() == ylim


def test_repeated_identical_overlay_is_idempotent_on_the_viewport(qapp):
    figure, canvas = _one_panel_figure_and_canvas()
    xlim, ylim = canvas.axes.get_xlim(), canvas.axes.get_ylim()
    preview_xy = ([-50.0, 50.0], [-500.0, -500.0])

    for _ in range(5):
        canvas.set_analysis_overlay(figure, peak_points=None, preview_xy=preview_xy)
        assert canvas.axes.get_xlim() == xlim
        assert canvas.axes.get_ylim() == ylim
        assert len(canvas._analysis_overlay_artists) == 1


def test_changed_overlay_replaces_rather_than_accumulates(qapp):
    figure, canvas = _one_panel_figure_and_canvas()

    canvas.set_analysis_overlay(
        figure, peak_points=None, preview_xy=([1.0, 2.0], [3.0, 4.0])
    )
    assert len(canvas._analysis_overlay_artists) == 1
    (first_line,) = canvas._analysis_overlay_artists

    canvas.set_analysis_overlay(
        figure, peak_points=None, preview_xy=([5.0, 6.0], [7.0, 8.0])
    )
    assert len(canvas._analysis_overlay_artists) == 1
    (second_line,) = canvas._analysis_overlay_artists
    assert second_line is not first_line
    assert list(second_line.get_ydata()) == [7.0, 8.0]


def test_genuine_render_still_autoscales_normally(qapp):
    """The viewport-preservation fix must be scoped to the transient-
    overlay path only -- a real render() with new/wider data must still
    autoscale exactly as before."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    canvas = PlotCanvas()
    canvas.render(figure)
    narrow_xlim = canvas.axes.get_xlim()

    wide_dataset = Dataset(
        name="wide", dataframe=pd.DataFrame({"x": [-100.0, 100.0], "y": [-100.0, 100.0]})
    )
    figure.add_series(PlotSeries.line(wide_dataset, "x", "y"))
    canvas.render(figure)

    assert canvas.axes.get_xlim() != narrow_xlim
    assert canvas.axes.get_xlim()[0] < -50
    assert canvas.axes.get_xlim()[1] > 50
