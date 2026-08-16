import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas, ReferenceCursorMode
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


# --- Aspect-locked responsive preview (letterbox/pillarbox) --------------------


def test_letterbox_rect_is_full_bleed_when_aspect_is_not_locked(qapp):
    figure = GnoviFigure()
    assert figure.lock_aspect_ratio is False
    canvas = PlotCanvas()
    canvas.resize(800, 600)

    assert canvas._letterbox_rect(figure) == (0.0, 0.0, 1.0, 1.0)


def test_letterbox_rect_pillarboxes_when_canvas_is_wider_than_target(qapp):
    figure = GnoviFigure(figure_width_in=4.0, figure_height_in=4.0, lock_aspect_ratio=True)
    canvas = PlotCanvas()
    canvas.resize(800, 400)  # 2:1, wider than the 1:1 target

    left, bottom, right, top = canvas._letterbox_rect(figure)

    assert bottom == pytest.approx(0.0)
    assert top == pytest.approx(1.0)
    assert left > 0.0
    assert right < 1.0
    assert (right - left) == pytest.approx(0.5, abs=0.01)  # 400/800 content fraction


def test_letterbox_rect_letterboxes_when_canvas_is_taller_than_target(qapp):
    figure = GnoviFigure(figure_width_in=4.0, figure_height_in=4.0, lock_aspect_ratio=True)
    canvas = PlotCanvas()
    canvas.resize(400, 800)  # 1:2, taller than the 1:1 target

    left, bottom, right, top = canvas._letterbox_rect(figure)

    assert left == pytest.approx(0.0)
    assert right == pytest.approx(1.0)
    assert bottom > 0.0
    assert top < 1.0
    assert (top - bottom) == pytest.approx(0.5, abs=0.01)


def test_letterbox_rect_matches_a_4_3_target_in_a_wider_canvas(qapp):
    figure = GnoviFigure(figure_width_in=8.0, figure_height_in=6.0, lock_aspect_ratio=True)  # 4:3
    canvas = PlotCanvas()
    canvas.resize(1000, 500)  # 2:1, wider than 4:3

    left, bottom, right, top = canvas._letterbox_rect(figure)
    content_w_px = (right - left) * canvas.width()
    content_h_px = (top - bottom) * canvas.height()

    assert bottom == pytest.approx(0.0)
    assert top == pytest.approx(1.0)
    assert content_w_px / content_h_px == pytest.approx(8.0 / 6.0, rel=1e-3)


def test_letterbox_rect_matches_a_16_9_target_in_a_taller_canvas(qapp):
    figure = GnoviFigure(figure_width_in=16.0, figure_height_in=9.0, lock_aspect_ratio=True)
    canvas = PlotCanvas()
    canvas.resize(600, 900)  # 2:3, taller than 16:9

    left, bottom, right, top = canvas._letterbox_rect(figure)
    content_w_px = (right - left) * canvas.width()
    content_h_px = (top - bottom) * canvas.height()

    assert left == pytest.approx(0.0)
    assert right == pytest.approx(1.0)
    assert content_w_px / content_h_px == pytest.approx(16.0 / 9.0, rel=1e-3)


def test_letterbox_rect_is_full_bleed_for_nonpositive_figure_size(qapp):
    figure = GnoviFigure(figure_width_in=0.0, figure_height_in=4.0, lock_aspect_ratio=True)
    canvas = PlotCanvas()
    canvas.resize(800, 600)

    assert canvas._letterbox_rect(figure) == (0.0, 0.0, 1.0, 1.0)


def test_resize_event_reapplies_letterbox_without_new_render_call(qapp):
    figure = GnoviFigure(figure_width_in=4.0, figure_height_in=4.0, lock_aspect_ratio=True)
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    canvas = PlotCanvas()
    canvas.render(figure)
    line_count_before = len(canvas.axes.lines)

    canvas.resize(300, 900)

    # Resizing must not re-render series data (no duplicate/dropped lines) --
    # only the letterbox margins should change.
    assert len(canvas.axes.lines) == line_count_before


def test_resize_event_is_a_no_op_before_the_first_render(qapp):
    canvas = PlotCanvas()
    canvas.resize(500, 400)  # must not raise even though nothing was rendered yet


# --- Active-panel badge (click-to-activate visual feedback) --------------------
#
# A plain Qt overlay widget (`_ActivePanelBadge`), never a spine/border
# color change on the Axes -- see `tests/test_active_panel_badge.py` for
# the full GUI-driven coverage (positioning, moving between panels,
# absence from Matplotlib Save/export). These stay here only as a smoke
# check that rendering with any panel count never raises and never mutates
# spine styling.


def test_single_panel_layout_renders_without_raising(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    canvas = PlotCanvas()

    canvas.render(figure)  # must not raise for a 1-panel layout

    assert len(canvas.axes_list) == 1


def test_activating_a_panel_never_changes_spine_color_or_width(qapp):
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.set_active_panel(1)
    canvas = PlotCanvas()

    canvas.render(figure)

    active_spine = canvas.axes_list[1].spines["top"]
    inactive_spine = canvas.axes_list[0].spines["top"]
    # Scientific axes styling stays completely separate from GUI selection
    # state -- both panels' spines look identical (default black, default
    # width) regardless of which one is active.
    assert active_spine.get_edgecolor() == inactive_spine.get_edgecolor()
    assert active_spine.get_linewidth() == inactive_spine.get_linewidth()


def test_panel_index_for_axes_resolves_click_targets(qapp):
    figure = GnoviFigure()
    figure.set_layout(2, 1)
    canvas = PlotCanvas()
    canvas.render(figure)

    assert canvas.panel_index_for_axes(canvas.axes_list[0]) == 0
    assert canvas.panel_index_for_axes(canvas.axes_list[1]) == 1


def test_panel_index_for_axes_returns_none_for_a_foreign_axes(qapp):
    from matplotlib.figure import Figure

    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    foreign_ax = Figure().subplots()

    assert canvas.panel_index_for_axes(foreign_ax) is None


# --- Reference cursor (Off / X line / Y line / Crosshair) ----------------------


def test_reference_cursor_defaults_to_off(qapp):
    canvas = PlotCanvas()
    assert canvas._cursor_mode == ReferenceCursorMode.OFF


def test_x_line_mode_draws_only_a_vertical_line(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    canvas.set_cursor_mode(ReferenceCursorMode.X_LINE)

    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)

    assert len(canvas.axes.lines) == 1
    assert len(canvas._cursor_artists) == 1


def test_y_line_mode_draws_only_a_horizontal_line(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    canvas.set_cursor_mode(ReferenceCursorMode.Y_LINE)

    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)

    assert len(canvas._cursor_artists) == 1


def test_crosshair_mode_draws_both_lines(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    canvas.set_cursor_mode(ReferenceCursorMode.CROSSHAIR)

    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)

    assert len(canvas._cursor_artists) == 2


def test_off_mode_draws_nothing(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    canvas.set_cursor_mode(ReferenceCursorMode.OFF)

    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)

    assert canvas._cursor_artists == []


def test_switching_modes_removes_the_previous_overlay(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    canvas.set_cursor_mode(ReferenceCursorMode.CROSSHAIR)
    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)
    assert len(canvas._cursor_artists) == 2

    canvas.set_cursor_mode(ReferenceCursorMode.X_LINE)
    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)

    assert len(canvas._cursor_artists) == 1


def test_clear_reference_cursor_removes_the_overlay(qapp):
    figure = GnoviFigure()
    canvas = PlotCanvas()
    canvas.render(figure)
    canvas.set_cursor_mode(ReferenceCursorMode.CROSSHAIR)
    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)

    canvas.clear_reference_cursor()

    assert canvas._cursor_artists == []


def test_re_render_drops_stale_cursor_artists_without_raising(qapp):
    """`render()` calls `ax.cla()` per panel, which already discards any
    cursor artists -- a second `render()` after drawing a cursor overlay
    must not try (and fail) to `.remove()` an artist Matplotlib already
    discarded."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    canvas = PlotCanvas()
    canvas.render(figure)
    canvas.set_cursor_mode(ReferenceCursorMode.CROSSHAIR)
    canvas.update_reference_cursor(canvas.axes, 2.0, 3.0)
    assert len(canvas._cursor_artists) == 2

    canvas.render(figure)  # must not raise

    assert canvas._cursor_artists == []
