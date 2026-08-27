"""Toolbar "Zoom Out" (`MainWindow._on_zoom_out` + `self._zoom_out_action`,
placed on Matplotlib's navigation toolbar next to "Zoom").

Zoom Out is a pure VIEW operation on the active 2D panel: it widens the
current X/Y view about their centers by a fixed step, participates in the
built-in Home/Back/Forward history, and never touches the Dataset,
PlotSeries, Panel model, project dirty flag, or the undo stack. These
tests exercise it through the real QAction the toolbar button triggers.
"""

import pandas as pd
import pytest
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.figure import Panel3D
from gnovi_plot.plotting.navigation import ZOOM_OUT_FACTOR
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    return Dataset(
        name=name,
        dataframe=pd.DataFrame(
            {"x": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], "y": [0.0, 4.0, 16.0, 36.0, 64.0, 100.0]}
        ),
    )


def _make_window(layout_index=None):
    window = MainWindow()
    if layout_index is not None:
        window.figure_size_panel.layout_combo.setCurrentIndex(layout_index)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for i, panel in enumerate(window.figure_model.panels):
        panel.add_series(PlotSeries.line(dataset, "x", "y", label=f"S{i + 1}"))
    window._on_figure_content_changed()  # one baseline undo checkpoint (see focus-panel fixture)
    window._set_dirty(False)
    return window, dataset


def _active_ax(window):
    return window.plot_canvas.active_axes(window.figure_model)


def _center(lo_hi):
    return (lo_hi[0] + lo_hi[1]) / 2


def _zoom_out(window):
    window._zoom_out_action.trigger()


# --- Core behaviour ---------------------------------------------------------------


def test_one_click_expands_the_x_range_about_its_center(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_xlim(20.0, 40.0)
    window.plot_canvas.draw()

    _zoom_out(window)

    assert ax.get_xlim() == pytest.approx((17.5, 42.5))
    window.close()


def test_one_click_expands_the_y_range_about_its_center(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_ylim(0.0, 10.0)
    window.plot_canvas.draw()

    _zoom_out(window)

    assert ax.get_ylim() == pytest.approx((-1.25, 11.25))
    window.close()


def test_center_stays_fixed_across_clicks(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_xlim(10.0, 30.0)
    ax.set_ylim(-5.0, 15.0)
    window.plot_canvas.draw()
    x_center, y_center = _center(ax.get_xlim()), _center(ax.get_ylim())

    for _ in range(4):
        _zoom_out(window)
        assert _center(ax.get_xlim()) == pytest.approx(x_center)
        assert _center(ax.get_ylim()) == pytest.approx(y_center)
    window.close()


def test_repeated_clicks_progressively_zoom_out(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_xlim(0.0, 10.0)
    window.plot_canvas.draw()

    widths = [ax.get_xlim()[1] - ax.get_xlim()[0]]
    for _ in range(3):
        _zoom_out(window)
        widths.append(ax.get_xlim()[1] - ax.get_xlim()[0])

    assert widths == sorted(widths)
    assert widths[-1] == pytest.approx(10.0 * ZOOM_OUT_FACTOR**3)
    assert widths[1] / widths[0] == pytest.approx(ZOOM_OUT_FACTOR)
    window.close()


def test_zoom_out_does_not_reset_to_full_data_extent(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_xlim(4.0, 6.0)  # zoomed well inside the data (x runs 0..10)
    window.plot_canvas.draw()

    _zoom_out(window)

    lo, hi = ax.get_xlim()
    assert (lo, hi) == pytest.approx((3.75, 6.25))  # a step, not a jump to ~0..10
    window.close()


# --- Inverted / log axes ---------------------------------------------------------


def test_inverted_x_axis_stays_inverted(qapp):
    window, _ = _make_window()
    window.figure_model.active_panel.invert_x = True
    window._rerender()
    ax = _active_ax(window)
    assert ax.get_xlim()[0] > ax.get_xlim()[1]

    _zoom_out(window)

    lo, hi = ax.get_xlim()
    assert lo > hi  # still inverted
    window.close()


def test_inverted_y_axis_stays_inverted(qapp):
    window, _ = _make_window()
    window.figure_model.active_panel.invert_y = True
    window._rerender()
    ax = _active_ax(window)
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    y_center = _center(ax.get_ylim())

    _zoom_out(window)

    lo, hi = ax.get_ylim()
    assert lo > hi
    assert _center((lo, hi)) == pytest.approx(y_center)
    window.close()


def test_log_x_axis_zooms_multiplicatively(qapp):
    window, _ = _make_window()
    window.figure_model.active_panel.xscale = "log"
    window._rerender()
    ax = _active_ax(window)
    ax.set_xlim(1.0, 100.0)
    window.plot_canvas.draw()

    _zoom_out(window)

    lo, hi = ax.get_xlim()
    assert lo > 0.0
    assert (lo * hi) ** 0.5 == pytest.approx(10.0)  # geometric center fixed
    window.close()


# --- State semantics: no model / dirty / undo mutation --------------------------


def test_zoom_out_never_marks_the_project_dirty_or_adds_an_undo_checkpoint(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_xlim(1.0, 9.0)
    window.plot_canvas.draw()
    undo_before = len(window._undo_manager._undo)

    for _ in range(5):
        _zoom_out(window)

    assert window._dirty is False
    assert len(window._undo_manager._undo) == undo_before
    window.close()


def test_zoom_out_never_mutates_the_dataset_or_panel_model(qapp):
    window, dataset = _make_window()
    panel = window.figure_model.active_panel
    before = {
        "xlim": panel.xlim,
        "ylim": panel.ylim,
        "xscale": panel.xscale,
        "invert_x": panel.invert_x,
        "invert_y": panel.invert_y,
    }
    dataframe_x = list(dataset.dataframe["x"])
    ax = _active_ax(window)
    ax.set_xlim(2.0, 5.0)
    window.plot_canvas.draw()

    for _ in range(3):
        _zoom_out(window)

    assert panel.xlim == before["xlim"] and panel.ylim == before["ylim"]
    assert panel.xscale == before["xscale"]
    assert panel.invert_x == before["invert_x"] and panel.invert_y == before["invert_y"]
    assert list(dataset.dataframe["x"]) == dataframe_x
    window.close()


# --- Multi-panel: active panel only --------------------------------------------


def test_zoom_out_only_touches_the_active_panel(qapp):
    window, _ = _make_window(layout_index=4)  # "1 x 3"
    window._set_active_panel(1)
    window._rerender()
    ax_active = window.plot_canvas.axes_list[1]
    ax_other_0 = window.plot_canvas.axes_list[0]
    ax_other_2 = window.plot_canvas.axes_list[2]
    ax_active.set_xlim(0.0, 10.0)
    window.plot_canvas.draw()
    other_0_lims = (ax_other_0.get_xlim(), ax_other_0.get_ylim())
    other_2_lims = (ax_other_2.get_xlim(), ax_other_2.get_ylim())

    _zoom_out(window)

    assert ax_active.get_xlim() == pytest.approx((-1.25, 11.25))
    assert (ax_other_0.get_xlim(), ax_other_0.get_ylim()) == other_0_lims
    assert (ax_other_2.get_xlim(), ax_other_2.get_ylim()) == other_2_lims
    window.close()


def test_zoom_out_follows_active_panel_switching(qapp):
    window, _ = _make_window(layout_index=4)
    window._set_active_panel(2)
    window._rerender()
    window.plot_canvas.axes_list[2].set_xlim(0.0, 10.0)
    window.plot_canvas.draw()
    # Snapshot the non-active panels' views right before the click.
    other_lims = {
        i: (window.plot_canvas.axes_list[i].get_xlim(), window.plot_canvas.axes_list[i].get_ylim())
        for i in (0, 1)
    }

    _zoom_out(window)

    assert window.plot_canvas.axes_list[2].get_xlim() == pytest.approx((-1.25, 11.25))
    for i, lims in other_lims.items():
        assert (window.plot_canvas.axes_list[i].get_xlim(), window.plot_canvas.axes_list[i].get_ylim()) == lims
    window.close()


# --- Focus mode / Workbench switching -----------------------------------------


def test_zoom_out_works_in_focus_mode(qapp):
    window, _ = _make_window(layout_index=4)
    window._set_active_panel(1)
    window._focus_panel(window.figure_model.active_panel)
    assert window.plot_canvas.is_focused
    ax = _active_ax(window)
    ax.set_xlim(20.0, 40.0)
    window.plot_canvas.draw()

    _zoom_out(window)

    assert ax.get_xlim() == pytest.approx((17.5, 42.5))
    window.close()


def test_zoom_out_works_after_switching_workbenches(qapp):
    window, dataset = _make_window()
    workbench_a = window._project.active_workbench_id

    window.workbench_tab_bar.new_button.click()  # Workbench B
    window._on_workbench_tab_selected(workbench_a)  # back to A
    window._rerender()
    ax = _active_ax(window)
    ax.set_xlim(10.0, 30.0)
    window.plot_canvas.draw()
    window._set_dirty(False)

    _zoom_out(window)

    assert ax.get_xlim() == pytest.approx((7.5, 32.5))
    assert window._dirty is False
    window.close()


# --- Home / Back / Forward coherence ------------------------------------------


def _nav(window):
    return window.findChildren(NavigationToolbar2QT)[0]


def test_home_returns_to_the_pre_zoom_view(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_xlim(20.0, 40.0)
    ax.set_ylim(0.0, 10.0)
    window.plot_canvas.draw()

    _zoom_out(window)
    _zoom_out(window)
    _nav(window).home()

    assert ax.get_xlim() == pytest.approx((20.0, 40.0))
    assert ax.get_ylim() == pytest.approx((0.0, 10.0))
    window.close()


def test_back_steps_out_one_zoom_click_at_a_time(qapp):
    window, _ = _make_window()
    ax = _active_ax(window)
    ax.set_xlim(0.0, 10.0)
    window.plot_canvas.draw()

    _zoom_out(window)
    after_one = ax.get_xlim()
    _zoom_out(window)
    after_two = ax.get_xlim()
    assert after_two != after_one

    _nav(window).back()
    assert ax.get_xlim() == pytest.approx(after_one)

    _nav(window).forward()
    assert ax.get_xlim() == pytest.approx(after_two)
    window.close()


# --- Panel3D: disabled / no-op ------------------------------------------------


def test_zoom_out_action_is_disabled_for_a_3d_panel(qapp):
    window, _ = _make_window()
    window.figure_model.panels[0] = Panel3D()
    window._refresh_active_panel_context()

    assert window._zoom_out_action.isEnabled() is False
    window.close()


def test_zoom_out_is_a_no_op_for_a_3d_panel(qapp):
    window, _ = _make_window()
    window.figure_model.panels[0] = Panel3D()
    window._refresh_active_panel_context()
    ax = _active_ax(window)
    before = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    window._on_zoom_out()  # bypass the disabled button -- must still be inert

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) == before
    window.close()


def test_zoom_out_action_re_enables_when_a_2d_panel_becomes_active_again(qapp):
    window, _ = _make_window(layout_index=4)
    window.figure_model.panels[1] = Panel3D()
    window._set_active_panel(1)
    assert window._zoom_out_action.isEnabled() is False

    window._set_active_panel(0)
    assert window._zoom_out_action.isEnabled() is True
    window.close()
