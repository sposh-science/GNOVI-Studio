import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets.figure_layout_panel import FigureLayoutPanel
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def test_default_values_match_figure_defaults(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    assert panel.left_spin.value() == pytest.approx(figure.margin_left)
    assert panel.right_spin.value() == pytest.approx(figure.margin_right)
    assert panel.bottom_spin.value() == pytest.approx(figure.margin_bottom)
    assert panel.top_spin.value() == pytest.approx(figure.margin_top)
    assert panel.wspace_spin.value() == pytest.approx(figure.panel_wspace)
    assert panel.hspace_spin.value() == pytest.approx(figure.panel_hspace)


def test_editing_left_margin_updates_the_figure(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.left_spin.setValue(0.2)

    assert figure.margin_left == pytest.approx(0.2)


def test_editing_right_margin_updates_the_figure(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.right_spin.setValue(0.75)

    assert figure.margin_right == pytest.approx(0.75)


def test_editing_top_margin_updates_the_figure(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.top_spin.setValue(0.8)

    assert figure.margin_top == pytest.approx(0.8)


def test_editing_bottom_margin_updates_the_figure(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.bottom_spin.setValue(0.2)

    assert figure.margin_bottom == pytest.approx(0.2)


def test_editing_horizontal_spacing_updates_the_figure(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.wspace_spin.setValue(0.6)

    assert figure.panel_wspace == pytest.approx(0.6)


def test_editing_vertical_spacing_updates_the_figure(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.hspace_spin.setValue(0.7)

    assert figure.panel_hspace == pytest.approx(0.7)


def test_changed_signal_emitted_on_margin_edit(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)
    received = []
    panel.changed.connect(lambda: received.append(True))

    panel.left_spin.setValue(0.3)

    assert received == [True]


def test_changed_signal_emitted_on_spacing_edit(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)
    received = []
    panel.changed.connect(lambda: received.append(True))

    panel.hspace_spin.setValue(0.9)

    assert received == [True]


def test_left_margin_cannot_be_dragged_past_right_margin(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.right_spin.setValue(0.3)
    panel.left_spin.setValue(0.9)  # would otherwise exceed the right margin

    assert figure.margin_left < figure.margin_right


def test_bottom_margin_cannot_be_dragged_past_top_margin(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    panel.top_spin.setValue(0.3)
    panel.bottom_spin.setValue(0.9)  # would otherwise exceed the top margin

    assert figure.margin_bottom < figure.margin_top


def test_tight_layout_bakes_computed_values_into_the_figure_and_widgets(qapp):
    figure = GnoviFigure()
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    dataset = Dataset(name="d", dataframe=df)
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    figure.xlabel = "X axis"
    figure.ylabel = "Y axis"
    figure.title = "Title"
    panel = FigureLayoutPanel(figure)

    panel.tight_layout_button.click()

    # A smoke check that the computed value was actually applied and the
    # widgets picked it up, not that it landed on any specific number.
    # Tolerances match each spin box's own display precision (3/2 decimals
    # -- see `_make_margin_spin`/`_make_spacing_spin`): the stored figure
    # value keeps full float precision, the widget rounds for display.
    assert panel.left_spin.value() == pytest.approx(figure.margin_left, abs=0.001)
    assert panel.wspace_spin.value() == pytest.approx(figure.panel_wspace, abs=0.01)


def test_tight_layout_is_not_applied_automatically_on_construction(qapp):
    """Tight Layout is a one-off action, never a persistent mode -- a fresh
    panel must show the plain Matplotlib rc defaults, not a recomputed
    "tight" layout."""
    figure = GnoviFigure()
    FigureLayoutPanel(figure)

    assert figure.margin_left == pytest.approx(0.125)
    assert figure.margin_right == pytest.approx(0.9)


def test_reset_to_defaults_restores_rc_default_margins_and_spacing(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)
    panel.left_spin.setValue(0.05)
    panel.wspace_spin.setValue(1.0)

    panel.reset_button.click()

    defaults = GnoviFigure()
    assert figure.margin_left == pytest.approx(defaults.margin_left)
    assert figure.margin_right == pytest.approx(defaults.margin_right)
    assert figure.panel_wspace == pytest.approx(defaults.panel_wspace)
    assert figure.panel_hspace == pytest.approx(defaults.panel_hspace)


def test_reset_to_defaults_does_not_touch_panel_layout_or_series(qapp):
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    dataset = Dataset(name="d", dataframe=df)
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    panel = FigureLayoutPanel(figure)

    panel.reset_button.click()

    assert figure.layout == (2, 2)
    assert len(figure.series) == 1


def test_refresh_reloads_widgets_from_an_externally_mutated_figure(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    figure.margin_left = 0.3
    panel.refresh()

    assert panel.left_spin.value() == pytest.approx(0.3)


def test_capture_and_restore_state_round_trips_layout_fields(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)
    snapshot = panel.capture_state()

    panel.left_spin.setValue(0.3)
    assert figure.margin_left != pytest.approx(snapshot["margin_left"])

    panel.restore_state(snapshot)

    assert figure.margin_left == pytest.approx(snapshot["margin_left"])


def test_tight_layout_does_not_reset_panel_aspect(qapp):
    """Tight Layout only bakes margin/spacing values -- Panel Aspect Ratio
    is now owned entirely by the Figure page (see
    `tests.test_figure_size_panel`) and must never be touched here."""
    figure = GnoviFigure()
    figure.panel_aspect_preset = "1:1"
    panel = FigureLayoutPanel(figure)

    panel.tight_layout_button.click()

    assert figure.panel_aspect_preset == "1:1"


def test_reset_to_defaults_does_not_touch_panel_aspect(qapp):
    """See `test_tight_layout_does_not_reset_panel_aspect` -- Reset here is
    scoped to margins/spacing only now."""
    figure = GnoviFigure()
    figure.panel_aspect_preset = "3:2"
    panel = FigureLayoutPanel(figure)

    panel.reset_button.click()

    assert figure.panel_aspect_preset == "3:2"


def test_panel_aspect_combo_no_longer_lives_on_the_layout_page(qapp):
    figure = GnoviFigure()
    panel = FigureLayoutPanel(figure)

    assert not hasattr(panel, "panel_aspect_combo")
