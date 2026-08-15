import pytest

from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.plotting.figure import GnoviFigure


# --- Grid (single authoritative location -- see the module's
# `_FIGURE_GRID_FIELDS` docstring note) --------------------------------------


def test_grid_on_off_and_which_are_per_panel(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.grid_check.setChecked(True)
    panel.grid_which_combo.setCurrentIndex(panel.grid_which_combo.findData("both"))

    assert figure.active_panel.grid is True
    assert figure.active_panel.grid_which == "both"


def test_grid_style_combo_updates_figure_linestyle(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.grid_style_combo.setCurrentIndex(panel.grid_style_combo.findData(":"))

    assert figure.grid_linestyle == ":"


def test_grid_width_and_alpha_spins_update_figure(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.grid_width_spin.setValue(2.5)
    panel.grid_alpha_spin.setValue(0.25)

    assert figure.grid_linewidth == pytest.approx(2.5)
    assert figure.grid_alpha == pytest.approx(0.25)


def test_grid_color_starts_disabled_and_unset(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    assert figure.grid_color is None
    assert panel.grid_custom_color_check.isChecked() is False
    assert panel.grid_color_button.isEnabled() is False


def test_enabling_custom_grid_color_sets_a_default_and_enables_the_button(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.grid_custom_color_check.setChecked(True)

    assert figure.grid_color is not None
    assert panel.grid_color_button.isEnabled() is True


def test_disabling_custom_grid_color_clears_it_back_to_theme_default(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)
    panel.grid_custom_color_check.setChecked(True)

    panel.grid_custom_color_check.setChecked(False)

    assert figure.grid_color is None
    assert panel.grid_color_button.isEnabled() is False


# --- Ticks: major/minor length and width ------------------------------------


def test_major_tick_length_and_width_spins_update_the_active_panel(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.major_tick_length_spin.setValue(6.0)
    panel.major_tick_width_spin.setValue(1.5)

    assert figure.active_panel.major_tick_length == pytest.approx(6.0)
    assert figure.active_panel.major_tick_width == pytest.approx(1.5)


def test_minor_tick_length_and_width_spins_update_the_active_panel(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.minor_tick_length_spin.setValue(1.0)
    panel.minor_tick_width_spin.setValue(0.3)

    assert figure.active_panel.minor_tick_length == pytest.approx(1.0)
    assert figure.active_panel.minor_tick_width == pytest.approx(0.3)


# --- Legend: Outside Right / Outside Bottom ---------------------------------


def test_legend_location_combo_offers_outside_right_and_outside_bottom(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    items = [panel.legend_loc_combo.itemText(i) for i in range(panel.legend_loc_combo.count())]

    assert "outside right" in items
    assert "outside bottom" in items


def test_selecting_outside_right_sets_panel_legend_loc(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.legend_loc_combo.setCurrentText("outside right")

    assert figure.active_panel.legend_loc == "outside right"


def test_selecting_outside_bottom_sets_panel_legend_loc(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    panel.legend_loc_combo.setCurrentText("outside bottom")

    assert figure.active_panel.legend_loc == "outside bottom"


# --- Apply / Cancel / Reset (capture_state / restore_state / reset_to_defaults) -


def test_capture_and_restore_state_round_trips_grid_fields(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)
    snapshot = panel.capture_state()

    panel.title_edit.setText("Changed")
    panel._apply_title()
    panel.grid_width_spin.setValue(4.0)
    assert figure.grid_linewidth == pytest.approx(4.0)

    panel.restore_state(snapshot)

    assert figure.active_panel.title == ""
    assert figure.grid_linewidth == pytest.approx(snapshot[2]["grid_linewidth"])


def test_reset_to_defaults_resets_panel_and_figure_wide_grid_fields(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)
    panel.title_edit.setText("Changed")
    panel._apply_title()
    panel.grid_alpha_spin.setValue(0.05)

    panel.reset_all_button.click()

    defaults = GnoviFigure()
    assert figure.active_panel.title == ""
    assert figure.grid_alpha == pytest.approx(defaults.grid_alpha)
