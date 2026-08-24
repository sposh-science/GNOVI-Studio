import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.widgets.figure_properties_panel import FigurePropertiesPanel
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D
from gnovi_plot.plotting.series3d import Series3D


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


def test_reset_to_defaults_preserves_the_active_panels_id(qapp):
    """Reset Panel to Defaults must never reassign `Panel.id` -- doing so
    would silently orphan that panel's own analysis-result history for a
    completely unrelated "reset formatting" action."""
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)
    original_id = figure.active_panel.id
    panel.title_edit.setText("Changed")
    panel._apply_title()

    panel.reset_all_button.click()

    assert figure.active_panel.id == original_id


def test_capture_and_restore_state_preserves_the_panels_id(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)
    original_id = figure.active_panel.id
    snapshot = panel.capture_state()

    panel.title_edit.setText("Changed")
    panel._apply_title()
    panel.restore_state(snapshot)

    assert figure.active_panel.id == original_id


# --- Adaptive 3D page (Panel3D) ----------------------------------------------


def _make_3d_figure():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0]})
    dataset = Dataset(name="ds", dataframe=df)
    panel3d = Panel3D()
    panel3d.add_series(Series3D(dataset=dataset, x_column="x", y_column="y", z_column="z"))
    return GnoviFigure(panels=[panel3d])


def test_a_panel3d_active_panel_shows_the_3d_stack_page(qapp):
    figure = _make_3d_figure()
    panel = FigurePropertiesPanel(figure)

    assert panel._stack.currentWidget() is panel._page_3d


def test_a_2d_panel_active_panel_shows_the_2d_stack_page(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)

    assert panel._stack.currentWidget() is panel._page_2d


def test_3d_title_and_labels_edit_the_active_panel3d(qapp):
    figure = _make_3d_figure()
    panel = FigurePropertiesPanel(figure)

    panel.d3_title_edit.setText("Conductivity")
    panel._apply_3d_title()
    panel.d3_zlabel_edit.setText("Z axis")
    panel._apply_3d_zlabel()

    assert figure.active_panel.title == "Conductivity"
    assert figure.active_panel.z_label == "Z axis"


def test_3d_elevation_and_azimuth_spins_update_the_panel(qapp):
    figure = _make_3d_figure()
    panel = FigurePropertiesPanel(figure)

    panel.d3_elevation_spin.setValue(12.0)
    panel.d3_azimuth_spin.setValue(200.0)

    assert figure.active_panel.elevation == pytest.approx(12.0)
    assert figure.active_panel.azimuth == pytest.approx(200.0)


def test_reset_view_button_restores_camera_defaults(qapp):
    figure = _make_3d_figure()
    figure.active_panel.elevation = 5.0
    figure.active_panel.azimuth = 5.0
    panel = FigurePropertiesPanel(figure)
    panel.refresh()

    panel.d3_reset_view_button.click()

    assert figure.active_panel.elevation == Panel3D().elevation
    assert figure.active_panel.azimuth == Panel3D().azimuth


def test_set_current_view_button_emits_a_signal_rather_than_mutating_directly(qapp):
    """`FigurePropertiesPanel` never reads the live canvas itself (see the
    class's own docstring on `set_current_view_requested`) -- clicking "Set
    Current View" must only emit the signal, leaving elevation/azimuth
    untouched until the owner (`MainWindow`) commits a value from the live
    Axes3D."""
    figure = _make_3d_figure()
    panel = FigurePropertiesPanel(figure)
    received = []
    panel.set_current_view_requested.connect(lambda: received.append(True))
    before_elev = figure.active_panel.elevation

    panel.d3_set_current_view_button.click()

    assert received == [True]
    assert figure.active_panel.elevation == before_elev


def test_3d_manual_limits_round_trip(qapp):
    figure = _make_3d_figure()
    panel = FigurePropertiesPanel(figure)

    panel.d3_x_manual_check.setChecked(True)
    panel.d3_x_min_spin.setValue(0.0)
    panel.d3_x_max_spin.setValue(10.0)

    assert figure.active_panel.xlim == (0.0, 10.0)


def test_3d_reset_limits_clears_xyz_limits(qapp):
    figure = _make_3d_figure()
    figure.active_panel.xlim = (0.0, 1.0)
    figure.active_panel.ylim = (0.0, 1.0)
    figure.active_panel.zlim = (0.0, 1.0)
    panel = FigurePropertiesPanel(figure)
    panel.refresh()

    panel.d3_reset_limits_button.click()

    assert figure.active_panel.xlim is None
    assert figure.active_panel.ylim is None
    assert figure.active_panel.zlim is None


def test_switching_from_2d_to_3d_panel_swaps_the_stack_page_on_refresh(qapp):
    figure = GnoviFigure()
    panel = FigurePropertiesPanel(figure)
    assert panel._stack.currentWidget() is panel._page_2d

    figure.panels.append(_make_3d_figure().panels[0])
    figure.set_active_panel(1)
    panel.refresh()

    assert panel._stack.currentWidget() is panel._page_3d
