import pytest

from gnovi_plot.gui.styles import PlotTheme
from gnovi_plot.gui.widgets.figure_size_panel import FigureSizePanel
from gnovi_plot.plotting.figure import GnoviFigure


def test_default_unit_is_inches_and_matches_figure_defaults(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)

    assert panel.width_spin.value() == pytest.approx(figure.figure_width_in)
    assert panel.height_spin.value() == pytest.approx(figure.figure_height_in)


def test_aspect_ratio_preset_recomputes_height_from_current_width(qapp):
    figure = GnoviFigure()
    figure.figure_width_in = 8.0
    panel = FigureSizePanel(figure)

    panel.aspect_combo.setCurrentText("4:3")

    assert figure.lock_aspect_ratio is True
    assert figure.figure_height_in == pytest.approx(6.0)


def test_1_to_1_preset_makes_width_equal_height(qapp):
    figure = GnoviFigure()
    figure.figure_width_in = 5.0
    panel = FigureSizePanel(figure)

    panel.aspect_combo.setCurrentText("1:1")

    assert figure.figure_height_in == pytest.approx(5.0)


def test_lock_check_label_is_unambiguously_figure_scoped(qapp):
    """Distinct wording from the Layout page's "Panel Aspect Ratio" so the
    two independent controls are never confused (see
    `gui.widgets.figure_layout_panel`)."""
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)

    assert panel.lock_check.text() == "Lock Figure aspect ratio"


def test_aspect_combo_tooltip_identifies_it_as_figure_scoped(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)

    tooltip = panel.aspect_combo.toolTip()
    assert "Figure Aspect Ratio" in tooltip
    assert "complete figure" in tooltip.lower() or "page" in tooltip.lower()


def test_auto_fit_workspace_unlocks_aspect_ratio(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    panel.aspect_combo.setCurrentText("4:3")
    assert figure.lock_aspect_ratio is True

    panel.aspect_combo.setCurrentText("Auto / Fit workspace")
    assert figure.lock_aspect_ratio is False


def test_unit_conversion_mm_matches_stored_inches(qapp):
    figure = GnoviFigure()
    figure.figure_width_in = 1.0
    figure.figure_height_in = 1.0
    panel = FigureSizePanel(figure)

    panel.unit_combo.setCurrentText("mm")

    assert panel.width_spin.value() == pytest.approx(25.4, abs=0.01)
    assert panel.height_spin.value() == pytest.approx(25.4, abs=0.01)


def test_unit_conversion_cm_matches_stored_inches(qapp):
    figure = GnoviFigure()
    figure.figure_width_in = 2.0
    panel = FigureSizePanel(figure)

    panel.unit_combo.setCurrentText("cm")

    assert panel.width_spin.value() == pytest.approx(5.08, abs=0.01)


def test_editing_width_in_mm_updates_figure_inches(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    panel.unit_combo.setCurrentText("mm")

    panel.width_spin.setValue(50.8)  # 2 inches

    assert figure.figure_width_in == pytest.approx(2.0, abs=0.001)


def test_locked_aspect_ratio_updates_height_when_width_changes(qapp):
    figure = GnoviFigure()
    figure.figure_width_in = 4.0
    figure.figure_height_in = 2.0
    panel = FigureSizePanel(figure)
    panel.lock_check.setChecked(True)

    panel.width_spin.setValue(8.0)

    assert figure.figure_height_in == pytest.approx(4.0)


def test_locked_aspect_ratio_updates_width_when_height_changes(qapp):
    figure = GnoviFigure()
    figure.figure_width_in = 4.0
    figure.figure_height_in = 2.0
    panel = FigureSizePanel(figure)
    panel.lock_check.setChecked(True)

    panel.height_spin.setValue(1.0)

    assert figure.figure_width_in == pytest.approx(2.0)


def test_unlocked_aspect_ratio_leaves_height_untouched_when_width_changes(qapp):
    figure = GnoviFigure()
    figure.figure_width_in = 4.0
    figure.figure_height_in = 2.0
    panel = FigureSizePanel(figure)
    assert panel.lock_check.isChecked() is False

    panel.width_spin.setValue(10.0)

    assert figure.figure_height_in == pytest.approx(2.0)


def test_publication_preset_sets_width_and_height_in_mm(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)

    panel.publication_combo.setCurrentText("Journal single column")

    assert figure.figure_width_in == pytest.approx(85.0 / 25.4)
    assert panel.unit_combo.currentText() == "mm"


def test_layout_preset_changes_figure_layout_and_panel_options(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)

    panel.layout_combo.setCurrentIndex(3)  # "2 x 2"

    assert figure.layout == (2, 2)
    assert panel.panel_combo.count() == 4


def test_panel_combo_switches_active_panel(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    panel.layout_combo.setCurrentIndex(1)  # "1 x 2"

    panel.panel_combo.setCurrentIndex(1)

    assert figure.active_panel_index == 1


def test_panel_labels_checkbox_toggles_figure_flag(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    assert figure.panel_labels_visible is False

    panel.panel_labels_check.setChecked(True)

    assert figure.panel_labels_visible is True


# --- Plot Theme -----------------------------------------------------------------


def test_theme_combo_defaults_to_light_and_emits_the_selected_mode(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    assert panel.theme_combo.currentText() == "Light"

    received = []
    panel.theme_change_requested.connect(received.append)
    panel.theme_combo.setCurrentIndex(panel.theme_combo.findText("Dark"))

    assert received == [PlotTheme.DARK]


def test_set_current_theme_updates_the_combo_without_emitting(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    received = []
    panel.theme_change_requested.connect(received.append)

    panel.set_current_theme(PlotTheme.DARK)

    assert panel.theme_combo.currentText() == "Dark"
    assert received == []


# --- Apply / Cancel / Reset (capture_state / restore_state / reset_to_defaults) -


def test_capture_and_restore_state_round_trips_figure_scalar_fields(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    snapshot = panel.capture_state()

    panel.width_spin.setValue(12.0)
    assert figure.figure_width_in != snapshot["figure_width_in"]

    panel.restore_state(snapshot)

    assert figure.figure_width_in == pytest.approx(snapshot["figure_width_in"])


def test_restore_state_does_not_touch_panel_layout(qapp):
    """Layout/panels are deliberately out of scope for Cancel -- see
    `_FIGURE_SCALAR_FIELDS`'s comment."""
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    snapshot = panel.capture_state()

    panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    assert figure.layout == (2, 2)

    panel.restore_state(snapshot)

    assert figure.layout == (2, 2)  # untouched by restore


def test_reset_to_defaults_restores_a_fresh_gnovifigures_values(qapp):
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)
    panel.width_spin.setValue(20.0)

    panel.reset_to_defaults()

    defaults = GnoviFigure()
    assert figure.figure_width_in == pytest.approx(defaults.figure_width_in)


def test_refresh_reloads_widgets_from_an_externally_mutated_figure(qapp):
    """Covers Undo/Redo restoring a snapshot onto the live figure in place
    -- the panel must pick up the new values on `refresh()`."""
    figure = GnoviFigure()
    panel = FigureSizePanel(figure)

    figure.figure_width_in = 11.0
    panel.refresh()

    assert panel.width_spin.value() == pytest.approx(11.0)
