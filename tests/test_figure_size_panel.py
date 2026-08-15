import pytest

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
