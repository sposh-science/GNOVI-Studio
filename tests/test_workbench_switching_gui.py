"""GUI retargeting when switching the active Workbench, driven through the
real `MainWindow` -- the core architectural guarantee of this phase: no
widget may ever hold a stale reference to a previous Workbench's
`GnoviFigure` (see `MainWindow._activate_workbench`).
"""

import pandas as pd
from PySide6.QtWidgets import QInputDialog

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.figure import PlotTheme
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


def _new_workbench(window) -> str:
    """Create a Workbench via the real "+" flow and return its id."""
    window.workbench_tab_bar.new_button.click()
    return window._project.active_workbench_id


# --- Core switching: figure_model always points at the active Workbench -------


def test_switching_retargets_figure_model(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    workbench_b_id = _new_workbench(window)
    assert window.figure_model is window._project.get_workbench(workbench_b_id).figure

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.figure_model is window._project.get_workbench(workbench_a_id).figure
    window.close()


def test_switching_retargets_the_four_figure_dependent_widgets(qapp):
    """series_panel/properties_panel/figure_size_panel/figure_layout_panel
    each hold a direct `GnoviFigure` reference from construction -- must be
    explicitly `set_figure()`-retargeted on every switch."""
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    workbench_b_id = _new_workbench(window)

    window._on_workbench_tab_selected(workbench_a_id)
    figure_a = window._project.get_workbench(workbench_a_id).figure
    assert window.series_panel._figure is figure_a
    assert window.properties_panel._figure is figure_a
    assert window.figure_size_panel._figure is figure_a
    assert window.figure_layout_panel._figure is figure_a

    window._on_workbench_tab_selected(workbench_b_id)
    figure_b = window._project.get_workbench(workbench_b_id).figure
    assert window.series_panel._figure is figure_b
    assert window.properties_panel._figure is figure_b
    assert window.figure_size_panel._figure is figure_b
    assert window.figure_layout_panel._figure is figure_b
    window.close()


# --- Layout / active panel -----------------------------------------------------


def test_switching_retargets_layout_and_active_panel(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    window.toolbar_panel_combo.setCurrentIndex(2)  # Panel 3
    assert window.figure_model.layout == (2, 2)
    assert window.figure_model.active_panel_index == 2

    workbench_b_id = _new_workbench(window)
    assert window.figure_model.layout == (1, 1)
    assert window.figure_model.active_panel_index == 0

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.figure_model.layout == (2, 2)
    assert window.figure_model.active_panel_index == 2
    assert window.toolbar_layout_combo.currentIndex() == 3
    assert window.toolbar_panel_combo.currentIndex() == 2
    window.close()
    del workbench_b_id


# --- Plot Theme -----------------------------------------------------------------


def test_switching_retargets_plot_theme_and_its_controls(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    window._on_theme_changed(PlotTheme.DARK)
    assert window.figure_model.plot_theme == PlotTheme.DARK

    workbench_b_id = _new_workbench(window)
    assert window.figure_model.plot_theme == PlotTheme.LIGHT
    assert window._theme_actions[PlotTheme.LIGHT].isChecked()

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.figure_model.plot_theme == PlotTheme.DARK
    assert window._theme_actions[PlotTheme.DARK].isChecked()
    assert window.toolbar_theme_combo.itemData(window.toolbar_theme_combo.currentIndex()) == PlotTheme.DARK
    assert window.figure_size_panel.theme_combo.currentText() == "Dark"
    window.close()
    del workbench_b_id


# --- Figure Aspect Ratio / Panel Aspect Ratio -----------------------------------


def test_switching_retargets_figure_and_panel_aspect_ratio(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    window.figure_size_panel.aspect_combo.setCurrentText("16:9")
    window.figure_size_panel.panel_aspect_combo.setCurrentText("1:1")
    assert window.figure_model.aspect_preset == "16:9"
    assert window.figure_model.panel_aspect_preset == "1:1"

    workbench_b_id = _new_workbench(window)
    assert window.figure_model.panel_aspect_preset == "Auto"

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.figure_model.aspect_preset == "16:9"
    assert window.figure_model.panel_aspect_preset == "1:1"
    assert window.figure_size_panel.aspect_combo.currentText() == "16:9"
    assert window.figure_size_panel.panel_aspect_combo.currentText() == "1:1"
    window.close()
    del workbench_b_id


# --- Series / datasets -----------------------------------------------------------


def test_switching_retargets_series_and_dataset_context(qapp):
    window = MainWindow()
    workbench_a_id = window._project.active_workbench_id
    dataset = _make_dataset("Ferricyanide SR-0.05")
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    assert len(window.figure_model.series) == 1
    assert window.series_panel.series_list.count() == 1

    workbench_b_id = _new_workbench(window)
    assert window.figure_model.series == []
    assert window.series_panel.series_list.count() == 0

    window._on_workbench_tab_selected(workbench_a_id)

    assert len(window.figure_model.series) == 1
    assert window.series_panel.series_list.count() == 1
    assert window.figure_model.series[0].dataset is dataset  # still the shared live Dataset
    window.close()
    del workbench_b_id


# --- Active panel / Graph / Data context, Workbench header ---------------------


def test_switching_retargets_active_panel_context_and_workbench_header(qapp, monkeypatch):
    window = MainWindow()
    dataset = _make_dataset("Ferricyanide SR-0.05")
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()
    workbench_a_id = window._project.active_workbench_id
    window._project.rename_workbench(workbench_a_id, "CV Comparison")
    window._refresh_active_panel_context()

    workbench_b_id = _new_workbench(window)
    assert window.workbench_header.label.text() == "WORKBENCH · Workbench 2 · 1 × 1"
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 1\nGraph: Unsaved graph\nData: No data"

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.workbench_header.label.text() == "WORKBENCH · CV Comparison · 1 × 1"
    assert window.plot_page_active_panel_label.text() == (
        "Active panel: Panel 1\nGraph: G1 (working copy)\nData: Ferricyanide SR-0.05"
    )
    window.close()
    del workbench_b_id


def test_graph_library_load_targets_the_active_workbenchs_active_panel(qapp, monkeypatch):
    """"Load into Active Panel" always means Active Workbench + Active
    Panel -- loading while Workbench B is active must never touch A."""
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()
    workbench_a_id = window._project.active_workbench_id

    workbench_b_id = _new_workbench(window)
    window.graph_library_panel.graph_list.setCurrentRow(0)
    window.graph_library_panel.load_button.click()

    assert len(window.figure_model.series) == 1  # loaded into B's active panel
    assert window._project.get_workbench(workbench_a_id).figure.series[0].label != ""  # A untouched
    assert len(window._project.get_workbench(workbench_a_id).figure.series) == 1
    window.close()
    del workbench_b_id


# --- Update Saved Graph button state follows the active Workbench --------------


def test_update_saved_graph_button_state_follows_the_active_workbench(qapp, monkeypatch):
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()
    workbench_a_id = window._project.active_workbench_id
    assert window.graph_library_panel.update_button.isEnabled() is True

    workbench_b_id = _new_workbench(window)
    assert window.graph_library_panel.update_button.isEnabled() is False

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.graph_library_panel.update_button.isEnabled() is True
    window.close()
    del workbench_b_id
