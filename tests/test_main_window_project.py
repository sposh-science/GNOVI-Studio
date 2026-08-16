import pandas as pd
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from gnovi_plot.core.app_info import APP_NAME
from gnovi_plot.core.project_io import load_project
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.figure import PlotTheme
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return Dataset(name=name, dataframe=df)


def _make_window(qapp):
    return MainWindow()


def _dirty_window(qapp):
    """A MainWindow with one dataset imported and one series plotted --
    a real, unsaved content edit (as opposed to pure navigation)."""
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.dataset_panel.datasets_changed.emit()
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    assert window._dirty is True
    return window


# --- File menu: New/Open/Save/Save As -----------------------------------------


def test_file_menu_has_new_open_save_save_as_actions(qapp):
    window = _make_window(qapp)
    assert window.new_project_action.text() == "New Project"
    assert window.open_project_action.text() == "Open Project…"
    assert window.save_project_action.text() == "Save Project"
    assert window.save_project_as_action.text() == "Save Project As…"
    window.close()


def test_new_project_action_resets_to_an_empty_untitled_project(qapp):
    window = _dirty_window(qapp)

    window.new_project_action.trigger()

    assert window._project.path is None
    assert window._project.name == "Untitled Project"
    assert len(window.dataset_manager) == 0
    assert window.figure_model.series == []
    assert window._dirty is False
    window.close()


def test_save_project_as_then_open_project_round_trips_through_the_real_gui(qapp, monkeypatch, tmp_path):
    window = _dirty_window(qapp)
    out_path = tmp_path / "my_project.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))

    window.save_project_as_action.trigger()

    assert out_path.exists()
    assert window._project.path == out_path
    assert window._project.name == "my_project"
    assert window._dirty is False

    # Reopen through the real File > Open Project action.
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert len(window.dataset_manager) == 1
    assert len(window.figure_model.series) == 1
    assert window._dirty is False
    window.close()


def test_dark_theme_project_reopens_dark_and_syncs_menu_toolbar_and_figure_size_panel(
    qapp, monkeypatch, tmp_path
):
    """The architecture correction this test guards: Plot Theme is
    declarative GnoviFigure state, so a project saved Dark must come back
    Dark on reopen -- not silently reset to the QSettings/light default --
    and every control that displays the current theme (View menu, toolbar
    combo, Figure panel's Theme combo) must reflect it too."""
    window = _dirty_window(qapp)
    window._on_theme_changed(PlotTheme.DARK)
    assert window.figure_model.plot_theme == PlotTheme.DARK

    out_path = tmp_path / "dark_project.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    # New Project resets to the ordinary (Light) default first, so reopening
    # below is a genuine restore, not a no-op continuation of state.
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))
    window.new_project_action.trigger()
    assert window.figure_model.plot_theme == PlotTheme.LIGHT

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert window.figure_model.plot_theme == PlotTheme.DARK
    assert window._theme_actions[PlotTheme.DARK].isChecked()
    assert not window._theme_actions[PlotTheme.LIGHT].isChecked()
    assert window.toolbar_theme_combo.itemData(window.toolbar_theme_combo.currentIndex()) == PlotTheme.DARK
    window.close()


def test_export_dialog_after_reopening_a_dark_project_shows_the_dark_figure_as_is(
    qapp, monkeypatch, tmp_path
):
    from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog

    window = _dirty_window(qapp)
    window._on_theme_changed(PlotTheme.DARK)
    out_path = tmp_path / "dark_for_export.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))
    window.new_project_action.trigger()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dialog.background_combo.currentText() == "As shown"
    assert window.plot_canvas.figure.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
    window.close()


def test_save_project_writes_to_the_existing_path_without_prompting(qapp, monkeypatch, tmp_path):
    window = _dirty_window(qapp)
    out_path = tmp_path / "existing.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()
    assert window._dirty is False

    # Dirty it again, then plain "Save Project" -- must not re-prompt.
    window._on_add_to_plot([PlotSeries.line(window.dataset_manager.datasets[0], "x", "y")])
    assert window._dirty is True

    def _fail_if_called(*a, **k):
        raise AssertionError("Save Project must not prompt when a path is already known")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_fail_if_called))
    window.save_project_action.trigger()

    assert window._dirty is False
    reloaded = load_project(out_path)
    assert len(reloaded.workbenches[0].figure.series) == 2
    window.close()


def test_open_project_with_malformed_file_shows_an_error_and_keeps_current_project(qapp, monkeypatch, tmp_path):
    window = _dirty_window(qapp)
    original_manager = window.dataset_manager
    bad_path = tmp_path / "bad.gnovi"
    bad_path.write_bytes(b"not a zip")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad_path), "")))
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: critical_calls.append(a)))
    # Allow the unsaved-changes prompt itself to proceed as Discard.
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))

    window.open_project_action.trigger()

    assert len(critical_calls) == 1
    assert window.dataset_manager is original_manager  # untouched by the failed load
    window.close()


# --- Graph Library tab, reached through MainWindow ----------------------------


def test_graphs_tab_exists_and_hosts_the_graph_library_panel(qapp):
    window = _make_window(qapp)
    tab_texts = [window.bottom_panel.tabText(i) for i in range(window.bottom_panel.count())]
    assert "Graphs" in tab_texts
    assert window.graph_library_panel.parent() is window.bottom_panel._graphs_tab
    window.close()


# --- Plot page dataset selector, through MainWindow ----------------------------


def test_dataset_selector_follows_data_page_selection(qapp):
    """The Data and Plot pages are two views over the same DatasetPanel
    instance (see MainWindow.__init__) -- selecting a dataset on the Data
    page must be reflected by the Plot page's combo immediately."""
    window = _make_window(qapp)
    dataset = _make_dataset("S1 (Ferricyanide) SR-0.05")
    window.dataset_manager.add(dataset)

    window.dataset_panel._refresh_list(select_id=dataset.id)

    assert window.dataset_panel.active_dataset_combo.currentData() == dataset.id
    window.close()


def test_dataset_selector_drives_data_page_selection(qapp):
    window = _make_window(qapp)
    d1 = _make_dataset("First")
    d2 = _make_dataset("Second")
    window.dataset_manager.add(d1)
    window.dataset_manager.add(d2)
    window.dataset_panel._refresh_list(select_id=d1.id)

    index = window.dataset_panel.active_dataset_combo.findData(d2.id)
    window.dataset_panel.active_dataset_combo.setCurrentIndex(index)

    assert window.dataset_panel.current_dataset is d2
    window.close()


def test_dataset_selector_resets_on_new_project(qapp):
    window = _dirty_window(qapp)
    dataset = window.dataset_manager.datasets[0]
    window.dataset_panel._refresh_list(select_id=dataset.id)
    assert window.dataset_panel.active_dataset_combo.currentData() == dataset.id

    window.new_project_action.trigger()  # dirty -> discard, via the autouse fixture

    assert window.dataset_panel.active_dataset_combo.count() == 1
    assert window.dataset_panel.active_dataset_combo.currentText() == "(no dataset)"
    window.close()


def test_dataset_selector_resyncs_after_project_reopen(qapp, monkeypatch, tmp_path):
    """A reopened project starts with nothing selected (same as import) --
    the combo must not keep showing the previous project's dataset, and
    must correctly track a fresh selection made in the reopened project."""
    window = _dirty_window(qapp)
    window.dataset_panel._refresh_list(select_id=window.dataset_manager.datasets[0].id)
    out_path = tmp_path / "selector_project.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert window.dataset_panel.active_dataset_combo.currentText() == "(no dataset)"

    reopened_dataset = window.dataset_manager.datasets[0]
    window.dataset_panel._refresh_list(select_id=reopened_dataset.id)
    assert window.dataset_panel.active_dataset_combo.currentData() == reopened_dataset.id
    window.close()


# --- Active panel context label, through MainWindow -----------------------------


def _no_context(panel_number: int) -> str:
    """The 3-line context text for a fresh, empty, never-saved panel --
    see `ActivePanelLabel.refresh`."""
    return f"Active panel: Panel {panel_number}\nGraph: Unsaved graph\nData: No data"


def test_active_panel_labels_start_at_panel_1_on_every_page(qapp):
    window = _make_window(qapp)
    assert window.plot_page_active_panel_label.text() == _no_context(1)
    assert window.series_panel.active_panel_label.text() == _no_context(1)
    assert window.figure_size_panel.active_panel_label.text() == _no_context(1)
    assert window.figure_layout_panel.active_panel_label.text() == _no_context(1)
    assert window.properties_panel.active_panel_label.text() == _no_context(1)
    window.close()


def test_active_panel_labels_update_after_clicking_a_different_panel(qapp):
    window = _make_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    ax = window.plot_canvas.axes_list[1]
    event = MouseEvent("button_press_event", window.plot_canvas, 1, 1)
    event.inaxes = ax

    window._on_canvas_click(event)

    assert window.figure_model.active_panel_index == 1
    assert window.plot_page_active_panel_label.text() == _no_context(2)
    assert window.series_panel.active_panel_label.text() == _no_context(2)
    assert window.figure_size_panel.active_panel_label.text() == _no_context(2)
    assert window.figure_layout_panel.active_panel_label.text() == _no_context(2)
    assert window.properties_panel.active_panel_label.text() == _no_context(2)
    window.close()


def test_active_panel_labels_update_after_the_toolbar_selector_changes(qapp):
    window = _make_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"

    window.toolbar_panel_combo.setCurrentIndex(2)

    assert window.figure_model.active_panel_index == 2
    assert window.plot_page_active_panel_label.text() == _no_context(3)
    assert window.figure_size_panel.active_panel_label.text() == _no_context(3)
    window.close()


def test_active_panel_labels_update_after_a_layout_change_clamps_the_index(qapp):
    """Shrinking the layout while a later panel is active clamps
    `active_panel_index` back into range (see `GnoviFigure.set_layout`) --
    every page's label must reflect the clamped value, not the stale one."""
    window = _make_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    window.toolbar_panel_combo.setCurrentIndex(3)  # Panel 4
    assert window.figure_model.active_panel_index == 3

    window.figure_size_panel.layout_combo.setCurrentIndex(0)  # "1 x 1"

    assert window.figure_model.active_panel_index == 0
    assert window.plot_page_active_panel_label.text() == _no_context(1)
    assert window.series_panel.active_panel_label.text() == _no_context(1)
    assert window.figure_layout_panel.active_panel_label.text() == _no_context(1)
    window.close()


def test_active_panel_labels_update_after_project_load(qapp, monkeypatch, tmp_path):
    window = _make_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    window.toolbar_panel_combo.setCurrentIndex(2)  # Panel 3
    out_path = tmp_path / "panel_context.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    window.new_project_action.trigger()
    assert window.plot_page_active_panel_label.text() == _no_context(1)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert window.figure_model.active_panel_index == 2
    assert window.plot_page_active_panel_label.text() == _no_context(3)
    assert window.series_panel.active_panel_label.text() == _no_context(3)
    assert window.figure_size_panel.active_panel_label.text() == _no_context(3)
    assert window.figure_layout_panel.active_panel_label.text() == _no_context(3)
    assert window.properties_panel.active_panel_label.text() == _no_context(3)
    window.close()


def test_active_panel_label_updates_after_new_project(qapp):
    window = _dirty_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window.toolbar_panel_combo.setCurrentIndex(1)
    assert window.plot_page_active_panel_label.text() == _no_context(2)  # Panel 2 is a fresh blank panel

    window.new_project_action.trigger()

    assert window.plot_page_active_panel_label.text() == _no_context(1)
    window.close()


def test_active_panel_label_updates_when_a_saved_graph_is_loaded_into_another_panel(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()

    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window.toolbar_panel_combo.setCurrentIndex(1)  # Panel 2 -- fresh, blank
    assert window.plot_page_active_panel_label.text() == _no_context(2)

    window.graph_library_panel.graph_list.setCurrentRow(0)
    window.graph_library_panel.load_button.click()

    assert window.plot_page_active_panel_label.text() == (
        "Active panel: Panel 2\nGraph: G1 (working copy)\nData: d"
    )
    window.close()


# --- Graph/Data context and Update Saved Graph, through MainWindow -------------


def test_saving_active_panel_as_graph_marks_it_a_working_copy(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset("Ferricyanide SR-0.05")
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("Ferricyanide 50 mV/s", True))
    )

    window.graph_library_panel.save_button.click()

    assert window.plot_page_active_panel_label.text() == (
        "Active panel: Panel 1\n"
        "Graph: Ferricyanide 50 mV/s (working copy)\n"
        "Data: Ferricyanide SR-0.05"
    )
    window.close()


def test_multiple_dataset_provenance_lists_names_inline(qapp):
    window = _make_window(qapp)
    d1 = _make_dataset("Ferricyanide SR-0.05")
    d2 = _make_dataset("Ascorbic Acid")
    window.dataset_manager.add(d1)
    window.dataset_manager.add(d2)
    window._on_add_to_plot(
        [PlotSeries.line(d1, "x", "y"), PlotSeries.line(d2, "x", "y")]
    )

    assert window.plot_page_active_panel_label.text() == (
        "Active panel: Panel 1\nGraph: Unsaved graph\nData:\n  Ferricyanide SR-0.05\n  Ascorbic Acid"
    )
    assert window.plot_page_active_panel_label.toolTip() == "Ferricyanide SR-0.05\nAscorbic Acid"
    window.close()


def test_update_saved_graph_button_disabled_until_panel_has_an_origin(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    assert window.graph_library_panel.update_button.isEnabled() is False

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()

    assert window.graph_library_panel.update_button.isEnabled() is True
    window.close()


def test_update_saved_graph_replaces_the_stored_snapshot_after_confirmation(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()
    graph_id = window._project.graph_library.graphs[0].id

    window._on_clear_plot()
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label="Updated")])
    window._set_dirty(False)

    def _update_button(self):
        for button in self.buttons():
            if button.text() == "Update":
                return button
        raise AssertionError("Update button not found")

    monkeypatch.setattr(QMessageBox, "clickedButton", _update_button)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)

    window.graph_library_panel.update_button.click()

    stored = window._project.graph_library.get(graph_id)
    assert len(stored.panel.series) == 1
    assert stored.panel.series[0].label == "Updated"
    assert window._dirty is True  # Update Saved Graph marks the project dirty
    window.close()


def test_update_saved_graph_cancel_leaves_the_stored_snapshot_untouched(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label="Original")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()
    graph_id = window._project.graph_library.graphs[0].id

    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label="Extra")])

    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)

    window.graph_library_panel.update_button.click()

    stored = window._project.graph_library.get(graph_id)
    assert len(stored.panel.series) == 1
    assert stored.panel.series[0].label == "Original"
    window.close()


# --- Active panel -> Graph Library selection sync, through MainWindow ----------


def test_panel_g1_selects_g1_in_the_graph_library(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"

    window._set_active_panel(0)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()
    g1_id = window._project.graph_library.graphs[0].id

    window._set_active_panel(1)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g2", True)))
    window.graph_library_panel.save_button.click()

    window._set_active_panel(0)

    assert window.graph_library_panel._current_graph_id() == g1_id
    window.close()


def test_panel_g2_selects_g2_in_the_graph_library(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"

    window._set_active_panel(0)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()

    window._set_active_panel(1)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g2", True)))
    window.graph_library_panel.save_button.click()
    g2_id = window._project.graph_library.graphs[-1].id

    assert window.graph_library_panel._current_graph_id() == g2_id
    window.close()


def test_unsaved_active_panel_clears_the_graph_library_selection(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()
    assert window.graph_library_panel._current_graph_id() is not None

    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2" -- Panel 2 is fresh/unsaved
    window.toolbar_panel_combo.setCurrentIndex(1)

    assert window.graph_library_panel.graph_list.currentRow() == -1
    assert window.graph_library_panel._current_graph_id() is None
    window.close()


def test_switching_workbenches_refreshes_the_graph_library_selection(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()
    g1_id = window._project.graph_library.graphs[0].id
    workbench_a_id = window._project.active_workbench_id

    window.workbench_tab_bar.new_button.click()  # Workbench 2: fresh, unsaved
    assert window.graph_library_panel._current_graph_id() is None

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.graph_library_panel._current_graph_id() == g1_id
    window.close()


def test_loading_a_saved_graph_updates_the_graph_library_selection(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()
    g1_id = window._project.graph_library.graphs[0].id

    window._on_clear_plot()  # active panel is now unsaved again
    window.workbench_tab_bar.new_button.click()
    assert window.graph_library_panel._current_graph_id() is None

    window.graph_library_panel.graph_list.setCurrentRow(0)
    window.graph_library_panel.load_button.click()

    assert window.graph_library_panel._current_graph_id() == g1_id
    window.close()


def test_deleting_the_origin_graph_clears_the_graph_library_selection(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()
    assert window.graph_library_panel._current_graph_id() is not None

    window.graph_library_panel.graph_list.setCurrentRow(0)
    window.graph_library_panel.delete_button.click()

    assert window.graph_library_panel.graph_list.currentRow() == -1
    assert window.graph_library_panel._current_graph_id() is None
    # The active panel keeps its independent working copy regardless.
    assert len(window.figure_model.series) == 1
    window.close()


def test_graph_library_selection_sync_does_not_mark_the_project_dirty(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window._set_dirty(False)

    window.toolbar_panel_combo.setCurrentIndex(1)  # Panel 2 -- pure navigation

    assert window._dirty is False
    window.close()


def test_graph_library_selection_sync_does_not_create_an_undo_checkpoint(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("g1", True)))
    window.graph_library_panel.save_button.click()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    can_undo_before = window.undo_action.isEnabled()

    window.toolbar_panel_combo.setCurrentIndex(1)  # Panel 2 -- pure navigation

    assert window.undo_action.isEnabled() == can_undo_before
    window.close()


def test_save_current_panel_as_graph_through_main_window(qapp, monkeypatch):
    window = _dirty_window(qapp)
    window._set_dirty(False)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("My Graph", True)))

    window.graph_library_panel.save_button.click()

    assert len(window._project.graph_library.graphs) == 1
    assert window._dirty is True  # library mutation dirties the project
    window.close()


def test_load_graph_into_active_panel_through_main_window_rerenders_and_updates_series_panel(qapp, monkeypatch):
    window = _dirty_window(qapp)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()
    window._on_clear_plot()
    assert window.figure_model.series == []
    window._set_dirty(False)

    window.graph_library_panel.graph_list.setCurrentRow(0)
    window.graph_library_panel.load_button.click()

    assert len(window.figure_model.series) == 1
    assert window.series_panel.series_list.count() == 1
    assert window._dirty is True
    window.close()


def test_rename_duplicate_delete_graph_through_main_window(qapp, monkeypatch):
    window = _dirty_window(qapp)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()
    window.graph_library_panel.graph_list.setCurrentRow(0)

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1 renamed", True)))
    window.graph_library_panel.rename_button.click()
    assert window._project.graph_library.graphs[0].name == "G1 renamed"

    window.graph_library_panel.duplicate_button.click()
    assert len(window._project.graph_library.graphs) == 2

    window.graph_library_panel.graph_list.setCurrentRow(0)
    window.graph_library_panel.delete_button.click()
    assert len(window._project.graph_library.graphs) == 1
    window.close()


# --- Dirty state ---------------------------------------------------------------


def test_clean_project_title_has_no_dirty_marker(qapp):
    window = _make_window(qapp)
    assert window.windowTitle() == f"Untitled Project — {APP_NAME}"
    assert window._dirty is False
    window.close()


def test_a_real_content_edit_sets_dirty_and_the_title_marker(qapp):
    window = _dirty_window(qapp)
    assert window._dirty is True
    assert window.windowTitle() == f"Untitled Project* — {APP_NAME}"
    window.close()


def test_save_clears_dirty_and_the_title_marker(qapp, monkeypatch, tmp_path):
    window = _dirty_window(qapp)
    out_path = tmp_path / "p.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))

    window.save_project_as_action.trigger()

    assert window._dirty is False
    assert "*" not in window.windowTitle()
    window.close()


def test_switching_active_panel_alone_does_not_set_dirty(qapp):
    window = _make_window(qapp)
    window.figure_model.set_layout(1, 2)
    window._set_dirty(False)  # set_layout above goes through _on_figure_content_changed; re-baseline

    window._set_active_panel(1)

    assert window._dirty is False
    window.close()


class _FakeMplEvent:
    """Minimal stand-in for a Matplotlib mouse event -- only the attributes
    `_on_mouse_move`/`_on_canvas_click` actually read."""

    def __init__(self, inaxes=None, xdata=None, ydata=None):
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata


def test_mouse_move_and_click_do_not_set_dirty(qapp):
    window = _make_window(qapp)
    assert window._dirty is False

    window._on_mouse_move(_FakeMplEvent())  # outside any axes
    window._on_mouse_leave(_FakeMplEvent())
    window._on_canvas_click(_FakeMplEvent())  # outside any axes -- a no-op

    assert window._dirty is False
    window.close()


def test_rerender_alone_does_not_set_dirty(qapp):
    """Matplotlib's own pan/zoom navigation only ever changes the Axes'
    view limits and re-renders (`_rerender`/`sync_axes_limits`) -- it never
    touches the declarative model, so it must never mark the project
    dirty. `_rerender()` is the shared code path pan/zoom-driven redraws
    and this test both go through."""
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    window._set_dirty(False)

    window._rerender()

    assert window._dirty is False
    window.close()


# --- Workbenches: menu, dirty state, undo isolation, core workflow -------------


def test_workbench_menu_has_new_rename_duplicate_delete_actions(qapp):
    window = _make_window(qapp)
    assert window.new_workbench_action.text() == "New Workbench"
    assert window.rename_workbench_action.text() == "Rename Workbench"
    assert window.duplicate_workbench_action.text() == "Duplicate Workbench"
    assert window.delete_workbench_action.text() == "Delete Workbench"
    window.close()


def test_delete_workbench_action_disabled_with_only_one_workbench(qapp):
    window = _make_window(qapp)
    window._sync_workbench_menu_state()
    assert window.delete_workbench_action.isEnabled() is False
    window.close()


def test_delete_workbench_action_enabled_once_a_second_exists(qapp):
    window = _make_window(qapp)
    window.workbench_tab_bar.new_button.click()
    window._sync_workbench_menu_state()
    assert window.delete_workbench_action.isEnabled() is True
    window.close()


def test_new_workbench_action_creates_and_switches_via_the_menu(qapp):
    window = _make_window(qapp)
    original_id = window._project.active_workbench_id

    window.new_workbench_action.trigger()

    assert len(window._project.workbenches) == 2
    assert window._project.active_workbench_id != original_id
    assert window.workbench_tab_bar.tab_bar.count() == 2
    window.close()


def test_rename_workbench_action_targets_the_active_workbench(qapp, monkeypatch):
    window = _make_window(qapp)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("CV Scan Rates", True)))

    window.rename_workbench_action.trigger()

    assert window._project.active_workbench.name == "CV Scan Rates"
    assert window.workbench_tab_bar.tab_bar.tabText(0) == "CV Scan Rates"
    window.close()


def test_duplicate_workbench_action_targets_the_active_workbench(qapp):
    window = _make_window(qapp)
    original_id = window._project.active_workbench_id

    window.duplicate_workbench_action.trigger()

    assert len(window._project.workbenches) == 2
    assert window._project.active_workbench_id != original_id
    assert window._project.active_workbench.name == "Workbench 1 (Copy)"
    window.close()


def test_delete_workbench_action_requires_confirmation(qapp, monkeypatch):
    window = _make_window(qapp)
    window.workbench_tab_bar.new_button.click()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Cancel))

    window.delete_workbench_action.trigger()

    assert len(window._project.workbenches) == 2  # cancelled -- nothing removed
    window.close()


def test_delete_workbench_action_removes_on_confirmation(qapp, monkeypatch):
    window = _make_window(qapp)
    window.workbench_tab_bar.new_button.click()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))

    window.delete_workbench_action.trigger()

    assert len(window._project.workbenches) == 1
    window.close()


# --- Workbench dirty state -------------------------------------------------------


def test_creating_a_workbench_marks_dirty(qapp):
    window = _make_window(qapp)
    window._set_dirty(False)

    window.workbench_tab_bar.new_button.click()

    assert window._dirty is True
    window.close()


def test_renaming_a_workbench_marks_dirty(qapp, monkeypatch):
    window = _make_window(qapp)
    window._set_dirty(False)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Renamed", True)))

    window.rename_workbench_action.trigger()

    assert window._dirty is True
    window.close()


def test_duplicating_a_workbench_marks_dirty(qapp):
    window = _make_window(qapp)
    window._set_dirty(False)

    window.duplicate_workbench_action.trigger()

    assert window._dirty is True
    window.close()


def test_deleting_a_workbench_marks_dirty(qapp, monkeypatch):
    window = _make_window(qapp)
    window.workbench_tab_bar.new_button.click()
    window._set_dirty(False)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))

    window.delete_workbench_action.trigger()

    assert window._dirty is True
    window.close()


def test_switching_the_active_workbench_alone_does_not_mark_dirty(qapp):
    window = _make_window(qapp)
    window.workbench_tab_bar.new_button.click()
    second_id = window._project.active_workbench_id
    first_id = next(w.id for w in window._project.workbenches if w.id != second_id)
    window._set_dirty(False)

    window._on_workbench_tab_selected(first_id)

    assert window._dirty is False
    window.close()


# --- Prominent Undo/Redo: shared QAction, icons, disabled state -----------------


def test_undo_redo_toolbar_and_menu_share_the_same_qaction_objects(qapp):
    """Menu Undo/Redo and toolbar Undo/Redo must be driven by the exact
    same QAction -- never two separate systems."""
    from PySide6.QtWidgets import QToolBar

    window = _make_window(qapp)
    main_toolbar = next(tb for tb in window.findChildren(QToolBar) if tb.windowTitle() == "Main")
    assert window.undo_action in main_toolbar.actions()
    assert window.redo_action in main_toolbar.actions()
    window.close()


def test_undo_redo_actions_have_icons(qapp):
    window = _make_window(qapp)
    assert not window.undo_action.icon().isNull()
    assert not window.redo_action.icon().isNull()
    window.close()


def test_undo_redo_actions_start_disabled(qapp):
    window = _make_window(qapp)
    assert window.undo_action.isEnabled() is False
    assert window.redo_action.isEnabled() is False
    window.close()


def test_undo_action_enabled_after_a_content_edit(qapp):
    window = _dirty_window(qapp)
    assert window.undo_action.isEnabled() is True
    assert window.redo_action.isEnabled() is False
    window.close()


def test_redo_action_enabled_after_an_undo(qapp):
    window = _dirty_window(qapp)
    window._on_undo()
    assert window.redo_action.isEnabled() is True
    window.close()


def test_redo_action_supports_ctrl_y_in_addition_to_the_standard_shortcut(qapp):
    from PySide6.QtGui import QKeySequence

    window = _make_window(qapp)
    shortcuts = [s.toString() for s in window.redo_action.shortcuts()]
    assert QKeySequence("Ctrl+Y").toString() in shortcuts
    window.close()


def test_undo_redo_tooltips_are_generic_not_per_action_descriptions(qapp):
    """Per-action descriptions (e.g. "Undo: Change Series Color") are
    explicitly deferred this phase -- generic tooltips are sufficient."""
    window = _make_window(qapp)
    assert window.undo_action.toolTip() == "Undo (Ctrl+Z)"
    assert window.redo_action.toolTip() == "Redo (Ctrl+Shift+Z)"
    window.close()


# --- Per-Workbench Undo isolation ------------------------------------------------


def test_undo_is_isolated_per_workbench(qapp):
    """The exact scenario: Workbench A changes a series color (A1);
    Workbench B changes its title twice (B1, B2). Undo while B is active
    must undo only B2; switching to A and undoing must undo only A1;
    switching back to B must show it still in its (already undone) B1
    state."""
    window = _dirty_window(qapp)  # one dataset + one series plotted
    workbench_a_id = window._project.active_workbench_id
    original_color = window.figure_model.series[0].color
    window.figure_model.series[0].color = "#111111"
    window._on_figure_content_changed()  # A1

    window.workbench_tab_bar.new_button.click()
    workbench_b_id = window._project.active_workbench_id
    window.figure_model.title = "B1"
    window._on_figure_content_changed()  # B1
    window.figure_model.title = "B2"
    window._on_figure_content_changed()  # B2

    window._on_undo()
    assert window.figure_model.title == "B1"

    window._on_workbench_tab_selected(workbench_a_id)
    assert window.figure_model.series[0].color == "#111111"

    window._on_undo()
    assert window.figure_model.series[0].color == original_color

    window._on_workbench_tab_selected(workbench_b_id)
    assert window.figure_model.title == "B1"  # untouched by A's undo
    window.close()


def test_deleting_a_workbench_drops_its_undo_state(qapp, monkeypatch):
    window = _make_window(qapp)
    window.workbench_tab_bar.new_button.click()
    second_id = window._project.active_workbench_id
    window.figure_model.title = "Edited"
    window._on_figure_content_changed()
    assert second_id in window._undo_managers

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    window._on_delete_workbench_requested(second_id)

    assert second_id not in window._undo_managers
    assert second_id not in window._pending_snapshots
    window.close()


# --- Core workflow scenario (task section 19) -----------------------------------


def test_core_multi_workbench_workflow_end_to_end(qapp, monkeypatch, tmp_path):
    window = _make_window(qapp)
    dataset = _make_dataset("d")
    window.dataset_manager.add(dataset)
    window.dataset_panel.datasets_changed.emit()

    # Workbench A: "CV Comparison", 2x2, four independent graphs.
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("CV Comparison", True)))
    window.rename_workbench_action.trigger()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # 2x2
    graph_ids = []
    for panel_index in range(4):
        window._set_active_panel(panel_index)
        window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label=f"g{panel_index}")])
        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, i=panel_index, **k: (f"Graph {i}", True))
        )
        window.graph_library_panel.save_button.click()
        graph_ids.append(window._project.graph_library.graphs[-1].id)
    workbench_a_id = window._project.active_workbench_id
    assert len(window._project.graph_library.graphs) == 4

    # Workbench B: "New Graph", 1x1, a fresh graph.
    window.new_workbench_action.trigger()
    window._project.rename_workbench(window._project.active_workbench_id, "New Graph")
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label="fresh")])
    workbench_b_id = window._project.active_workbench_id

    # Switch back to A: all 4 graphs unchanged.
    window._on_workbench_tab_selected(workbench_a_id)
    assert window.figure_model.layout == (2, 2)
    assert sum(len(panel.series) for panel in window.figure_model.panels) == 4
    assert all(len(panel.series) == 1 for panel in window.figure_model.panels)

    # Switch to B: new graph unchanged.
    window._on_workbench_tab_selected(workbench_b_id)
    assert window.figure_model.layout == (1, 1)
    assert window.figure_model.series[0].label == "fresh"

    # Workbench C: 1x2, load Saved Graph 0 into Panel 1, Graph 1 into Panel 2.
    window.new_workbench_action.trigger()
    window._project.rename_workbench(window._project.active_workbench_id, "Publication Figure")
    workbench_c_id = window._project.active_workbench_id
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # 1x2
    window._set_active_panel(0)
    list_row = next(
        i
        for i in range(window.graph_library_panel.graph_list.count())
        if window.graph_library_panel.graph_list.item(i).data(Qt.UserRole) == graph_ids[0]
    )
    window.graph_library_panel.graph_list.setCurrentRow(list_row)
    window.graph_library_panel.load_button.click()
    window._set_active_panel(1)
    list_row = next(
        i
        for i in range(window.graph_library_panel.graph_list.count())
        if window.graph_library_panel.graph_list.item(i).data(Qt.UserRole) == graph_ids[1]
    )
    window.graph_library_panel.graph_list.setCurrentRow(list_row)
    window.graph_library_panel.load_button.click()

    assert len(window._project.workbenches) == 3

    out_path = tmp_path / "core_workflow.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()
    assert window._dirty is False

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))
    window.new_project_action.trigger()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    reopened = window._project
    assert len(reopened.workbenches) == 3
    names = {w.name for w in reopened.workbenches}
    assert names == {"CV Comparison", "New Graph", "Publication Figure"}
    assert reopened.active_workbench_id == workbench_c_id

    reopened_a = reopened.get_workbench(workbench_a_id)
    assert reopened_a.figure.layout == (2, 2)
    assert sum(len(panel.series) for panel in reopened_a.figure.panels) == 4

    reopened_b = reopened.get_workbench(workbench_b_id)
    assert reopened_b.figure.series[0].label == "fresh"

    reopened_c = reopened.get_workbench(workbench_c_id)
    assert reopened_c.figure.layout == (1, 2)
    assert len(reopened_c.figure.panels[0].series) == 1
    assert len(reopened_c.figure.panels[1].series) == 1

    # Shared datasets/Graph Library.
    assert len(reopened.dataset_manager.datasets) == 1
    assert len(reopened.graph_library.graphs) == 4

    # Independent working copies remain independent.
    reopened_c.figure.panels[0].title = "Edited in C"
    assert reopened_a.figure.panels[0].title == ""
    window.close()


# --- Workbench duplication (task section 20) -------------------------------------


def test_duplicate_workbench_end_to_end_through_main_window(qapp):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", color="#111111")])
    original_id = window._project.active_workbench_id

    window.duplicate_workbench_action.trigger()
    duplicate_id = window._project.active_workbench_id
    assert duplicate_id != original_id

    # Change title, one series color, and panel layout on the duplicate.
    window.figure_model.active_panel.title = "Duplicate Title"
    window.figure_model.series[0].color = "#999999"
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # 1x2

    original = window._project.get_workbench(original_id)
    assert original.figure.active_panel.title == ""
    assert original.figure.series[0].color == "#111111"
    assert original.figure.layout == (1, 1)

    # Datasets remain shared, not duplicated.
    assert original.figure.series[0].dataset is window.figure_model.series[0].dataset
    assert len(window.dataset_manager) == 1
    window.close()


# --- Unsaved-work close protection ---------------------------------------------


def test_close_event_prompts_when_dirty(qapp, monkeypatch):
    window = _dirty_window(qapp)
    calls = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: calls.append(True) or QMessageBox.Discard)
    )

    event = QCloseEvent()
    window.closeEvent(event)

    assert calls == [True]
    assert event.isAccepted()


def test_close_event_does_not_prompt_when_clean(qapp, monkeypatch):
    window = _make_window(qapp)
    calls = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: calls.append(True)))

    event = QCloseEvent()
    window.closeEvent(event)

    assert calls == []
    assert event.isAccepted()


def test_close_event_discard_accepts_and_leaves_project_unsaved(qapp, monkeypatch):
    window = _dirty_window(qapp)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted()
    assert window._project.path is None  # nothing was saved


def test_close_event_cancel_ignores_the_close(qapp, monkeypatch):
    window = _dirty_window(qapp)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Cancel))

    event = QCloseEvent()
    window.closeEvent(event)

    assert not event.isAccepted()
    assert window._dirty is True
    window.close()  # actually close it now so the test doesn't leak a window


def test_close_event_save_writes_the_file_and_accepts(qapp, monkeypatch, tmp_path):
    window = _dirty_window(qapp)
    out_path = tmp_path / "on_close.gnovi"
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Save))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted()
    assert out_path.exists()
    assert window._dirty is False


def test_close_event_save_failure_does_not_close(qapp, monkeypatch, tmp_path):
    window = _dirty_window(qapp)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Save))
    out_path = tmp_path / "will_fail.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))

    def _raise_os_error(*a, **k):
        raise OSError("disk is full")

    monkeypatch.setattr("gnovi_plot.gui.main_window.save_project", _raise_os_error)
    critical_calls = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: critical_calls.append(a)))

    event = QCloseEvent()
    window.closeEvent(event)

    assert not event.isAccepted()  # failed save must not close the window
    assert len(critical_calls) == 1
    assert window._dirty is True
    # Cleanup: bypass the still-failing save path rather than relying on
    # `monkeypatch.undo()`, which would also revert the autouse
    # discard-by-default patch shared by this same `monkeypatch` fixture
    # instance and risk a real blocking dialog on close.
    window._set_dirty(False)
    window.close()


def test_new_project_prompts_and_cancel_leaves_project_untouched(qapp, monkeypatch):
    window = _dirty_window(qapp)
    original_project = window._project
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Cancel))

    window.new_project_action.trigger()

    assert window._project is original_project
    assert window._dirty is True
    window.close()


def test_open_project_prompts_and_cancel_leaves_project_untouched(qapp, monkeypatch):
    window = _dirty_window(qapp)
    original_project = window._project
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Cancel))

    file_dialog_calls = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: file_dialog_calls.append(True))
    )

    window.open_project_action.trigger()

    assert window._project is original_project
    assert file_dialog_calls == []  # never even got to the file picker
    window.close()


# --- A cancelled/failed Save must never let a destructive action proceed -------


def test_close_event_save_then_cancelled_save_as_does_not_close(qapp, monkeypatch):
    """The project has never been saved (no path yet), so choosing Save
    falls through to Save As -- if the user then cancels that file picker,
    nothing was actually saved, so closing must still be blocked (see
    `_on_save_project_as`'s docstring)."""
    window = _dirty_window(qapp)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Save))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))

    event = QCloseEvent()
    window.closeEvent(event)

    assert not event.isAccepted()
    assert window._project.path is None
    assert window._dirty is True
    window._set_dirty(False)
    window.close()


def test_new_project_save_then_cancelled_save_as_leaves_project_untouched(qapp, monkeypatch):
    window = _dirty_window(qapp)
    original_project = window._project
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Save))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))

    window.new_project_action.trigger()

    assert window._project is original_project
    assert window._dirty is True
    window.close()


def test_open_project_save_then_cancelled_save_as_leaves_project_untouched(qapp, monkeypatch):
    window = _dirty_window(qapp)
    original_project = window._project
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Save))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    open_dialog_calls = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: open_dialog_calls.append(True))
    )

    window.open_project_action.trigger()

    assert window._project is original_project
    assert open_dialog_calls == []  # never even got to the file picker
    window.close()
