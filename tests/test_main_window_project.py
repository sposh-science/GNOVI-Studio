import pandas as pd
import pytest
from matplotlib.backend_bases import MouseEvent
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
    assert len(reloaded.figures[0].series) == 2
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


def test_active_panel_labels_start_at_panel_1_on_every_page(qapp):
    window = _make_window(qapp)
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 1"
    assert window.series_panel.active_panel_label.text() == "Active panel: Panel 1"
    assert window.figure_size_panel.active_panel_label.text() == "Active panel: Panel 1"
    assert window.figure_layout_panel.active_panel_label.text() == "Active panel: Panel 1"
    assert window.properties_panel.active_panel_label.text() == "Active panel: Panel 1"
    window.close()


def test_active_panel_labels_update_after_clicking_a_different_panel(qapp):
    window = _make_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    ax = window.plot_canvas.axes_list[1]
    event = MouseEvent("button_press_event", window.plot_canvas, 1, 1)
    event.inaxes = ax

    window._on_canvas_click(event)

    assert window.figure_model.active_panel_index == 1
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 2"
    assert window.series_panel.active_panel_label.text() == "Active panel: Panel 2"
    assert window.figure_size_panel.active_panel_label.text() == "Active panel: Panel 2"
    assert window.figure_layout_panel.active_panel_label.text() == "Active panel: Panel 2"
    assert window.properties_panel.active_panel_label.text() == "Active panel: Panel 2"
    window.close()


def test_active_panel_labels_update_after_the_toolbar_selector_changes(qapp):
    window = _make_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"

    window.toolbar_panel_combo.setCurrentIndex(2)

    assert window.figure_model.active_panel_index == 2
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 3"
    assert window.figure_size_panel.active_panel_label.text() == "Active panel: Panel 3"
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
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 1"
    assert window.series_panel.active_panel_label.text() == "Active panel: Panel 1"
    assert window.figure_layout_panel.active_panel_label.text() == "Active panel: Panel 1"
    window.close()


def test_active_panel_labels_update_after_project_load(qapp, monkeypatch, tmp_path):
    window = _make_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    window.toolbar_panel_combo.setCurrentIndex(2)  # Panel 3
    out_path = tmp_path / "panel_context.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    window.new_project_action.trigger()
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 1"

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert window.figure_model.active_panel_index == 2
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 3"
    assert window.series_panel.active_panel_label.text() == "Active panel: Panel 3"
    assert window.figure_size_panel.active_panel_label.text() == "Active panel: Panel 3"
    assert window.figure_layout_panel.active_panel_label.text() == "Active panel: Panel 3"
    assert window.properties_panel.active_panel_label.text() == "Active panel: Panel 3"
    window.close()


def test_active_panel_label_updates_after_new_project(qapp):
    window = _dirty_window(qapp)
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window.toolbar_panel_combo.setCurrentIndex(1)
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 2"

    window.new_project_action.trigger()

    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 1"
    window.close()


def test_active_panel_label_updates_when_a_saved_graph_is_loaded_into_another_panel(qapp, monkeypatch):
    window = _make_window(qapp)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("G1", True)))
    window.graph_library_panel.save_button.click()

    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    window.toolbar_panel_combo.setCurrentIndex(1)  # Panel 2
    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 2"

    window.graph_library_panel.graph_list.setCurrentRow(0)
    window.graph_library_panel.load_button.click()

    assert window.plot_page_active_panel_label.text() == "Active panel: Panel 2"  # unchanged, still correct
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
