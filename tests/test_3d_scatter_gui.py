"""3D Scatter end-to-end GUI workflows: creation (the "3D" sidebar page via
`gui.widgets.plot3d_panel.Plot3DPanel`, reached directly or via "Panels ->
Add 3D Scatter…"), creation safety (empty vs. populated 2D panel, append to
an existing Panel3D), active-panel selection/right-click targeting, Focus/
Restore, Extract, direct Panel export, full-Figure export, and Workbench
switching -- all reusing the exact same handlers/domain operations earlier
PRs already established for 2D Panels (see each handler's own docstring in
`gui.main_window`); most of these tests exist to prove Panel3D integrates
with them, not to re-prove those mechanisms work at all.
"""

import re

import pandas as pd
from PySide6.QtWidgets import QMessageBox

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.figure import Panel, Panel3D
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="mat"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0, 300.0], "composition": [0.1, 0.1, 0.1, 0.2], "conductivity": [2.4, 2.9, 3.5, 3.1]}
    )
    return Dataset(name=name, dataframe=df)


def _make_3_panel_window():
    """Every panel starts POPULATED with 2D content -- deliberately, so
    tests exercising creation on an already-occupied panel (the common
    case in this file) exercise the real confirm-before-replace path (see
    `_make_3d_panel_at`), not an empty-panel shortcut."""
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(4)  # "1 x 3"
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.plot3d_panel.set_manager(window.dataset_manager)
    for i, panel in enumerate(window.figure_model.panels):
        panel.title = f"Panel {i + 1}"
        panel.add_series(PlotSeries.line(dataset, "temperature", "conductivity", label=f"Series {i + 1}"))
    window._rerender()
    return window, dataset


class _FakeAction:
    def __init__(self, text):
        self.text = text
        self.enabled = True

    def setEnabled(self, value):
        self.enabled = value


def _fake_menu(chosen_text=None):
    class _FakeMenu:
        def __init__(self, *_a, **_k):
            self.actions = {}

        def addSeparator(self):
            pass

        def addAction(self, text):
            action = _FakeAction(text)
            self.actions[text] = action
            return action

        def exec(self, *_a, **_k):
            return self.actions.get(chosen_text) if chosen_text is not None else None

    return _FakeMenu


class _FakeContextMenuEvent:
    def __init__(self, inaxes=None, button=3, x=0, y=0):
        self.inaxes = inaxes
        self.button = button
        self.x = x
        self.y = y


def _fill_3d_form(window, dataset, **overrides):
    plot3d = window.plot3d_panel
    dataset_index = plot3d.dataset_combo.findData(dataset.id)
    plot3d.dataset_combo.setCurrentIndex(dataset_index)
    plot3d.x_combo.setCurrentText(overrides.get("x_column", "temperature"))
    plot3d.y_combo.setCurrentText(overrides.get("y_column", "composition"))
    plot3d.z_combo.setCurrentText(overrides.get("z_column", "conductivity"))


def _make_3d_panel_at(window, dataset, index, monkeypatch, **overrides):
    """Drives the real "3D" sidebar page end to end: fills the Add 3D
    Series form and clicks "Add to 3D Plot" -- exactly what a user does,
    no dialog/`.exec()` faking needed since this is a plain embedded
    widget. `QMessageBox.question` is monkeypatched to auto-confirm (Yes)
    since `_make_3_panel_window`'s panels all start populated with 2D
    content, so converting any of them requires that confirmation (see
    `MainWindow._on_add_3d_series_requested`) -- dedicated tests below
    exercise the Cancel path and the truly-empty-panel no-confirmation
    path directly."""
    window.toolbar_panel_combo.setCurrentIndex(index)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    _fill_3d_form(window, dataset, **overrides)
    window.plot3d_panel.add_button.click()
    return window.figure_model.panels[index]


# --- Creation: empty panel, populated panel, append ---------------------------------


def test_add_to_3d_plot_on_an_empty_panel_converts_without_confirmation(qapp, monkeypatch):
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.plot3d_panel.set_manager(window.dataset_manager)

    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.Yes))
    _fill_3d_form(window, dataset)
    window.plot3d_panel.add_button.click()

    panel = window.figure_model.panels[0]
    assert isinstance(panel, Panel3D)
    assert asked == []  # never asked -- the panel had no series to lose
    assert panel.series[0].x_column == "temperature"
    assert panel.series[0].z_column == "conductivity"
    window.close()


def test_add_to_3d_plot_on_a_populated_2d_panel_requires_confirmation(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    window.toolbar_panel_combo.setCurrentIndex(1)

    asked = []
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.Yes)
    )
    _fill_3d_form(window, dataset)
    window.plot3d_panel.add_button.click()

    assert asked == [True]
    assert isinstance(window.figure_model.panels[1], Panel3D)
    window.close()


def test_add_to_3d_plot_on_a_populated_2d_panel_cancel_leaves_everything_unchanged(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    window.toolbar_panel_combo.setCurrentIndex(1)
    original_panel = window.figure_model.panels[1]
    window._set_dirty(False)

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    _fill_3d_form(window, dataset)
    window.plot3d_panel.add_button.click()

    assert window.figure_model.panels[1] is original_panel
    assert isinstance(window.figure_model.panels[1], Panel)
    assert len(window.figure_model.panels[1].series) == 1
    assert window._dirty is False
    window.close()


def test_add_to_3d_plot_marks_the_project_dirty(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    window._set_dirty(False)

    _make_3d_panel_at(window, dataset, 1, monkeypatch)

    assert window._dirty is True
    window.close()


def test_add_to_3d_plot_does_not_change_the_layout_or_panel_count(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()

    _make_3d_panel_at(window, dataset, 1, monkeypatch)

    assert window.figure_model.layout == (1, 3)
    assert len(window.figure_model.panels) == 3
    window.close()


def test_reinvoking_add_to_3d_plot_on_an_existing_panel3d_appends_a_series(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    original_id = panel.id
    assert len(panel.series) == 1

    window.plot3d_panel.add_button.click()  # form is already filled from the first Add

    assert window.figure_model.panels[1] is panel  # never replaced
    assert panel.id == original_id
    assert len(panel.series) == 2
    window.close()


def test_add_to_3d_plot_invalid_numeric_data_shows_a_controlled_error_not_a_crash(qapp, monkeypatch):
    window, _dataset = _make_3_panel_window()
    bad_dataset = Dataset(name="bad", dataframe=pd.DataFrame({"a": ["x", "y"], "b": ["x", "y"], "c": ["x", "y"]}))
    window.dataset_manager.add(bad_dataset)
    window.plot3d_panel.set_manager(window.dataset_manager)
    window.toolbar_panel_combo.setCurrentIndex(1)

    _fill_3d_form(window, bad_dataset, x_column="a", y_column="b", z_column="c")
    window.plot3d_panel.add_button.click()  # must not raise

    # `.isVisible()` needs a shown top-level window to mean anything under
    # Qt's offscreen platform (never called here) -- the error text itself
    # is the meaningful assertion: a controlled, user-facing message, not a
    # crash.
    assert window.plot3d_panel.error_label.text() != ""
    assert not isinstance(window.figure_model.panels[1], Panel3D)  # unchanged
    window.close()


# --- Adaptive Series/Axes tabs: mixed 2D|3D|2D, no stale controls -------------------


def test_series_and_axes_tabs_show_the_correct_page_for_each_panel_in_a_mixed_figure(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)

    window.toolbar_panel_combo.setCurrentIndex(0)
    window.series_panel.refresh()
    window.properties_panel.refresh()
    assert window.series_panel._stack.currentWidget() is window.series_panel._page_2d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_2d

    window.toolbar_panel_combo.setCurrentIndex(1)
    window.series_panel.refresh()
    window.properties_panel.refresh()
    assert window.series_panel._stack.currentWidget() is window.series_panel._page_3d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_3d
    assert window.series_panel.series3d_list.count() == len(panel3d.series)

    window.toolbar_panel_combo.setCurrentIndex(2)
    window.series_panel.refresh()
    window.properties_panel.refresh()
    assert window.series_panel._stack.currentWidget() is window.series_panel._page_2d
    assert window.properties_panel._stack.currentWidget() is window.properties_panel._page_2d
    window.close()


def test_3d_axes_page_edits_title_and_labels(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)
    window.properties_panel.refresh()

    window.properties_panel.d3_title_edit.setText("Conductivity scatter")
    window.properties_panel._apply_3d_title()
    window.properties_panel.d3_xlabel_edit.setText("T (K)")
    window.properties_panel._apply_3d_xlabel()

    assert panel3d.title == "Conductivity scatter"
    assert panel3d.x_label == "T (K)"
    window.close()


def test_3d_series_page_edits_label_and_marker(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)
    window.series_panel.refresh()
    window.series_panel.series3d_list.setCurrentRow(0)

    window.series_panel.d3_label_edit.setText("mat scatter")
    window.series_panel._apply_3d_label()
    window.series_panel.d3_marker_size_spin.setValue(9.0)
    window.series_panel._apply_3d_marker_size(9.0)

    assert panel3d.series[0].label == "mat scatter"
    assert panel3d.series[0].marker_size == 9.0
    window.close()


# --- Clear 3D Plot --------------------------------------------------------------------


def test_clear_3d_plot_keeps_the_panel3d_and_removes_series(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    panel3d.title = "Keep me"
    window.toolbar_panel_combo.setCurrentIndex(1)
    window.plot3d_panel.refresh()

    window.plot3d_panel.clear_button.click()

    assert window.figure_model.panels[1] is panel3d
    assert isinstance(window.figure_model.panels[1], Panel3D)
    assert window.figure_model.panels[1].title == "Keep me"
    assert window.figure_model.panels[1].series == []
    window.close()


# --- Set Current View / Reset View, undo/dirty boundary ------------------------------


def test_set_current_view_commits_the_live_camera_and_marks_dirty_and_undoable(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)
    window._rerender()
    ax = window.plot_canvas.active_axes(window.figure_model)
    ax.view_init(elev=12.0, azim=99.0)
    window._set_dirty(False)
    undo_count_before = len(window._undo_manager._undo)

    window._on_set_current_3d_view_requested()

    assert panel3d.elevation == 12.0
    assert panel3d.azimuth == 99.0
    assert window._dirty is True
    assert len(window._undo_manager._undo) > undo_count_before
    window._on_undo()
    assert window.figure_model.panels[1].elevation != 12.0 or window.figure_model.panels[1].azimuth != 99.0
    window.close()


def test_ordinary_rotation_never_marks_dirty_or_touches_the_model(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)
    window._rerender()
    before_elev, before_azim = panel3d.elevation, panel3d.azimuth
    window._set_dirty(False)

    ax = window.plot_canvas.active_axes(window.figure_model)
    ax.view_init(elev=55.0, azim=155.0)  # interactive rotation -- never committed on its own

    assert panel3d.elevation == before_elev
    assert panel3d.azimuth == before_azim
    assert window._dirty is False
    window.close()


def test_reset_view_restores_default_camera(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    panel3d.elevation = 5.0
    panel3d.azimuth = 5.0
    window.toolbar_panel_combo.setCurrentIndex(1)
    window.properties_panel.refresh()

    window.properties_panel.d3_reset_view_button.click()

    assert panel3d.elevation == Panel3D().elevation
    assert panel3d.azimuth == Panel3D().azimuth
    window.close()


# --- Active-panel selection / right-click targeting ---------------------------------


def test_selecting_the_3d_panel_makes_it_active(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    _make_3d_panel_at(window, dataset, 1, monkeypatch)

    window.toolbar_panel_combo.setCurrentIndex(1)

    assert isinstance(window.figure_model.active_panel, Panel3D)
    window.close()


def test_right_click_maps_to_the_correct_3d_panel(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 2, monkeypatch)  # 3rd panel
    window.toolbar_panel_combo.setCurrentIndex(0)  # active panel starts elsewhere

    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu(None))
    clicked_axes = window.plot_canvas.axes_list[2]
    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert window.figure_model.active_panel is panel3d
    window.close()


# --- Focus / Restore -----------------------------------------------------------------


def test_focus_works_for_a_3d_panel(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)

    window._focus_panel(panel3d)

    assert window.plot_canvas.is_focused is True
    assert window._current_focused_panel() is panel3d  # same original object
    assert len(window.plot_canvas.axes_list) == 1
    window.close()


def test_restore_returns_the_mixed_figure_correctly(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)
    window._focus_panel(panel3d)

    window._restore_multi_panel_view()

    assert window.plot_canvas.is_focused is False
    assert len(window.plot_canvas.axes_list) == 3
    assert window.figure_model.panels[1] is panel3d  # never cloned, never converted
    window.close()


def test_focus_does_not_mark_dirty_and_does_not_clone_the_panel3d(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)
    window._set_dirty(False)

    window._focus_panel(panel3d)

    assert window._dirty is False
    assert len(window.figure_model.panels) == 3  # no clone appended
    window.close()


# --- Extract ---------------------------------------------------------------------


def test_extract_creates_an_independent_panel3d_structure(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)

    window._on_extract_panel_requested()

    assert len(window._project.workbenches) == 2
    extracted_panel = window._project.workbenches[-1].figure.panels[0]
    assert isinstance(extracted_panel, Panel3D)
    assert extracted_panel.id != panel3d.id
    # Original untouched.
    source_panel = window._project.workbenches[0].figure.panels[1]
    assert source_panel is panel3d


def test_extract_shares_dataset_identity_for_a_panel3d(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window.toolbar_panel_combo.setCurrentIndex(1)

    window._on_extract_panel_requested()

    extracted_panel = window._project.workbenches[-1].figure.panels[0]
    assert extracted_panel.series[0].dataset is dataset


# --- Export Panel / full-Figure export --------------------------------------------


def test_export_panel_exports_only_the_3d_panel(qapp, monkeypatch, tmp_path):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    panel3d.title = "ThreeD"
    window.toolbar_panel_combo.setCurrentIndex(1)

    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window, panel=panel3d, dataset_manager=window.dataset_manager
    )
    dialog.format_combo.setCurrentText("SVG")
    out_path = tmp_path / "panel3d.svg"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    content = out_path.read_text(encoding="utf-8")
    assert "ThreeD" in content
    assert "Panel 1" not in content
    assert "Panel 3" not in content
    window.close()


def test_export_panel_for_3d_causes_no_project_mutation(qapp, monkeypatch, tmp_path):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    window._set_dirty(False)
    original_ids = [w.id for w in window._project.workbenches]

    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window, panel=panel3d, dataset_manager=window.dataset_manager
    )
    dialog.path_edit.setText(str(tmp_path / "panel3d.png"))
    dialog._on_accept()

    assert window._dirty is False
    assert [w.id for w in window._project.workbenches] == original_ids
    window.close()


def test_full_figure_export_works_for_a_mixed_figure(qapp, monkeypatch, tmp_path):
    window, dataset = _make_3_panel_window()
    _make_3d_panel_at(window, dataset, 1, monkeypatch)
    # Set the title through the real Axes-tab GUI path (not a direct model
    # mutation) so the on-screen canvas actually re-renders with it -- an
    # unfocused "Complete Figure" export intentionally saves the LIVE canvas
    # Figure as-is (see `ExportFigureDialog._complete_figure_export_uses_
    # headless_path`'s own docstring), so a title set only on the model
    # without a following render would correctly NOT appear in this export.
    window.toolbar_panel_combo.setCurrentIndex(1)
    window.properties_panel.refresh()
    window.properties_panel.d3_title_edit.setText("ThreeD")
    window.properties_panel._apply_3d_title()

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.format_combo.setCurrentText("SVG")
    out_path = tmp_path / "mixed.svg"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    content = out_path.read_text(encoding="utf-8")
    assert "Panel 1" in content
    assert "ThreeD" in content
    assert "Panel 3" in content
    assert re.search(r"<image[\s>]", content) is None  # genuinely vector, not a screenshot
    window.close()


# --- Workbench switching ----------------------------------------------------------


def test_switching_workbenches_preserves_normal_behavior_with_a_3d_panel(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    workbench_a_id = window._project.active_workbench_id

    window.workbench_tab_bar.new_button.click()
    assert window.plot_canvas.is_focused is False
    assert len(window.plot_canvas.axes_list) == 1  # fresh 1x1 Workbench B

    window._on_workbench_tab_selected(workbench_a_id)

    assert len(window.plot_canvas.axes_list) == 3
    assert window.figure_model.panels[1] is panel3d
    window.close()


# --- Old "Panels -> Add 3D Scatter…" menu command ------------------------------------


def test_add_3d_scatter_menu_action_opens_the_3d_sidebar_page(qapp, monkeypatch):
    window, _dataset = _make_3_panel_window()
    window.tool_drawer.show_page("data")

    window._on_add_3d_scatter_requested()

    assert window.tool_drawer.active_key == "3d"
    window.close()
