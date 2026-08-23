""""Focus Panel" / "Restore Multi-Panel View" -- temporarily showing the
SAME original Panel object alone for focused editing (see
`MainWindow._focus_panel`/`_restore_multi_panel_view`/`_current_focused_
panel`, and `PlotCanvas.render(..., focused_panel=...)`).

Deliberately NOT a Panel clone, NOT a new Workbench, NOT a linked/synced
copy: Focus is pure GUI/session view state (`MainWindow._focused_panel_ids`,
keyed by Workbench.id -> Panel.id, never serialized). Every edit made while
focused happens directly on the one real Panel object, so there is no
Apply/Discard step -- these tests exist to prove that invariant holds
through every angle (identity, rendering, undo/redo, dirty state, Workbench/
Project navigation) rather than to re-prove ordinary editing behavior
(already covered elsewhere) works at all.

Export-while-focused is covered separately in
`test_focus_panel_export_gui.py`.
"""

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog, QMessageBox

from gnovi_plot.analysis.fitting import LINEAR
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


def _make_3_panel_window():
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(4)  # "1 x 3"
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for i, panel in enumerate(window.figure_model.panels):
        panel.title = f"Panel {i + 1}"
        panel.add_series(PlotSeries.line(dataset, "x", "y", label=f"Series {i + 1}"))
    # Commits this setup as an undo checkpoint, so a single subsequent edit's
    # Undo reverts to *this* fixture state ("Panel 2", ...) rather than
    # skipping past it to the blank pre-fixture state no test cares about.
    window._on_figure_content_changed()
    return window


class _FakeAction:
    def __init__(self, text):
        self.text = text
        self.enabled = True

    def setEnabled(self, value):
        self.enabled = value


def _fake_menu(chosen_text=None, exec_calls=None):
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
            if exec_calls is not None:
                exec_calls.append(True)
            return self.actions.get(chosen_text) if chosen_text is not None else None

    return _FakeMenu


class _FakeContextMenuEvent:
    def __init__(self, inaxes=None, button=3, x=0, y=0):
        self.inaxes = inaxes
        self.button = button
        self.x = x
        self.y = y


def _focus_panel_via_context_menu(window, panel_index, monkeypatch):
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu("Focus Panel"))
    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=window.plot_canvas.axes_list[panel_index]))


def _run_fit_on_active_panel(window, dataset, model=LINEAR, label="curve"):
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label=label)])
    window.analysis_panel.model_combo.setCurrentIndex(window.analysis_panel.model_combo.findData(model))
    window.analysis_panel.run_fit_button.click()
    return window.analysis_result_view.result


# --- Basic focus: identity, rendering, no clone/Workbench/dirty ------------------


def test_focus_inactive_panel_via_context_menu_activates_and_focuses_it(qapp, monkeypatch):
    window = _make_3_panel_window()
    assert window.figure_model.active_panel_index == 0
    original_panel = window.figure_model.panels[1]

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert window.figure_model.active_panel_index == 1
    assert window._current_focused_panel() is original_panel
    window.close()


def test_focused_canvas_renders_only_the_focused_panel(qapp, monkeypatch):
    window = _make_3_panel_window()

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert window.plot_canvas.is_focused is True
    assert len(window.plot_canvas.axes_list) == 1
    assert window.plot_canvas._focused_panel_index == 1
    window.close()


def test_focus_uses_the_same_original_panel_object_not_a_clone(qapp, monkeypatch):
    window = _make_3_panel_window()
    original_panel = window.figure_model.panels[1]

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert window._current_focused_panel() is original_panel
    assert len(window.figure_model.panels) == 3  # no clone appended
    window.close()


def test_focus_does_not_change_panel_id(qapp, monkeypatch):
    window = _make_3_panel_window()
    original_id = window.figure_model.panels[1].id

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert window.figure_model.panels[1].id == original_id
    assert window._current_focused_panel().id == original_id
    window.close()


def test_focus_does_not_change_dataset_identity(qapp, monkeypatch):
    window = _make_3_panel_window()
    original_dataset = window.figure_model.panels[1].series[0].dataset

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert window._current_focused_panel().series[0].dataset is original_dataset
    window.close()


def test_focus_creates_no_new_workbench(qapp, monkeypatch):
    window = _make_3_panel_window()
    original_ids = [w.id for w in window._project.workbenches]

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert [w.id for w in window._project.workbenches] == original_ids
    window.close()


def test_focus_creates_no_new_panel(qapp, monkeypatch):
    window = _make_3_panel_window()

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert len(window.figure_model.panels) == 3
    assert window.figure_model.layout == (1, 3)  # model layout untouched
    window.close()


def test_focus_does_not_mark_project_dirty(qapp, monkeypatch):
    window = _make_3_panel_window()
    window._set_dirty(False)

    _focus_panel_via_context_menu(window, 1, monkeypatch)

    assert window._dirty is False
    window.close()


# --- Restore -----------------------------------------------------------------


def test_restore_returns_to_full_multi_panel_layout(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)

    window._restore_multi_panel_view()

    assert window.plot_canvas.is_focused is False
    assert len(window.plot_canvas.axes_list) == 3
    assert window.figure_model.layout == (1, 3)
    window.close()


def test_restore_keeps_the_focused_panel_active(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)

    window._restore_multi_panel_view()

    assert window.figure_model.active_panel_index == 1
    window.close()


def test_restore_does_not_change_panel_id(qapp, monkeypatch):
    window = _make_3_panel_window()
    original_id = window.figure_model.panels[1].id
    _focus_panel_via_context_menu(window, 1, monkeypatch)

    window._restore_multi_panel_view()

    assert window.figure_model.panels[1].id == original_id
    window.close()


def test_restore_does_not_mark_project_dirty(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    window._set_dirty(False)

    window._restore_multi_panel_view()

    assert window._dirty is False
    window.close()


# --- Edit persistence: edits land directly on the original Panel ----------------


def test_title_change_while_focused_persists_after_restore(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)

    window.figure_model.active_panel.title = "Ferricyanide CV"
    window._on_figure_content_changed()
    window._restore_multi_panel_view()

    assert window.figure_model.panels[1].title == "Ferricyanide CV"
    window.close()


def test_axis_limits_change_while_focused_persists_after_restore(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)

    window.figure_model.active_panel.xlim = (0.5, 5.0)
    window.figure_model.active_panel.xscale = "log"
    window._on_figure_content_changed()
    window._restore_multi_panel_view()

    assert window.figure_model.panels[1].xlim == (0.5, 5.0)
    assert window.figure_model.panels[1].xscale == "log"
    window.close()


def test_styling_change_while_focused_persists_after_restore(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)

    window.figure_model.active_panel.series[0].color = "#ff00ff"
    window.figure_model.active_panel.grid = True
    window._on_figure_content_changed()
    window._restore_multi_panel_view()

    assert window.figure_model.panels[1].series[0].color == "#ff00ff"
    assert window.figure_model.panels[1].grid is True
    window.close()


def test_series_visibility_change_while_focused_persists_after_restore(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)

    window.figure_model.active_panel.series[0].visible = False
    window._on_figure_content_changed()
    window._restore_multi_panel_view()

    assert window.figure_model.panels[1].series[0].visible is False
    window.close()


def test_fit_performed_while_focused_is_retained_after_restore(qapp, monkeypatch):
    window = _make_3_panel_window()
    window.toolbar_panel_combo.setCurrentIndex(1)
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    original_panel_id = window.figure_model.panels[1].id
    fit_dataset = _make_dataset("for-fit")
    fit_dataset.dataframe = pd.DataFrame({"x": np.arange(20.0), "y": 3.0 * np.arange(20.0) + 2.0})
    window.dataset_manager.add(fit_dataset)

    result = _run_fit_on_active_panel(window, fit_dataset)
    window._restore_multi_panel_view()

    workbench = window._project.active_workbench
    current = workbench.analysis_results.current(original_panel_id)
    assert current is not None
    assert current.result_id == result.result_id
    window.close()


def test_fit_curve_added_while_focused_is_retained_after_restore(qapp, monkeypatch):
    window = _make_3_panel_window()
    window.toolbar_panel_combo.setCurrentIndex(1)
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    fit_dataset = _make_dataset("for-fit")
    fit_dataset.dataframe = pd.DataFrame({"x": np.arange(20.0), "y": 3.0 * np.arange(20.0) + 2.0})
    window.dataset_manager.add(fit_dataset)
    result = _run_fit_on_active_panel(window, fit_dataset)
    window.analysis_panel.add_fit_curve_button.click()

    window._restore_multi_panel_view()

    fit_series = [
        s for s in window.figure_model.panels[1].series if s.dataset.metadata.get("result_id") == result.result_id
    ]
    assert len(fit_series) == 1
    window.close()


def test_analysis_history_stays_keyed_to_the_original_panel_id_while_focused(qapp, monkeypatch):
    window = _make_3_panel_window()
    window.toolbar_panel_combo.setCurrentIndex(1)
    original_panel_id = window.figure_model.panels[1].id
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    fit_dataset = _make_dataset("for-fit")
    fit_dataset.dataframe = pd.DataFrame({"x": np.arange(20.0), "y": 3.0 * np.arange(20.0) + 2.0})
    window.dataset_manager.add(fit_dataset)

    result = _run_fit_on_active_panel(window, fit_dataset)

    workbench = window._project.active_workbench
    assert workbench.analysis_results.current(original_panel_id).result_id == result.result_id
    assert result.source_panel_id == original_panel_id
    window.close()


# --- Undo/redo: same existing mechanism, no Focus-specific stack -----------------


def test_normal_undo_works_while_focused(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    original_title = window.figure_model.panels[1].title
    window.figure_model.active_panel.title = "Changed"
    window._on_figure_content_changed()
    assert window.figure_model.panels[1].title == "Changed"

    window._on_undo()

    assert window.figure_model.panels[1].title == original_title
    assert window.plot_canvas.is_focused is True  # undo doesn't exit Focus
    window.close()


def test_redo_works_while_focused(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    window.figure_model.active_panel.title = "Changed"
    window._on_figure_content_changed()
    window._on_undo()

    window._on_redo()

    assert window.figure_model.panels[1].title == "Changed"
    window.close()


def test_restore_does_not_break_undo_history(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    original_title = window.figure_model.panels[1].title
    window.figure_model.active_panel.title = "Changed"
    window._on_figure_content_changed()

    window._restore_multi_panel_view()
    window._on_undo()

    assert window.figure_model.panels[1].title == original_title
    window.close()


def test_entering_and_restoring_focus_are_not_themselves_undoable(qapp, monkeypatch):
    window = _make_3_panel_window()
    can_undo_before = window.undo_action.isEnabled()

    _focus_panel_via_context_menu(window, 1, monkeypatch)
    assert window.undo_action.isEnabled() == can_undo_before
    window._restore_multi_panel_view()
    assert window.undo_action.isEnabled() == can_undo_before
    window.close()


# --- Workbench / Project navigation ----------------------------------------------


def test_switching_workbench_does_not_leak_focus_state(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    workbench_a_id = window._project.active_workbench_id

    window.workbench_tab_bar.new_button.click()  # Workbench B, becomes active

    assert window.plot_canvas.is_focused is False  # B renders normally
    assert workbench_a_id in window._focused_panel_ids  # A's own session focus is untouched
    window.close()


def test_returning_to_a_focused_workbench_restores_its_session_focus(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    workbench_a_id = window._project.active_workbench_id
    window.workbench_tab_bar.new_button.click()  # switch to new Workbench B
    assert window.plot_canvas.is_focused is False

    window._on_workbench_tab_selected(workbench_a_id)

    assert window.plot_canvas.is_focused is True
    assert window.figure_model.active_panel_index == 1
    window.close()


def test_closing_a_focused_workbench_leaves_no_stale_focus_state(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    workbench_a_id = window._project.active_workbench_id
    window.workbench_tab_bar.new_button.click()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))

    window._on_delete_workbench_requested(workbench_a_id)

    assert workbench_a_id not in window._focused_panel_ids
    window.close()


def test_new_project_clears_focus_state(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))

    window.new_project_action.trigger()

    assert window._focused_panel_ids == {}
    assert window.plot_canvas.is_focused is False
    window.close()


def test_open_project_clears_old_focus_state(qapp, monkeypatch, tmp_path):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    out_path = tmp_path / "other.gnovi"
    other_window = MainWindow()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    other_window.save_project_as_action.trigger()
    other_window.close()

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert window._focused_panel_ids == {}
    assert window.plot_canvas.is_focused is False
    window.close()


def test_save_while_focused_serializes_the_full_figure_with_focused_edits(qapp, monkeypatch, tmp_path):
    from gnovi_plot.core.project_io import load_project

    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    window.figure_model.active_panel.title = "Edited While Focused"
    window._on_figure_content_changed()

    out_path = tmp_path / "focused.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    reloaded = load_project(out_path)
    assert reloaded.workbenches[0].figure.layout == (1, 3)
    assert len(reloaded.workbenches[0].figure.panels) == 3
    assert reloaded.workbenches[0].figure.panels[1].title == "Edited While Focused"
    window.close()


def test_reopen_after_save_while_focused_displays_normal_multi_panel_layout(qapp, monkeypatch, tmp_path):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    window.figure_model.active_panel.title = "Edited While Focused"
    window._on_figure_content_changed()
    out_path = tmp_path / "focused.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Discard))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert window.plot_canvas.is_focused is False
    assert window.figure_model.layout == (1, 3)
    assert window.figure_model.panels[1].title == "Edited While Focused"
    window.close()


# --- Interaction: menu/context-menu wording, targeting, single-panel, stale ------


def test_focused_context_menu_offers_restore_multi_panel_view(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    captured = {}

    def _fake_menu_capturing(*_a, **_k):
        class _FakeMenu:
            def __init__(self, *_a2, **_k2):
                self.actions = {}

            def addSeparator(self):
                pass

            def addAction(self, text):
                action = _FakeAction(text)
                self.actions[text] = action
                captured[text] = action
                return action

            def exec(self, *_a2, **_k2):
                return None

        return _FakeMenu()

    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu_capturing)
    axes = window.plot_canvas.axes_list[0]  # the one visible (focused) Axes

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=axes))

    assert "Restore Multi-Panel View" in captured
    assert "Focus Panel" not in captured
    window.close()


def test_panels_menu_focus_action_toggles_text_between_focus_and_restore(qapp, monkeypatch):
    window = _make_3_panel_window()
    window._sync_focus_panel_action_state()
    assert window.focus_panel_action.text() == "Focus Active Panel"

    _focus_panel_via_context_menu(window, 1, monkeypatch)
    window._sync_focus_panel_action_state()
    assert window.focus_panel_action.text() == "Restore Multi-Panel View"

    window.focus_panel_action.trigger()
    window._sync_focus_panel_action_state()
    assert window.focus_panel_action.text() == "Focus Active Panel"
    assert window.plot_canvas.is_focused is False
    window.close()


def test_panels_menu_focus_action_enters_focus_on_the_active_panel(qapp):
    window = _make_3_panel_window()
    window.toolbar_panel_combo.setCurrentIndex(2)

    window.focus_panel_action.trigger()

    assert window.plot_canvas.is_focused is True
    assert window._current_focused_panel() is window.figure_model.panels[2]
    window.close()


def test_right_click_mapping_maps_the_single_focused_axes_back_to_the_original_panel(qapp, monkeypatch):
    """Focused on Panel 3 (not Panel 1): right-clicking the one visible
    Axes and choosing Export must act on Panel 3, proving the Axes->Panel
    mapping doesn't assume "the one Axes is always index 0"."""
    window = _make_3_panel_window()
    panel_3_id = window.figure_model.panels[2].id
    _focus_panel_via_context_menu(window, 2, monkeypatch)
    captured = []
    original = window._on_export_panel_requested
    window._on_export_panel_requested = lambda: captured.append(window.figure_model.active_panel.id) or original()
    monkeypatch.setattr("gnovi_plot.gui.main_window.ExportFigureDialog", lambda *a, **k: type("D", (), {"exec": lambda self: None})())
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu("Export Panel…"))
    axes = window.plot_canvas.axes_list[0]  # the one focused Axes

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=axes))

    assert captured == [panel_3_id]
    window.close()


def test_focus_action_disabled_for_a_single_panel_figure(qapp):
    window = MainWindow()  # default 1x1
    window._sync_focus_panel_action_state()

    assert window.focus_panel_action.isEnabled() is False
    window.close()


def test_focus_context_menu_item_disabled_for_a_single_panel_figure(qapp, monkeypatch):
    window = MainWindow()  # default 1x1
    captured = {}

    def _fake_menu_capturing(*_a, **_k):
        class _FakeMenu:
            def __init__(self, *_a2, **_k2):
                self.actions = {}

            def addSeparator(self):
                pass

            def addAction(self, text):
                action = _FakeAction(text)
                captured[text] = action
                return action

            def exec(self, *_a2, **_k2):
                return None

        return _FakeMenu()

    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu_capturing)

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=window.plot_canvas.axes_list[0]))

    assert captured["Focus Panel"].enabled is False
    window.close()


def test_stale_focused_panel_reference_is_cleared_defensively(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 2, monkeypatch)  # focus Panel 3
    assert window.plot_canvas.is_focused is True

    window.figure_size_panel.layout_combo.setCurrentIndex(0)  # shrink to "1 x 1" -- drops Panel 3

    assert window._current_focused_panel() is None
    assert window.plot_canvas.is_focused is False
    assert window._current_workbench_id not in window._focused_panel_ids
    window.close()


def test_double_click_while_focused_restores_multi_panel_view(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    axes = window.plot_canvas.axes_list[0]

    class _DblClickEvent(_FakeContextMenuEvent):
        def __init__(self, inaxes):
            super().__init__(inaxes=inaxes, button=1)
            self.dblclick = True

    window._on_canvas_click(_DblClickEvent(inaxes=axes))

    assert window.plot_canvas.is_focused is False
    window.close()


def test_double_click_when_not_focused_does_not_restore_anything(qapp):
    window = _make_3_panel_window()
    axes = window.plot_canvas.axes_list[1]

    class _DblClickEvent(_FakeContextMenuEvent):
        def __init__(self, inaxes):
            super().__init__(inaxes=inaxes, button=1)
            self.dblclick = True

    window._on_canvas_click(_DblClickEvent(inaxes=axes))  # must not raise

    assert window.figure_model.active_panel_index == 1  # ordinary click-to-activate still ran
    window.close()


# --- Regression: unrelated existing behavior is unaffected -----------------------


def test_extract_panel_still_behaves_independently_of_focus(qapp, monkeypatch):
    window = _make_3_panel_window()
    _focus_panel_via_context_menu(window, 1, monkeypatch)
    original_panel_id = window.figure_model.panels[1].id

    window._on_extract_panel_requested()

    assert len(window._project.workbenches) == 2
    source = window._project.workbenches[0]
    assert source.figure.panels[1].id == original_panel_id  # original untouched
    window.close()


def test_full_figure_rendering_unchanged_outside_focus_mode(qapp):
    window = _make_3_panel_window()

    assert window.plot_canvas.is_focused is False
    assert len(window.plot_canvas.axes_list) == 3
    window.close()
