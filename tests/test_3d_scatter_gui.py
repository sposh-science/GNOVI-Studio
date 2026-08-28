"""3D Scatter end-to-end GUI workflows: creation (`Add3DScatterDialog` via
"Panels -> Add 3D Scatter…"), active-panel selection/right-click targeting,
Focus/Restore, Extract, direct Panel export, full-Figure export, and
Workbench switching -- all reusing the exact same handlers/domain
operations PR #9/#10/#11 already established for 2D Panels (see each
handler's own docstring in `gui.main_window`); these tests exist to prove
Panel3D integrates with them, not to re-prove those mechanisms work at all.
"""

import re

import pandas as pd
from PySide6.QtWidgets import QDialog

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.dialogs.add_3d_scatter_dialog import Add3DScatterDialog
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.figure import Panel3D
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="mat"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0, 300.0], "composition": [0.1, 0.1, 0.1, 0.2], "conductivity": [2.4, 2.9, 3.5, 3.1]}
    )
    return Dataset(name=name, dataframe=df)


def _make_3_panel_window():
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(4)  # "1 x 3"
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
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


def _make_3d_panel_at(window, dataset, index, monkeypatch, **overrides):
    """Drives the real "Panels -> Add 3D Scatter…" handler end to end,
    with `Add3DScatterDialog` itself faked (a real dialog can't be
    `.exec()`d headlessly) -- everything AFTER `dialog.exec()` is the real
    handler code."""
    window.toolbar_panel_combo.setCurrentIndex(index)

    class _FakeDialog:
        Accepted = QDialog.Accepted

        def __init__(self, *_a, **_k):
            self.dataset = dataset
            self.x_column = overrides.get("x_column", "temperature")
            self.y_column = overrides.get("y_column", "composition")
            self.z_column = overrides.get("z_column", "conductivity")
            self.title = overrides.get("title", "3D Scatter")
            self.x_label = overrides.get("x_label", "T")
            self.y_label = overrides.get("y_label", "Comp")
            self.z_label = overrides.get("z_label", "Cond")
            self.marker = overrides.get("marker", "o")
            self.marker_size = overrides.get("marker_size", 6.0)

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr("gnovi_plot.gui.main_window.Add3DScatterDialog", _FakeDialog)
    window._on_add_3d_scatter_requested()
    return window.figure_model.panels[index]


# --- Creation ----------------------------------------------------------------------


def test_add_3d_scatter_replaces_the_active_panel_with_a_panel3d(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()

    panel = _make_3d_panel_at(window, dataset, 1, monkeypatch)

    assert isinstance(panel, Panel3D)
    assert panel.title == "3D Scatter"
    assert panel.series[0].x_column == "temperature"
    assert panel.series[0].z_column == "conductivity"
    window.close()


def test_add_3d_scatter_marks_the_project_dirty(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    window._set_dirty(False)

    _make_3d_panel_at(window, dataset, 1, monkeypatch)

    assert window._dirty is True
    window.close()


def test_add_3d_scatter_does_not_change_the_layout_or_panel_count(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()

    _make_3d_panel_at(window, dataset, 1, monkeypatch)

    assert window.figure_model.layout == (1, 3)
    assert len(window.figure_model.panels) == 3
    window.close()


def test_reinvoking_add_3d_scatter_on_an_existing_panel3d_edits_it_in_place(qapp, monkeypatch):
    window, dataset = _make_3_panel_window()
    panel = _make_3d_panel_at(window, dataset, 1, monkeypatch)
    original_id = panel.id

    edited = _make_3d_panel_at(window, dataset, 1, monkeypatch, title="Renamed")

    assert edited is panel  # mutated in place, never replaced
    assert edited.id == original_id
    assert edited.title == "Renamed"
    window.close()


def test_invalid_numeric_data_shows_a_controlled_error_not_a_crash(qapp, monkeypatch):
    window, _dataset = _make_3_panel_window()
    bad_dataset = Dataset(name="bad", dataframe=pd.DataFrame({"a": ["x", "y"], "b": ["x", "y"], "c": ["x", "y"]}))
    window.dataset_manager.add(bad_dataset)
    window.toolbar_panel_combo.setCurrentIndex(1)

    class _FakeDialog:
        Accepted = QDialog.Accepted

        def __init__(self, *_a, **_k):
            self.dataset = bad_dataset
            self.x_column, self.y_column, self.z_column = "a", "b", "c"
            self.title = self.x_label = self.y_label = self.z_label = ""
            self.marker, self.marker_size = "o", 6.0

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr("gnovi_plot.gui.main_window.Add3DScatterDialog", _FakeDialog)
    warnings = []
    monkeypatch.setattr(
        "gnovi_plot.gui.main_window.QMessageBox.critical",
        staticmethod(lambda *a, **k: warnings.append(True)),
    )

    window._on_add_3d_scatter_requested()  # must not raise

    assert warnings == [True]
    assert not isinstance(window.figure_model.panels[1], Panel3D)  # unchanged
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
    panel3d = _make_3d_panel_at(window, dataset, 1, monkeypatch, title="ThreeD")
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
    _make_3d_panel_at(window, dataset, 1, monkeypatch, title="ThreeD")

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
