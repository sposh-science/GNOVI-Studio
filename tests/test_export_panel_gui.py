""""Export Panel…" -- direct single-Panel publication export, reachable
from the Panel right-click context menu and "Panels -> Export Active
Panel…" (see `MainWindow._on_export_panel_requested`/
`_show_panel_context_menu`).

Both entry points must converge on the exact same handler, which opens the
same `ExportFigureDialog` class "Export Figure…" uses (constructed with
`panel=`/`dataset_manager=` so it exports via the headless transient-Figure
path, see `export.figure_export.build_panel_export_figure`/`export_panel`)
-- never a second export implementation, never a live-canvas screenshot,
never any Project/Workbench/Dataset/analysis-history mutation.

Mirrors `test_panel_context_menu_gui.py`'s style for targeting/reuse
assertions and `test_export_wysiwyg_parity.py`'s `_gnovi_export_bytes`-style
pattern (`dialog._on_accept()` directly, never a real modal `.exec()`) for
driving the dialog in tests.
"""

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFileDialog

from gnovi_plot.analysis.fitting import LINEAR
from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.plotting.figure import PlotTheme
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
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    window._rerender()
    return window, dataset


class _FakeAction:
    def __init__(self, text):
        self.text = text


def _fake_menu(chosen_text):
    class _FakeMenu:
        def __init__(self, *_a, **_k):
            self.actions = {}

        def addAction(self, text):
            action = _FakeAction(text)
            self.actions[text] = action
            return action

        def addSeparator(self):
            pass

        def exec(self, *_a, **_k):
            return self.actions.get(chosen_text) if chosen_text is not None else None

    return _FakeMenu


class _FakeContextMenuEvent:
    def __init__(self, inaxes=None, button=3, x=0, y=0):
        self.inaxes = inaxes
        self.button = button
        self.x = x
        self.y = y


def _spy_on_dialog_construction(monkeypatch):
    """Patches `ExportFigureDialog` in `main_window`'s namespace with a
    subclass that records every `panel=`/`figure=` it was constructed
    with, then skips `.exec()` (via `_on_export_panel_requested` calling
    `dialog.exec()` -- patched here to a no-op) so no modal event loop
    blocks the test. Returns the list of captured `(figure, panel)` pairs."""
    captured = []
    real_init = ExportFigureDialog.__init__

    class _SpyDialog(ExportFigureDialog):
        def __init__(self, figure, plot_canvas, parent=None, **kwargs):
            real_init(self, figure, plot_canvas, parent, **kwargs)
            captured.append((figure, kwargs.get("panel")))

        def exec(self):
            return None

    monkeypatch.setattr("gnovi_plot.gui.main_window.ExportFigureDialog", _SpyDialog)
    return captured


# --- Targeting: right-click / Panels menu resolve the correct Panel -------------


def test_right_click_export_targets_the_clicked_panel(qapp, monkeypatch):
    window, _dataset = _make_3_panel_window()
    assert window.figure_model.active_panel_index == 0
    captured = _spy_on_dialog_construction(monkeypatch)
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu("Export Panel…"))
    clicked_axes = window.plot_canvas.axes_list[2]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert window.figure_model.active_panel_index == 2
    assert len(captured) == 1
    _figure, panel = captured[0]
    assert panel is window.figure_model.panels[2]
    window.close()


def test_right_click_on_active_panel_export_still_opens_the_menu(qapp, monkeypatch):
    window, _dataset = _make_3_panel_window()
    captured = _spy_on_dialog_construction(monkeypatch)
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu("Export Panel…"))
    active_axes = window.plot_canvas.axes_list[window.figure_model.active_panel_index]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=active_axes))

    assert len(captured) == 1
    assert captured[0][1] is window.figure_model.panels[0]
    window.close()


def test_panels_menu_export_targets_the_active_panel(qapp, monkeypatch):
    window, _dataset = _make_3_panel_window()
    window.toolbar_panel_combo.setCurrentIndex(2)
    captured = _spy_on_dialog_construction(monkeypatch)

    window.export_panel_action.trigger()

    assert len(captured) == 1
    assert captured[0][1] is window.figure_model.panels[2]
    window.close()


def test_context_menu_and_panels_menu_call_the_same_handler(qapp, monkeypatch):
    """No duplicate export domain path: both entry points must invoke the
    exact same bound `_on_export_panel_requested` method."""
    window, _dataset = _make_3_panel_window()
    calls = []
    original = window._on_export_panel_requested

    def _spy():
        calls.append(True)
        original()

    window._on_export_panel_requested = _spy
    monkeypatch.setattr("gnovi_plot.gui.main_window.ExportFigureDialog", lambda *a, **k: type("D", (), {"exec": lambda self: None})())
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu("Export Panel…"))

    window.export_panel_action.trigger()
    clicked_axes = window.plot_canvas.axes_list[1]
    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert calls == [True, True]
    window.close()


def test_right_click_cancel_does_not_open_export_dialog(qapp, monkeypatch):
    window, _dataset = _make_3_panel_window()
    captured = _spy_on_dialog_construction(monkeypatch)
    monkeypatch.setattr("gnovi_plot.gui.main_window.QMenu", _fake_menu(None))
    clicked_axes = window.plot_canvas.axes_list[1]

    window._on_canvas_context_menu(_FakeContextMenuEvent(inaxes=clicked_axes))

    assert captured == []
    window.close()


# --- Zero Project mutation --------------------------------------------------------


def test_export_does_not_mark_project_dirty(qapp, monkeypatch, tmp_path):
    window, _dataset = _make_3_panel_window()
    window._set_dirty(False)
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )
    dialog.path_edit.setText(str(tmp_path / "panel.png"))

    dialog._on_accept()

    assert window._dirty is False
    window.close()


def test_export_does_not_touch_undo_redo_state(qapp, tmp_path):
    window, _dataset = _make_3_panel_window()
    can_undo_before = window.undo_action.isEnabled()
    can_redo_before = window.redo_action.isEnabled()
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )
    dialog.path_edit.setText(str(tmp_path / "panel.png"))

    dialog._on_accept()

    assert window.undo_action.isEnabled() == can_undo_before
    assert window.redo_action.isEnabled() == can_redo_before
    window.close()


def test_export_creates_no_new_workbench(qapp, tmp_path):
    window, _dataset = _make_3_panel_window()
    original_ids = [w.id for w in window._project.workbenches]
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )
    dialog.path_edit.setText(str(tmp_path / "panel.png"))

    dialog._on_accept()

    assert [w.id for w in window._project.workbenches] == original_ids
    window.close()


def test_source_panel_and_workbench_unchanged_after_export(qapp, tmp_path):
    window, _dataset = _make_3_panel_window()
    original_titles = [p.title for p in window.figure_model.panels]
    original_ids = [p.id for p in window.figure_model.panels]
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )
    dialog.path_edit.setText(str(tmp_path / "panel.png"))

    dialog._on_accept()

    assert window.figure_model.layout == (1, 3)
    assert [p.title for p in window.figure_model.panels] == original_titles
    assert [p.id for p in window.figure_model.panels] == original_ids
    window.close()


def test_dataset_identity_is_shared_in_the_transient_clone(qapp, tmp_path):
    window, dataset = _make_3_panel_window()
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )

    assert dialog._panel_export_model.panels[0].series[0].dataset is dataset
    window.close()


# --- Analysis history / fit curves -------------------------------------------------


def _run_fit_on_active_panel(window, dataset, model=LINEAR, label="curve"):
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y", label=label)])
    window.analysis_panel.model_combo.setCurrentIndex(window.analysis_panel.model_combo.findData(model))
    window.analysis_panel.run_fit_button.click()
    return window.analysis_result_view.result


def test_fit_derived_curve_and_result_id_metadata_appear_unchanged_in_the_export(qapp, tmp_path):
    window, _dataset = _make_3_panel_window()
    fit_dataset = _make_dataset("for-fit")
    fit_dataset.dataframe = pd.DataFrame({"x": np.arange(20.0), "y": 3.0 * np.arange(20.0) + 2.0})
    window.dataset_manager.add(fit_dataset)
    result = _run_fit_on_active_panel(window, fit_dataset)
    window.analysis_panel.add_fit_curve_button.click()
    fit_series = next(
        s for s in window.figure_model.active_panel.series if s.dataset.metadata.get("result_id") == result.result_id
    )
    original_metadata = dict(fit_series.dataset.metadata)

    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.active_panel, dataset_manager=window.dataset_manager,
    )

    exported_fit_series = next(
        s
        for s in dialog._panel_export_model.panels[0].series
        if s.dataset.metadata.get("result_id") == result.result_id
    )
    assert exported_fit_series.dataset is fit_series.dataset
    assert fit_series.dataset.metadata == original_metadata
    window.close()


# --- Formats / DPI / transparency / theme ------------------------------------------


def test_png_svg_pdf_tiff_export_all_succeed_via_the_dialog(qapp, tmp_path):
    window, _dataset = _make_3_panel_window()
    for fmt, ext in [("PNG", "png"), ("SVG", "svg"), ("PDF", "pdf"), ("TIFF", "tiff")]:
        dialog = ExportFigureDialog(
            window.figure_model, window.plot_canvas, window,
            panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
        )
        dialog.format_combo.setCurrentText(fmt)
        out_path = tmp_path / f"panel.{ext}"
        dialog.path_edit.setText(str(out_path))
        dialog._on_accept()
        assert out_path.exists(), f"{fmt} export failed"
    window.close()


def test_transparency_option_produces_an_alpha_channel_png(qapp, tmp_path):
    from PIL import Image

    window, _dataset = _make_3_panel_window()
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )
    dialog.background_combo.setCurrentText("Transparent")
    out_path = tmp_path / "panel.png"
    dialog.path_edit.setText(str(out_path))

    dialog._on_accept()

    with Image.open(out_path) as img:
        assert img.mode in ("RGBA", "LA", "PA")
    window.close()


def test_dark_theme_project_exports_a_dark_panel(qapp, tmp_path):
    from PIL import Image

    window, _dataset = _make_3_panel_window()
    window._on_theme_changed(PlotTheme.DARK)
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )
    out_path = tmp_path / "panel.png"
    dialog.path_edit.setText(str(out_path))

    dialog._on_accept()

    with Image.open(out_path) as img:
        corner = img.convert("RGB").getpixel((0, 0))
    assert corner != (255, 255, 255)
    window.close()


# --- Geometry: not stretched to the full multi-panel Figure size ----------------


def test_pixel_size_label_reflects_panel_geometry_not_the_full_figure_size(qapp):
    window, _dataset = _make_3_panel_window()
    figure_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    panel_dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )

    figure_width_in, _h = figure_dialog._current_physical_size_in()
    panel_width_in, _h2 = panel_dialog._current_physical_size_in()

    assert panel_width_in < figure_width_in
    window.close()


def test_single_panel_1x1_workbench_export_works(qapp, tmp_path):
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window.figure_model.add_series(PlotSeries.line(dataset, "x", "y"))
    window._rerender()
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.active_panel, dataset_manager=window.dataset_manager,
    )
    out_path = tmp_path / "panel.png"
    dialog.path_edit.setText(str(out_path))

    dialog._on_accept()

    assert out_path.exists()
    window.close()


# --- Repetition / persistence -----------------------------------------------------


def test_repeated_export_produces_no_state_mutation(qapp, tmp_path):
    window, _dataset = _make_3_panel_window()
    original_ids = [p.id for p in window.figure_model.panels]

    for i in range(2):
        dialog = ExportFigureDialog(
            window.figure_model, window.plot_canvas, window,
            panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
        )
        dialog.path_edit.setText(str(tmp_path / f"panel_{i}.png"))
        dialog._on_accept()

    assert [p.id for p in window.figure_model.panels] == original_ids
    assert len(window._project.workbenches) == 1
    window.close()


def test_save_reopen_after_export_contains_no_export_only_state(qapp, monkeypatch, tmp_path):
    window, _dataset = _make_3_panel_window()
    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.panels[1], dataset_manager=window.dataset_manager,
    )
    dialog.path_edit.setText(str(tmp_path / "panel.png"))
    dialog._on_accept()

    out_path = tmp_path / "project.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    from gnovi_plot.core.project_io import load_project

    reloaded = load_project(out_path)
    assert len(reloaded.workbenches) == 1
    assert reloaded.workbenches[0].figure.layout == (1, 3)
    window.close()


# --- Content: labels/title/legend/grid preserved ----------------------------------


def test_labels_title_legend_grid_are_preserved_in_the_export_model(qapp):
    window, _dataset = _make_3_panel_window()
    panel = window.figure_model.panels[1]
    panel.title = "Ferricyanide CV"
    panel.xlabel = "E / V"
    panel.ylabel = "i / mA"
    panel.grid = True
    panel.legend_visible = True

    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=panel, dataset_manager=window.dataset_manager,
    )
    exported = dialog._panel_export_model.panels[0]

    assert exported.title == "Ferricyanide CV"
    assert exported.xlabel == "E / V"
    assert exported.ylabel == "i / mA"
    assert exported.grid is True
    assert exported.legend_visible is True
    window.close()
