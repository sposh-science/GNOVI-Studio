"""Export behavior while a Panel is focused (see `MainWindow._focus_panel`,
`gui.dialogs.export_figure_dialog.ExportFigureDialog.
_complete_figure_export_uses_headless_path`).

Two independent guarantees under test:

1. Direct "Export Panel…" (PR #10) must keep exporting exactly the focused
   Panel, with the exact same publication geometry it would have outside
   Focus mode -- it already derives everything from the model
   (`export.figure_export.build_panel_export_figure`), never the live
   canvas, so Focus mode changes nothing about it structurally.
2. "Export Figure…" -> "Complete Figure" scope must keep meaning the
   COMPLETE underlying multi-panel Figure even while focused, even though
   the live on-screen canvas only has one Axes -- routed through the
   headless `export_figure` model path instead of the live-canvas
   `export_live_figure` specifically because of that mismatch.
"""

import re

import pandas as pd
from PIL import Image

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
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
        panel.title = f"Panel {i + 1} of 3"
        panel.add_series(PlotSeries.line(dataset, "x", "y", label=f"Series {i + 1}"))
    window._rerender()
    return window


def _focus(window, panel_index):
    window.toolbar_panel_combo.setCurrentIndex(panel_index)
    window._focus_panel(window.figure_model.active_panel)


# --- Direct Panel export while focused -------------------------------------------


def test_export_panel_while_focused_exports_only_the_focused_panel(qapp, tmp_path):
    window = _make_3_panel_window()
    _focus(window, 1)

    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.active_panel, dataset_manager=window.dataset_manager,
    )
    dialog.format_combo.setCurrentText("SVG")
    out_path = tmp_path / "panel.svg"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    content = out_path.read_text(encoding="utf-8")
    assert "Panel 2 of 3" in content
    assert "Panel 1 of 3" not in content
    assert "Panel 3 of 3" not in content
    window.close()


def test_panel_export_geometry_is_identical_focused_or_not(qapp):
    """PR #10's Panel-export geometry (`build_panel_export_figure`) is
    derived purely from the source `GnoviFigure` MODEL (layout, margins,
    page size) -- never the live canvas -- so it must come out byte-for-
    byte identical whether or not the canvas happens to be focused."""
    unfocused_window = _make_3_panel_window()
    unfocused_dialog = ExportFigureDialog(
        unfocused_window.figure_model, unfocused_window.plot_canvas, unfocused_window,
        panel=unfocused_window.figure_model.panels[1], dataset_manager=unfocused_window.dataset_manager,
    )

    focused_window = _make_3_panel_window()
    _focus(focused_window, 1)
    focused_dialog = ExportFigureDialog(
        focused_window.figure_model, focused_window.plot_canvas, focused_window,
        panel=focused_window.figure_model.active_panel, dataset_manager=focused_window.dataset_manager,
    )

    assert unfocused_dialog._panel_export_model.figure_width_in == focused_dialog._panel_export_model.figure_width_in
    assert (
        unfocused_dialog._panel_export_model.figure_height_in == focused_dialog._panel_export_model.figure_height_in
    )
    unfocused_window.close()
    focused_window.close()


def test_export_panel_action_while_focused_targets_the_focused_panel(qapp, monkeypatch):
    """End-to-end through the real menu handler (not just the dialog
    directly): "Panels -> Export Active Panel…" while focused must build
    the dialog around the focused (== active) Panel."""
    window = _make_3_panel_window()
    _focus(window, 2)
    captured = []
    real_init = ExportFigureDialog.__init__

    class _SpyDialog(ExportFigureDialog):
        def __init__(self, figure, plot_canvas, parent=None, **kwargs):
            real_init(self, figure, plot_canvas, parent, **kwargs)
            captured.append(kwargs.get("panel"))

        def exec(self):
            return None

    monkeypatch.setattr("gnovi_plot.gui.main_window.ExportFigureDialog", _SpyDialog)

    window.export_panel_action.trigger()

    assert captured == [window.figure_model.panels[2]]
    window.close()


# --- Complete Figure export while focused -----------------------------------------


def test_complete_figure_export_while_focused_exports_all_panels(qapp, tmp_path):
    window = _make_3_panel_window()
    _focus(window, 1)
    assert window.plot_canvas.is_focused is True

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dialog.scope_combo.currentText() == "Complete Figure"
    dialog.format_combo.setCurrentText("SVG")
    out_path = tmp_path / "complete.svg"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    content = out_path.read_text(encoding="utf-8")
    assert "Panel 1 of 3" in content
    assert "Panel 2 of 3" in content
    assert "Panel 3 of 3" in content
    window.close()


def test_complete_figure_export_uses_headless_path_only_when_focused(qapp):
    window = _make_3_panel_window()
    dialog_unfocused = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dialog_unfocused._complete_figure_export_uses_headless_path() is False

    _focus(window, 1)
    dialog_focused = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dialog_focused._complete_figure_export_uses_headless_path() is True
    window.close()


def test_active_panel_scope_export_while_focused_still_targets_the_focused_panel(qapp, tmp_path):
    """"Active Panel" scope (a *different*, pre-existing scope from PR
    #10's "Export Panel…") is unaffected by the headless-routing fix --
    `PlotCanvas.active_axes` already resolves to the one focused Axes."""
    window = _make_3_panel_window()
    window.show()
    from PySide6.QtWidgets import QApplication

    QApplication.instance().processEvents()
    _focus(window, 1)

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.scope_combo.setCurrentText("Active Panel")
    dialog.format_combo.setCurrentText("SVG")
    out_path = tmp_path / "active.svg"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    content = out_path.read_text(encoding="utf-8")
    assert "Panel 2 of 3" in content
    window.close()


def test_complete_figure_export_does_not_exit_focus(qapp, tmp_path):
    window = _make_3_panel_window()
    _focus(window, 1)

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.path_edit.setText(str(tmp_path / "complete.png"))
    dialog._on_accept()

    assert window.plot_canvas.is_focused is True
    assert window._current_focused_panel() is window.figure_model.panels[1]
    window.close()


def test_export_panel_while_focused_does_not_exit_focus(qapp, tmp_path):
    window = _make_3_panel_window()
    _focus(window, 1)

    dialog = ExportFigureDialog(
        window.figure_model, window.plot_canvas, window,
        panel=window.figure_model.active_panel, dataset_manager=window.dataset_manager,
    )
    dialog.path_edit.setText(str(tmp_path / "panel.png"))
    dialog._on_accept()

    assert window.plot_canvas.is_focused is True
    window.close()


def test_complete_figure_export_while_focused_is_genuinely_vector_svg(qapp, tmp_path):
    window = _make_3_panel_window()
    _focus(window, 1)

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.format_combo.setCurrentText("SVG")
    out_path = tmp_path / "complete.svg"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    content = out_path.read_text(encoding="utf-8")
    assert content.lstrip().startswith("<?xml")
    assert re.search(r"<image[\s>]", content) is None
    assert content.count("<path") > 5
    window.close()


def test_complete_figure_export_while_focused_pdf_succeeds(qapp, tmp_path):
    window = _make_3_panel_window()
    _focus(window, 1)

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.format_combo.setCurrentText("PDF")
    out_path = tmp_path / "complete.pdf"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    assert out_path.exists()
    with open(out_path, "rb") as fh:
        assert fh.read(5) == b"%PDF-"
    window.close()


def test_complete_figure_export_while_focused_raster_is_dpi_aware(qapp, tmp_path):
    window = _make_3_panel_window()
    _focus(window, 1)

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.dpi_spin.setValue(200)
    out_path = tmp_path / "complete.png"
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    with Image.open(out_path) as img:
        dpi_x, _dpi_y = img.info.get("dpi", (200, 200))
        assert dpi_x == 200 or round(dpi_x) == 200
    window.close()
