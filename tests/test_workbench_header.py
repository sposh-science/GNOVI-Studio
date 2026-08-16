"""`WorkbenchHeader` -- the small, restrained application-chrome strip
docked above the central plotting/figure work area (the "Workbench"; see
`gui.widgets.workbench_header` and `gui.main_window.MainWindow`).

GUI-only application chrome, same guarantee as the existing active-panel
badge (`tests/test_active_panel_badge.py`): a plain Qt widget, never a
Matplotlib artist, so it structurally cannot appear in `GnoviFigure`/`Panel`
state, in Matplotlib's own `figure.savefig()`, or in any
`export.figure_export` format.
"""

import dataclasses

import pandas as pd
import pytest

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.export.figure_export import export_figure
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.workbench_header import WorkbenchHeader
from gnovi_plot.plotting.figure import GnoviFigure, Panel
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


# --- Label content -----------------------------------------------------------


def test_header_shows_workbench_name_and_layout(qapp):
    header = WorkbenchHeader("CV Comparison", GnoviFigure())
    assert header.label.text() == "WORKBENCH · CV Comparison · 1 × 1"


def test_header_shows_the_current_layout(qapp):
    figure = GnoviFigure()
    figure.set_layout(2, 2)

    header = WorkbenchHeader("CV Comparison", figure)

    assert header.label.text() == "WORKBENCH · CV Comparison · 2 × 2"


def test_refresh_reloads_name_and_layout(qapp):
    figure = GnoviFigure()
    header = WorkbenchHeader("CV Comparison", figure)

    figure.set_layout(1, 3)
    header.refresh("New Scan", figure)

    assert header.label.text() == "WORKBENCH · New Scan · 1 × 3"


def test_header_does_not_duplicate_the_active_panel_badge(qapp):
    """The badge ("P1 · ACTIVE") already carries which-panel-is-active
    information -- the Workbench header must never repeat it."""
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    figure.set_active_panel(2)

    header = WorkbenchHeader("CV Comparison", figure)

    text = header.label.text()
    assert "ACTIVE" not in text
    assert "P3" not in text
    assert "Panel" not in text


# --- Wired into MainWindow, tracks the real figure and active Workbench -------


def test_main_window_hosts_a_workbench_header(qapp):
    window = MainWindow()
    assert window.workbench_header.label.text() == "WORKBENCH · Workbench 1 · 1 × 1"
    window.close()


@pytest.mark.parametrize("index,expected", [(0, "1 × 1"), (1, "1 × 2"), (3, "2 × 2"), (5, "2 × 3")])
def test_workbench_header_updates_when_layout_changes(qapp, index, expected):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(index)
    assert window.workbench_header.label.text() == f"WORKBENCH · Workbench 1 · {expected}"
    window.close()


def test_workbench_header_updates_after_project_load(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    out_path = tmp_path / "workbench_header.gnovi"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.save_project_as_action.trigger()

    window.new_project_action.trigger()
    assert window.workbench_header.label.text() == "WORKBENCH · Workbench 1 · 1 × 1"

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out_path), "")))
    window.open_project_action.trigger()

    assert window.workbench_header.label.text() == "WORKBENCH · Workbench 1 · 2 × 2"
    window.close()


def test_workbench_header_updates_on_undo_redo(qapp):
    window = MainWindow()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    assert window.workbench_header.label.text() == "WORKBENCH · Workbench 1 · 2 × 2"

    window._on_undo()

    assert window.workbench_header.label.text() == "WORKBENCH · Workbench 1 · 1 × 1"

    window._on_redo()

    assert window.workbench_header.label.text() == "WORKBENCH · Workbench 1 · 2 × 2"
    window.close()


# --- The label is application chrome only -- absent from the model/exports ----


def test_workbench_header_is_not_gnovi_figure_or_panel_state():
    figure_fields = {f.name for f in dataclasses.fields(Panel)}
    assert not any("workbench" in name.lower() for name in figure_fields)
    assert not hasattr(GnoviFigure(), "workbench_header")
    assert not hasattr(GnoviFigure(), "workbench")


def test_workbench_header_text_is_not_serialized_in_to_dict():
    figure = GnoviFigure()
    figure.set_layout(2, 2)

    data = figure.to_dict()

    assert "workbench" not in str(data).lower()


def test_workbench_header_is_not_a_child_of_the_live_wysiwyg_canvas(qapp):
    """`window.plot_canvas.figure` is GNOVI's own WYSIWYG export source
    (the navigation toolbar's "Save" and Export Figure's "Complete Figure"
    both save it directly) -- confirms structurally that the Workbench
    header is a sibling Qt widget in the splitter, never a child of the
    canvas, so it cannot end up in that Figure."""
    window = MainWindow()
    assert not window.plot_canvas.isAncestorOf(window.workbench_header)
    assert window.workbench_header.parent() is not window.plot_canvas
    window.close()


def test_workbench_header_never_appears_in_any_export_format(qapp, tmp_path):
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    dataset = _make_dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))

    for fmt in ("png", "tiff", "svg", "pdf"):
        out_path = tmp_path / f"export.{fmt}"
        export_figure(figure, out_path, dpi=150)
        content = out_path.read_bytes()
        assert b"WORKBENCH" not in content
