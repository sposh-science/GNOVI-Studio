"""GUI-only active-panel indicator (`PlotCanvas._ActivePanelBadge`) --
replaces the old spine/border highlight (see `tests/test_plot_canvas_polish.py`
for the "spines never change" smoke checks). A badge is a plain Qt child
widget of `PlotCanvas`, never a Matplotlib artist, so it structurally cannot
appear in the Matplotlib Figure itself -- these tests confirm that
end-to-end: it's absent from `GnoviFigure`/`Panel` state, from Matplotlib's
own `figure.savefig()` (what the navigation toolbar's "Save" button calls),
and from every `export.figure_export` format.
"""

import dataclasses
import io

import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import MouseEvent
from PIL import Image

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.export.figure_export import export_figure
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS
from gnovi_plot.gui.styles import ACTIVE_PANEL_BADGE_COLOR
from gnovi_plot.gui.widgets.plot_canvas import PlotCanvas
from gnovi_plot.plotting.figure import GnoviFigure, Panel
from gnovi_plot.plotting.series import PlotSeries

_LAYOUT_TEXT = {
    (1, 1): "1 x 1",
    (1, 2): "1 x 2",
    (1, 3): "1 x 3",
    (2, 2): "2 x 2",
    (2, 3): "2 x 3",
    (3, 2): "3 x 2",
}


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


def _accent_rgb() -> tuple[int, int, int]:
    hex_color = ACTIVE_PANEL_BADGE_COLOR.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _image_contains_color(img: Image.Image, rgb: tuple[int, int, int], tolerance: int = 10) -> bool:
    arr = np.asarray(img.convert("RGB"))
    close = np.all(np.abs(arr.astype(int) - np.array(rgb)) <= tolerance, axis=-1)
    return bool(close.any())


# --- Badge tracks the active panel through every trigger ------------------------


def test_badge_shows_p1_for_a_single_panel_layout(qapp):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    canvas = PlotCanvas()
    canvas.show()

    canvas.render(figure)

    assert canvas._active_panel_badge.isVisible()
    assert canvas._active_panel_badge.text() == "P1 · ACTIVE"
    canvas.close()


def test_clicking_a_different_panel_moves_the_badge(qapp):
    window = MainWindow()
    window.show()
    window.figure_size_panel.layout_combo.setCurrentIndex(1)  # "1 x 2"
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for panel in window.figure_model.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    window._rerender()
    badge = window.plot_canvas._active_panel_badge
    assert badge.text() == "P1 · ACTIVE"
    position_before = badge.pos()

    ax = window.plot_canvas.axes_list[1]
    event = MouseEvent("button_press_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    window._on_canvas_click(event)

    assert badge.text() == "P2 · ACTIVE"
    assert badge.pos() != position_before
    window.close()


def test_badge_updates_when_the_toolbar_active_panel_selector_changes(qapp):
    window = MainWindow()
    window.show()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"

    window.toolbar_panel_combo.setCurrentIndex(2)

    assert window.plot_canvas._active_panel_badge.text() == "P3 · ACTIVE"
    window.close()


def test_badge_updates_after_a_layout_change(qapp):
    window = MainWindow()
    window.show()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2"
    window.toolbar_panel_combo.setCurrentIndex(3)  # Panel 4
    assert window.plot_canvas._active_panel_badge.text() == "P4 · ACTIVE"

    window.figure_size_panel.layout_combo.setCurrentIndex(0)  # "1 x 1" -- clamps index

    assert window.figure_model.active_panel_index == 0
    assert window.plot_canvas._active_panel_badge.text() == "P1 · ACTIVE"
    window.close()


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 2)])
def test_badge_works_across_every_supported_layout(qapp, layout):
    window = MainWindow()
    window.show()
    index = next(i for i, (t, _dims) in enumerate(LAYOUT_PRESETS) if t == _LAYOUT_TEXT[layout])
    window.figure_size_panel.layout_combo.setCurrentIndex(index)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for panel in window.figure_model.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    window._rerender()

    badge = window.plot_canvas._active_panel_badge
    assert badge.isVisible()
    assert badge.text() == "P1 · ACTIVE"

    last_index = layout[0] * layout[1] - 1
    window.toolbar_panel_combo.setCurrentIndex(last_index)
    assert badge.text() == f"P{last_index + 1} · ACTIVE"
    window.close()


# --- Axes/spines remain scientifically unchanged ----------------------------------


def test_active_panel_never_mutates_spine_color_or_width_across_layouts(qapp):
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    dataset = _make_dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    canvas = PlotCanvas()
    canvas.show()

    for index in range(4):
        figure.set_active_panel(index)
        canvas.render(figure)
        for ax in canvas.axes_list:
            for spine in ax.spines.values():
                assert spine.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
                assert spine.get_linewidth() == 1.0  # Panel's own default, untouched
    canvas.close()


# --- The badge is GUI state only: absent from the model ---------------------------


def test_gnovi_figure_has_no_active_panel_badge_field():
    figure_fields = {f.name for f in dataclasses.fields(Panel)}
    assert not any("badge" in name for name in figure_fields)
    assert not any("highlight" in name for name in figure_fields)
    assert not hasattr(GnoviFigure(), "active_panel_badge")


def test_active_panel_badge_is_not_serialized_in_to_dict():
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    figure.set_active_panel(1)

    data = figure.to_dict()

    serialized_text = str(data)
    assert "badge" not in serialized_text.lower()
    assert "highlight" not in serialized_text.lower()


def test_graph_snapshot_panel_has_no_badge_state():
    from gnovi_plot.plotting.graph import Graph

    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_make_dataset(), "x", "y"))
    graph = Graph(name="G", panel=figure.active_panel)

    data = graph.to_dict()

    assert "badge" not in str(data).lower()


# --- Absent from Matplotlib's own "Save" and every export format ------------------


def test_badge_color_never_appears_in_matplotlib_figure_savefig(qapp):
    """Simulates the navigation toolbar's own "Save" action: it calls
    `canvas.figure.savefig(...)` directly on the live Matplotlib Figure --
    which never contained the Qt-widget badge to begin with."""
    window = MainWindow()
    window.show()
    window.figure_size_panel.layout_combo.setCurrentIndex(3)  # "2 x 2", so a badge is visibly showing
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for panel in window.figure_model.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    window._rerender()
    window.toolbar_panel_combo.setCurrentIndex(2)
    assert window.plot_canvas._active_panel_badge.isVisible()

    buffer = io.BytesIO()
    window.plot_canvas.figure.savefig(buffer, format="png")
    buffer.seek(0)
    img = Image.open(buffer)

    assert not _image_contains_color(img, _accent_rgb())
    window.close()


@pytest.mark.parametrize("fmt", ["png", "tiff", "svg", "pdf"])
def test_badge_never_appears_in_any_export_format(qapp, tmp_path, fmt):
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    figure.set_active_panel(2)
    dataset = _make_dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))

    out_path = tmp_path / f"export.{fmt}"
    export_figure(figure, out_path, dpi=150)

    if fmt in ("png", "tiff"):
        img = Image.open(out_path)
        assert not _image_contains_color(img, _accent_rgb())
    else:
        content = out_path.read_bytes()
        # Vector formats: the accent hex string itself must never appear as
        # a fill/stroke color anywhere in the file.
        assert ACTIVE_PANEL_BADGE_COLOR.encode() not in content.lower()


def test_badge_never_appears_even_when_the_active_panel_export_would_be_index_zero(qapp, tmp_path):
    """Panel 1 (index 0) is the default active panel -- confirms absence
    isn't just because a non-zero active index happens to export cleanly."""
    figure = GnoviFigure()
    dataset = _make_dataset()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    out_path = tmp_path / "single.png"
    export_figure(figure, out_path, dpi=150)

    img = Image.open(out_path)
    assert not _image_contains_color(img, _accent_rgb())
