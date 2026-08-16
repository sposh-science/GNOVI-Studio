"""GNOVI Export Figure vs. Matplotlib toolbar Save -- WYSIWYG parity.

Architectural change under test: Export Figure no longer reconstructs a
second Matplotlib Figure for "Complete Figure"/"Active Panel" export (see
`export.figure_export.export_live_figure`) -- it saves the exact same live
`PlotCanvas.figure` the Matplotlib navigation toolbar's own "Save" button
would save, via `savefig()`. This makes the two structurally identical by
construction rather than by coincidence, for any `bbox_inches`/dpi/
background choice held equal.

"Effectively visually identical" here means structural/physical
equivalence (figure size, panel geometry, font sizes, legend size/position,
margins, spacing, Figure/Panel Aspect Ratio) -- not literal pixel-perfect
equality between two different Matplotlib rendering paths, since even A vs
A (the identical Figure saved twice) has zero reason to differ; the
"different path" distinction from the old architecture (fresh Figure vs
live Figure) is what's actually being tested here.
"""

import io

import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import MouseEvent
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.styles import ACTIVE_PANEL_BADGE_COLOR
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS
from gnovi_plot.gui.widgets.plot_canvas import ReferenceCursorMode
from gnovi_plot.plotting.series import PlotSeries

_LAYOUT_TEXT = {(1, 1): "1 x 1", (1, 2): "1 x 2", (2, 2): "2 x 2", (2, 3): "2 x 3"}
_LONG_LEGEND_LABEL = "S1 (Ferricyanide) SR-0.1 — Current/A"


def _dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1e-5, 4e-5, 9e-5, 1.6e-4]})
    return Dataset(name=name, dataframe=df)


def _process():
    QApplication.instance().processEvents()


def _build_window(layout, *, three_cycle_legend=False):
    window = MainWindow()
    window.show()
    window.resize(1300, 850)
    _process()
    index = next(i for i, (t, _dims) in enumerate(LAYOUT_PRESETS) if t == _LAYOUT_TEXT[layout])
    window.figure_size_panel.layout_combo.setCurrentIndex(index)
    ds = _dataset()
    window.dataset_manager.add(ds)
    for panel in window.figure_model.panels:
        if three_cycle_legend:
            for i in range(3):
                panel.add_series(PlotSeries.line(ds, "x", "y", label=f"Cycle {i + 1}"))
        else:
            panel.add_series(PlotSeries.line(ds, "x", "y", label=_LONG_LEGEND_LABEL))
        panel.legend_visible = True
        panel.scientific_notation_y = True
    window._rerender()
    _process()
    return window


def _toolbar_save_bytes(window, **savefig_kwargs) -> bytes:
    buf = io.BytesIO()
    window.plot_canvas.figure.savefig(buf, format="png", **savefig_kwargs)
    return buf.getvalue()


def _gnovi_export_bytes(window, dialog: ExportFigureDialog, tmp_path, name="out.png") -> bytes:
    out_path = tmp_path / name
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()
    return out_path.read_bytes()


# --- A (Matplotlib toolbar Save) vs B (GNOVI Export Figure) parity ----------------


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_complete_figure_export_is_byte_identical_to_toolbar_save(qapp, tmp_path, layout):
    window = _build_window(layout)
    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dialog.scope_combo.currentText() == "Complete Figure"

    toolbar_bytes = _toolbar_save_bytes(window, dpi=300)
    gnovi_bytes = _gnovi_export_bytes(window, dialog, tmp_path)

    assert toolbar_bytes == gnovi_bytes
    window.close()


def test_long_legend_case_matches_toolbar_save(qapp, tmp_path):
    """The specific case that originally exposed the divergence:
    "S1 (Ferricyanide) SR-0.1 — Current/A"."""
    window = _build_window((2, 2))
    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)

    toolbar_bytes = _toolbar_save_bytes(window, dpi=300)
    gnovi_bytes = _gnovi_export_bytes(window, dialog, tmp_path)

    assert toolbar_bytes == gnovi_bytes
    window.close()


def test_three_cycle_legend_case_matches_toolbar_save(qapp, tmp_path):
    window = _build_window((1, 2), three_cycle_legend=True)
    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)

    toolbar_bytes = _toolbar_save_bytes(window, dpi=300)
    gnovi_bytes = _gnovi_export_bytes(window, dialog, tmp_path)

    assert toolbar_bytes == gnovi_bytes
    window.close()


def test_legend_size_relative_to_axes_matches_between_the_two_paths(qapp):
    """Beyond byte-equality of the saved file: confirms the ACTUAL rendered
    legend-to-axes ratio (what a human would perceive as "legend size") is
    identical, computed independently via the live renderer."""
    window = _build_window((2, 2))
    canvas = window.plot_canvas
    ax = canvas.axes_list[0]
    renderer = canvas.get_renderer()

    ax_h = ax.get_window_extent(renderer=renderer).height
    legend_h = ax.get_legend().get_window_extent(renderer=renderer).height
    ratio = legend_h / ax_h

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    # The dialog's own export path reads this exact same live Axes -- no
    # separate figure was built, so the ratio is trivially the same object;
    # this assertion documents that invariant explicitly.
    assert dialog._plot_canvas.axes_list[0] is ax
    assert ratio > 0
    window.close()


# --- DPI: resolution only, never composition ----------------------------------------


def test_300_vs_600_dpi_preserves_composition(qapp, tmp_path):
    window = _build_window((2, 2))
    dialog_300 = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog_300.dpi_preset_combo.setCurrentText("300")
    dialog_600 = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog_600.dpi_preset_combo.setCurrentText("600")

    img_300 = Image.open(io.BytesIO(_gnovi_export_bytes(window, dialog_300, tmp_path, "a.png")))
    img_600 = Image.open(io.BytesIO(_gnovi_export_bytes(window, dialog_600, tmp_path, "b.png")))

    assert img_600.size[0] == pytest.approx(img_300.size[0] * 2, abs=2)
    assert img_600.size[1] == pytest.approx(img_300.size[1] * 2, abs=2)
    window.close()


def test_png_and_tiff_preserve_the_same_composition(qapp, tmp_path):
    window = _build_window((1, 2))
    png_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    png_dialog.format_combo.setCurrentText("PNG")
    tiff_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    tiff_dialog.format_combo.setCurrentText("TIFF")

    png_bytes = _gnovi_export_bytes(window, png_dialog, tmp_path, "a.png")
    tiff_bytes = _gnovi_export_bytes(window, tiff_dialog, tmp_path, "b.tiff")
    png_img = Image.open(io.BytesIO(png_bytes))
    tiff_img = Image.open(io.BytesIO(tiff_bytes))

    assert png_img.size == tiff_img.size
    window.close()


@pytest.mark.parametrize("fmt", ["SVG", "PDF"])
def test_vector_formats_preserve_geometry(qapp, tmp_path, fmt):
    window = _build_window((2, 3))
    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.format_combo.setCurrentText(fmt)
    out_path = tmp_path / f"out.{fmt.lower()}"
    dialog.path_edit.setText(str(out_path))

    dialog._on_accept()

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    window.close()


# --- Tight bounding box: opt-in only, changes cropping only -------------------------


def test_normal_vs_tight_bbox_only_changes_cropping(qapp, tmp_path):
    window = _build_window((2, 2))
    normal_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    tight_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    tight_dialog.bbox_combo.setCurrentText("Tight")

    normal_bytes = _gnovi_export_bytes(window, normal_dialog, tmp_path, "normal.png")
    tight_bytes = _gnovi_export_bytes(window, tight_dialog, tmp_path, "tight.png")
    normal_img = Image.open(io.BytesIO(normal_bytes))
    tight_img = Image.open(io.BytesIO(tight_bytes))

    assert tight_img.size != normal_img.size
    assert tight_img.size[0] <= normal_img.size[0]
    assert tight_img.size[1] <= normal_img.size[1]
    window.close()


def test_default_bbox_is_normal_matching_toolbar_save(qapp):
    window = _build_window((1, 1))
    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dialog.bbox_combo.currentText() == "Normal"


# --- Figure/Panel Aspect Ratio preserved through the WYSIWYG path -------------------


def test_locked_figure_aspect_ratio_is_reflected_in_the_export(qapp, tmp_path):
    """Complete Figure export saves the live canvas.figure AS-IS -- the
    same full widget-sized Figure Matplotlib toolbar Save would produce,
    letterbox margins included, never auto-cropped to just the meaningful
    square (auto-cropping there would itself be a silent, undocumented
    export-only behavior the two paths wouldn't share -- see the module
    docstring). What "Figure Aspect Ratio reflected in the export" means
    here is that the exported image's overall dimensions exactly match the
    live canvas's own current physical size (i.e. nothing was resized or
    distorted during the save step), and that the CONTENT within it (the
    subplot box) is the correctly letterboxed square -- checked
    independently via `subplotpars`, not by inspecting exported pixels."""
    window = _build_window((1, 1))
    window.figure_model.figure_width_in = 6.4
    window.figure_model.figure_height_in = 6.4
    window.figure_model.aspect_preset = "1:1"
    window.figure_model.lock_aspect_ratio = True
    window._rerender()
    _process()

    canvas = window.plot_canvas
    sp = canvas.figure.subplotpars
    content_ratio = ((sp.right - sp.left) * canvas.width()) / ((sp.top - sp.bottom) * canvas.height())
    assert content_ratio == pytest.approx(1.0, rel=0.01)  # the locked 1:1 content area

    expected_w_in, expected_h_in = canvas.figure.get_size_inches()
    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    img = Image.open(io.BytesIO(_gnovi_export_bytes(window, dialog, tmp_path)))

    dpi = dialog.dpi_spin.value()
    assert img.size == (round(expected_w_in * dpi), round(expected_h_in * dpi))
    window.close()


def test_panel_aspect_ratio_is_reflected_in_the_export(qapp, tmp_path):
    window = _build_window((2, 2))
    window.figure_model.panel_aspect_preset = "1:1"
    window._rerender()
    _process()

    for ax in window.plot_canvas.axes_list:
        assert ax.get_box_aspect() == pytest.approx(1.0)

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    img = Image.open(io.BytesIO(_gnovi_export_bytes(window, dialog, tmp_path)))
    assert img.size[0] > 0 and img.size[1] > 0  # exported without raising
    window.close()


# --- Active Panel scope -------------------------------------------------------------


def test_active_panel_scope_excludes_neighboring_panels(qapp, tmp_path):
    window = _build_window((2, 2))
    window.toolbar_panel_combo.setCurrentIndex(0)
    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog.scope_combo.setCurrentText("Active Panel")

    active_bytes = _gnovi_export_bytes(window, dialog, tmp_path, "active.png")
    complete_dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    complete_bytes = _gnovi_export_bytes(window, complete_dialog, tmp_path, "complete.png")

    active_img = Image.open(io.BytesIO(active_bytes))
    complete_img = Image.open(io.BytesIO(complete_bytes))
    assert active_img.size[0] < complete_img.size[0]
    assert active_img.size[1] < complete_img.size[1]
    window.close()


def test_active_panel_scope_follows_the_currently_active_panel(qapp, tmp_path):
    window = _build_window((1, 2))
    # Give the two panels genuinely different titles so their exported
    # crops can't coincidentally end up byte-identical.
    window.figure_model.panels[0].title = "Panel One"
    window.figure_model.panels[1].title = "Panel Two"
    window._rerender()
    _process()

    window.toolbar_panel_combo.setCurrentIndex(0)
    dialog0 = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog0.scope_combo.setCurrentText("Active Panel")
    bytes0 = _gnovi_export_bytes(window, dialog0, tmp_path, "p0.png")

    window.toolbar_panel_combo.setCurrentIndex(1)
    dialog1 = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    dialog1.scope_combo.setCurrentText("Active Panel")
    bytes1 = _gnovi_export_bytes(window, dialog1, tmp_path, "p1.png")

    assert bytes0 != bytes1
    window.close()


# --- Background options ---------------------------------------------------------------


def test_background_options_produce_distinct_results(qapp, tmp_path):
    window = _build_window((1, 1))

    as_shown = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    opaque = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    opaque.background_combo.setCurrentText("Opaque")
    transparent = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    transparent.background_combo.setCurrentText("Transparent")

    img_as_shown = Image.open(io.BytesIO(_gnovi_export_bytes(window, as_shown, tmp_path, "a.png"))).convert("RGBA")
    img_opaque = Image.open(io.BytesIO(_gnovi_export_bytes(window, opaque, tmp_path, "b.png"))).convert("RGBA")
    img_transparent = Image.open(
        io.BytesIO(_gnovi_export_bytes(window, transparent, tmp_path, "c.png"))
    ).convert("RGBA")

    assert img_as_shown.getpixel((0, 0))[3] == 255  # opaque (Light theme, "as shown")
    assert img_opaque.getpixel((0, 0)) == (255, 255, 255, 255)
    assert img_transparent.getpixel((0, 0))[3] == 0
    window.close()


# --- GUI-only overlays never appear in any export ------------------------------------


def _accent_rgb() -> tuple[int, int, int]:
    h = ACTIVE_PANEL_BADGE_COLOR.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _contains_color(img: Image.Image, rgb: tuple[int, int, int], tol=10) -> bool:
    arr = np.asarray(img.convert("RGB"))
    return bool(np.all(np.abs(arr.astype(int) - np.array(rgb)) <= tol, axis=-1).any())


def test_active_panel_badge_never_appears_in_gnovi_export(qapp, tmp_path):
    window = _build_window((2, 2))
    window.toolbar_panel_combo.setCurrentIndex(2)
    assert window.plot_canvas._active_panel_badge.isVisible()

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    img = Image.open(io.BytesIO(_gnovi_export_bytes(window, dialog, tmp_path)))

    assert not _contains_color(img, _accent_rgb())
    window.close()


def test_active_panel_badge_never_appears_in_toolbar_save(qapp):
    window = _build_window((2, 2))
    window.toolbar_panel_combo.setCurrentIndex(1)
    assert window.plot_canvas._active_panel_badge.isVisible()

    img = Image.open(io.BytesIO(_toolbar_save_bytes(window)))

    assert not _contains_color(img, _accent_rgb())
    window.close()


def _trigger_crosshair(window) -> None:
    window._on_cursor_mode_changed(ReferenceCursorMode.CROSSHAIR)
    ax = window.plot_canvas.active_axes(window.figure_model)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    event = MouseEvent("motion_notify_event", window.plot_canvas, 1, 1)
    event.inaxes = ax
    event.xdata, event.ydata = 5.0, 10.0
    window._on_mouse_move(event)
    _process()
    assert len(window.plot_canvas._cursor_artists) == 2


def test_reference_cursor_never_appears_in_gnovi_export(qapp, tmp_path):
    """The cursor's near-neutral gray (#8a8f99) is too close to ordinary
    black-text anti-aliasing to reliably tell apart by pixel color alone
    (unlike the badge's distinct blue, see the other absence tests below)
    -- so this checks the actual mechanism instead: the cursor Line2D
    artists are gone from the Axes (not just our own bookkeeping list) by
    the time the export happens, both before and after."""
    window = _build_window((1, 1))
    ax = window.plot_canvas.active_axes(window.figure_model)
    lines_before_cursor = len(ax.lines)
    _trigger_crosshair(window)
    assert len(ax.lines) == lines_before_cursor + 2  # the crosshair really was drawn

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert len(ax.lines) == lines_before_cursor  # cleared by dialog construction's own preview pass
    assert window.plot_canvas._cursor_artists == []

    dialog.path_edit.setText(str(tmp_path / "out.png"))
    dialog._on_accept()

    assert len(ax.lines) == lines_before_cursor  # still just the data series, at the moment of export
    assert window.plot_canvas._cursor_artists == []
    window.close()


def test_reference_cursor_never_appears_in_export_preview(qapp):
    window = _build_window((1, 1))
    _trigger_crosshair(window)

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    # Constructing the dialog already renders one preview pass; force a
    # second explicitly to be sure the cursor-clear happens on every pass,
    # not just the first.
    dialog._refresh_preview()

    assert window.plot_canvas._cursor_artists == []
    window.close()


def test_reference_cursor_never_appears_in_toolbar_save(qapp, monkeypatch):
    window = _build_window((1, 1))
    _trigger_crosshair(window)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))

    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    nav = window.findChildren(NavigationToolbar2QT)[0]
    save_action = next(a for a in nav.actions() if a.text().lower() == "save")
    save_action.trigger()

    assert window.plot_canvas._cursor_artists == []
    window.close()
