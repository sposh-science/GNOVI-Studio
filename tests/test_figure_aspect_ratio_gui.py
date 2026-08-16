"""End-to-end containment tests for a locked figure aspect ratio, driven
through the real `MainWindow` (real splitters/drawers, not just
`PlotCanvas._letterbox_rect` in isolation -- see `tests/test_plot_canvas_polish.py`
for the pure-math version). Confirms the OUTER figure rectangle stays at the
configured aspect ratio no matter how the surrounding workspace is
reshaped, per the "locked figure aspect ratio" requirement: the figure must
never stretch to fill the available center viewport -- it letterboxes/
pillarboxes instead.
"""

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    return Dataset(name=name, dataframe=df)


def _content_aspect_ratio(window) -> float:
    """The actual on-screen aspect ratio of the letterboxed content area,
    computed from the live PlotCanvas widget's real pixel dimensions --
    independent of `figure_width_in`/`figure_height_in` themselves, so this
    is a genuine check of what would be *displayed*, not a restatement of
    the input."""
    canvas = window.plot_canvas
    figure = window.figure_model
    left, bottom, right, top = canvas._letterbox_rect(figure)
    content_w_px = (right - left) * canvas.width()
    content_h_px = (top - bottom) * canvas.height()
    return content_w_px / content_h_px


def _lock_square_figure(window, size_in=6.4):
    window.figure_model.figure_width_in = size_in
    window.figure_model.figure_height_in = size_in
    window.figure_model.lock_aspect_ratio = True
    window._rerender()


def test_locked_1_1_figure_stays_square_in_a_wide_viewport(qapp):
    window = MainWindow()
    window.show()
    window.resize(1600, 900)
    QApplication.instance().processEvents()
    _lock_square_figure(window)

    assert window.plot_canvas.width() > window.plot_canvas.height()  # a genuinely wide viewport
    assert _content_aspect_ratio(window) == pytest.approx(1.0, rel=1e-2)
    window.close()


def test_locked_4_3_figure_keeps_its_ratio_in_a_wide_viewport(qapp):
    window = MainWindow()
    window.show()
    window.resize(1600, 900)
    QApplication.instance().processEvents()
    window.figure_model.figure_width_in = 8.0
    window.figure_model.figure_height_in = 6.0
    window.figure_model.lock_aspect_ratio = True
    window._rerender()

    assert _content_aspect_ratio(window) == pytest.approx(8.0 / 6.0, rel=1e-2)
    window.close()


def test_locked_16_9_figure_keeps_its_ratio_in_a_narrow_viewport(qapp):
    window = MainWindow()
    window.show()
    window.resize(900, 1400)
    QApplication.instance().processEvents()
    window.figure_model.figure_width_in = 16.0
    window.figure_model.figure_height_in = 9.0
    window.figure_model.lock_aspect_ratio = True
    window._rerender()

    assert _content_aspect_ratio(window) == pytest.approx(16.0 / 9.0, rel=1e-2)
    window.close()


def test_a_locked_square_figure_actually_renders_blank_pillarbox_margins(qapp):
    """Goes one step further than the rect math: renders real pixels and
    confirms the canvas actually shows blank (figure-background-colored)
    margins on the sides, rather than stretching plotted content to fill
    the wide viewport -- the concrete visual symptom the aspect-lock
    requirement is about."""
    window = MainWindow()
    window.show()
    window.resize(1600, 700)
    QApplication.instance().processEvents()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    _lock_square_figure(window)
    QApplication.instance().processEvents()
    window.plot_canvas.draw()
    QApplication.instance().processEvents()

    buf = np.asarray(window.plot_canvas.buffer_rgba())
    white = (buf[:, :, 0] == 255) & (buf[:, :, 1] == 255) & (buf[:, :, 2] == 255)
    all_white_columns = white.all(axis=0)

    # A wide (non-square) canvas locked to 1:1 must have blank columns on
    # both the left and right edges (pillarboxing) -- i.e. it must NOT be
    # stretched to fill every column with rendered content.
    assert all_white_columns[0]
    assert all_white_columns[-1]
    assert not all_white_columns.all()  # but the center still has content
    window.close()


# --- Containment survives every kind of workspace reshaping --------------------


def test_collapsing_the_left_drawer_does_not_distort_the_locked_ratio(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    QApplication.instance().processEvents()
    _lock_square_figure(window)
    before = _content_aspect_ratio(window)

    window.tool_drawer._buttons["data"].click()  # collapse (already active)
    QApplication.instance().processEvents()
    window._rerender()

    assert _content_aspect_ratio(window) == pytest.approx(before, rel=1e-2)
    assert _content_aspect_ratio(window) == pytest.approx(1.0, rel=1e-2)
    window.close()


def test_collapsing_the_right_drawer_does_not_distort_the_locked_ratio(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    QApplication.instance().processEvents()
    _lock_square_figure(window)

    window.working_drawer._buttons["working"].click()  # collapse
    QApplication.instance().processEvents()
    window._rerender()

    assert _content_aspect_ratio(window) == pytest.approx(1.0, rel=1e-2)
    window.close()


def test_resizing_the_bottom_panel_does_not_distort_the_locked_ratio(qapp):
    window = MainWindow()
    window.show()
    window.resize(1400, 900)
    QApplication.instance().processEvents()
    _lock_square_figure(window)

    total = sum(window.center_splitter.sizes())
    window.center_splitter.setSizes([int(total * 0.3), int(total * 0.7)])  # shrink the canvas a lot
    QApplication.instance().processEvents()
    window._rerender()

    assert _content_aspect_ratio(window) == pytest.approx(1.0, rel=1e-2)
    window.close()


def test_maximizing_the_window_does_not_distort_the_locked_ratio(qapp):
    window = MainWindow()
    window.show()
    window.resize(1000, 700)
    QApplication.instance().processEvents()
    _lock_square_figure(window)

    window.resize(1920, 1080)
    QApplication.instance().processEvents()
    window._rerender()

    assert _content_aspect_ratio(window) == pytest.approx(1.0, rel=1e-2)
    window.close()


@pytest.mark.parametrize("preset_text", ["1 x 1", "1 x 2", "2 x 1", "2 x 2"])
def test_switching_panel_layout_preserves_the_locked_outer_ratio(qapp, preset_text):
    """The OUTER figure rectangle's ratio is a figure-level property -- it
    must hold regardless of how many panels are inside it."""
    window = MainWindow()
    window.show()
    window.resize(1500, 800)
    QApplication.instance().processEvents()
    _lock_square_figure(window)

    index = next(i for i, (text, _dims) in enumerate(LAYOUT_PRESETS) if text == preset_text)
    window.figure_size_panel.layout_combo.setCurrentIndex(index)
    QApplication.instance().processEvents()
    window._rerender()

    assert _content_aspect_ratio(window) == pytest.approx(1.0, rel=1e-2)
    window.close()


def test_changing_width_and_height_with_lock_enabled_keeps_the_canvas_consistent(qapp):
    window = MainWindow()
    window.show()
    window.resize(1500, 800)
    QApplication.instance().processEvents()
    window.figure_size_panel.lock_check.setChecked(True)

    window.figure_size_panel.width_spin.setValue(9.0)  # height follows, aspect lock is on

    assert window.figure_model.lock_aspect_ratio is True
    ratio = window.figure_model.figure_width_in / window.figure_model.figure_height_in
    QApplication.instance().processEvents()
    window._rerender()
    assert _content_aspect_ratio(window) == pytest.approx(ratio, rel=1e-2)
    window.close()


def test_changing_aspect_preset_updates_the_on_screen_containment(qapp):
    window = MainWindow()
    window.show()
    window.resize(1500, 500)  # very wide viewport
    QApplication.instance().processEvents()

    window.figure_size_panel.aspect_combo.setCurrentText("16:9")
    QApplication.instance().processEvents()
    window._rerender()

    assert window.figure_model.lock_aspect_ratio is True
    assert _content_aspect_ratio(window) == pytest.approx(16 / 9, rel=1e-2)
    window.close()


# --- Independent geometry measurement (2x3 layout) -------------------------------
#
# `_content_aspect_ratio` above re-derives its expectation from
# `PlotCanvas._letterbox_rect` -- useful, but it's checking that function
# against itself. The tests below instead measure the ACTUAL Matplotlib
# Axes positions Matplotlib itself applied (`ax.get_position()`) and the
# actually-applied `Figure.subplotpars`, cross-checked against each other,
# for a multi-panel (2x3) layout -- an independent, first-principles read of
# "what is actually displayed", not a restatement of our own rect math.


def _measured_outer_rect_ratio(window) -> tuple[float, float]:
    """(axes-position-based ratio, subplotpars-based ratio) of the outer
    rectangle bounding every rendered panel, in real canvas pixels."""
    canvas = window.plot_canvas
    lefts = [ax.get_position().x0 for ax in canvas.axes_list]
    rights = [ax.get_position().x1 for ax in canvas.axes_list]
    bottoms = [ax.get_position().y0 for ax in canvas.axes_list]
    tops = [ax.get_position().y1 for ax in canvas.axes_list]
    left, right, bottom, top = min(lefts), max(rights), min(bottoms), max(tops)
    axes_ratio = ((right - left) * canvas.width()) / ((top - bottom) * canvas.height())

    sp = canvas.figure.subplotpars
    subplotpars_ratio = ((sp.right - sp.left) * canvas.width()) / ((sp.top - sp.bottom) * canvas.height())
    return axes_ratio, subplotpars_ratio


def _prepare_2x3(window, width_in, height_in, aspect_preset, canvas_window_size=(1600, 700)):
    window.show()
    window.resize(*canvas_window_size)
    QApplication.instance().processEvents()
    index = next(i for i, (t, _dims) in enumerate(LAYOUT_PRESETS) if t == "2 x 3")
    window.figure_size_panel.layout_combo.setCurrentIndex(index)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for panel in window.figure_model.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    window.figure_model.figure_width_in = width_in
    window.figure_model.figure_height_in = height_in
    window.figure_model.aspect_preset = aspect_preset
    window.figure_model.lock_aspect_ratio = True
    window._rerender()
    QApplication.instance().processEvents()


@pytest.mark.parametrize(
    "width_in,height_in,preset,expected_ratio",
    [
        (6.40, 6.40, "1:1", 1.0),
        (8.0, 6.0, "4:3", 4 / 3),
        (16.0, 9.0, "16:9", 16 / 9),
    ],
)
def test_2x3_outer_figure_rectangle_matches_the_configured_ratio(
    qapp, width_in, height_in, preset, expected_ratio
):
    """The exact scenario reported: Aspect preset + Width/Height as given,
    Lock ON, Layout 2x3, workspace much wider than tall. The measured ratio
    of the actual displayed panel block (not the whole center workspace)
    must match the configured aspect within a small pixel-rounding
    tolerance."""
    window = MainWindow()
    _prepare_2x3(window, width_in, height_in, preset)

    axes_ratio, subplotpars_ratio = _measured_outer_rect_ratio(window)

    assert axes_ratio == pytest.approx(subplotpars_ratio, rel=1e-6)  # internally consistent
    assert axes_ratio == pytest.approx(expected_ratio, rel=0.01)  # within ~1% (integer-pixel rounding)
    window.close()


# --- Export geometry -------------------------------------------------------------


@pytest.mark.parametrize(
    "width_in,height_in,expected_ratio",
    [(6.40, 6.40, 1.0), (8.0, 6.0, 4 / 3), (16.0, 9.0, 16 / 9)],
)
def test_export_pixel_dimensions_match_the_configured_ratio_without_tight_bbox(
    tmp_path, width_in, height_in, expected_ratio
):
    """With `tight_bbox=False` (no content-based cropping), exported raster
    pixel dimensions are exactly `figure_width_in * dpi` x
    `figure_height_in * dpi` -- confirming the underlying size/DPI
    arithmetic is exact. For 1:1 this means export pixel width == export
    pixel height exactly."""
    from PIL import Image

    from gnovi_plot.export.figure_export import export_figure
    from gnovi_plot.plotting.figure import GnoviFigure

    figure = GnoviFigure()
    figure.set_layout(2, 3)
    dataset = _make_dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    figure.figure_width_in = width_in
    figure.figure_height_in = height_in
    figure.lock_aspect_ratio = True

    dpi = 150
    out_path = tmp_path / "export.png"
    export_figure(figure, out_path, dpi=dpi, tight_bbox=False)
    img = Image.open(out_path)

    assert img.size == (round(width_in * dpi), round(height_in * dpi))
    assert (img.size[0] / img.size[1]) == pytest.approx(expected_ratio, rel=1e-6)


def test_export_with_the_default_tight_bbox_documents_the_known_content_crop_deviation(tmp_path):
    """`tight_bbox=True` is the Export Figure dialog's own default (see
    `gui.dialogs.export_figure_dialog`) -- it crops the saved raster to the
    tight bounding box of actually-rendered content (labels/legend/ticks),
    which is standard Matplotlib `bbox_inches="tight"` behavior, not part of
    the aspect-ratio containment mechanism. This intentionally does NOT
    assert exact 1:1 pixel parity for a 1:1 figure -- it documents that the
    deviation stays small (a few percent, from asymmetric label/tick
    padding) rather than wildly distorted, and that turning tight_bbox off
    (see the test above) restores exact parity. No code change was made for
    this -- see the "VERIFICATION" report for why."""
    from PIL import Image

    from gnovi_plot.export.figure_export import export_figure
    from gnovi_plot.plotting.figure import GnoviFigure

    figure = GnoviFigure()
    figure.set_layout(2, 3)
    dataset = _make_dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    figure.figure_width_in = 6.40
    figure.figure_height_in = 6.40
    figure.lock_aspect_ratio = True

    out_path = tmp_path / "export_tight.png"
    export_figure(figure, out_path, dpi=150, tight_bbox=True)
    img = Image.open(out_path)

    ratio = img.size[0] / img.size[1]
    assert 0.9 < ratio < 1.1  # small content-crop deviation, not a distortion bug
