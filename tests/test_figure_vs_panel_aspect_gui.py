"""Figure Aspect Ratio vs. Panel Aspect Ratio, driven through the real
`MainWindow`/`PlotCanvas`. See `tests/test_panel_aspect_ratio.py` for the
pure backend-level coverage and `tests/test_figure_aspect_ratio_gui.py` for
Figure-Aspect-only containment tests (no Panel Aspect Ratio involved there).

Measurement note: once Panel Aspect Ratio constrains individual panels, the
union of the panels' *actual* (possibly box-aspect-shrunk) positions is a
tight *content* bounding box, not the *configured* Figure Aspect Ratio --
analogous to how `bbox_inches="tight"` export cropping deviates from the
nominal figure size (see the prior aspect-ratio task's report). The
configured Figure Aspect Ratio is measured here via
`canvas.figure.subplotpars` instead -- the actual applied `subplots_adjust`
box, unaffected by what individual panels do inside their own cells.
"""

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.export.figure_export import export_figure
from gnovi_plot.gui.main_window import MainWindow
from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS
from gnovi_plot.plotting.series import PlotSeries

_LAYOUT_TEXT = {(1, 1): "1 x 1", (1, 2): "1 x 2", (2, 1): "2 x 1", (2, 2): "2 x 2", (2, 3): "2 x 3"}


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [-0.2, 0.0, 0.6], "y": [-5e-5, 0.0, 5e-5]})
    return Dataset(name=name, dataframe=df)


def _process():
    QApplication.instance().processEvents()


def _configured_figure_ratio(window) -> float:
    """The NOMINAL outer figure rectangle Figure Aspect Ratio actually
    governs -- see module docstring for why this is `subplotpars`, not an
    axes-position union, once Panel Aspect Ratio is also in play."""
    canvas = window.plot_canvas
    sp = canvas.figure.subplotpars
    return ((sp.right - sp.left) * canvas.width()) / ((sp.top - sp.bottom) * canvas.height())


def _panel_box_ratios(window) -> list[float]:
    canvas = window.plot_canvas
    ratios = []
    for ax in canvas.axes_list:
        pos = ax.get_position()
        ratios.append((pos.width * canvas.width()) / (pos.height * canvas.height()))
    return ratios


def _configure(
    window,
    *,
    layout,
    figure_width_in,
    figure_height_in,
    figure_aspect_preset,
    lock_figure_aspect,
    panel_aspect_preset,
    canvas_window_size=(1600, 900),
):
    window.show()
    window.resize(*canvas_window_size)
    _process()
    index = next(i for i, (t, _dims) in enumerate(LAYOUT_PRESETS) if t == _LAYOUT_TEXT[layout])
    window.figure_size_panel.layout_combo.setCurrentIndex(index)
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    for panel in window.figure_model.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    window.figure_model.figure_width_in = figure_width_in
    window.figure_model.figure_height_in = figure_height_in
    window.figure_model.aspect_preset = figure_aspect_preset
    window.figure_model.lock_aspect_ratio = lock_figure_aspect
    window.figure_model.panel_aspect_preset = panel_aspect_preset
    window._rerender()
    _process()


# --- Combined Figure + Panel aspect scenarios --------------------------------------


def test_figure_1_1_with_panel_auto_matches_prior_figure_only_behavior(qapp):
    window = MainWindow()
    _configure(
        window,
        layout=(1, 1),
        figure_width_in=6.4,
        figure_height_in=6.4,
        figure_aspect_preset="1:1",
        lock_figure_aspect=True,
        panel_aspect_preset="Auto",
    )

    assert _configured_figure_ratio(window) == pytest.approx(1.0, rel=0.01)
    assert window.plot_canvas.axes_list[0].get_box_aspect() is None
    window.close()


def test_figure_16_9_with_panel_1_1_on_a_2x3_layout(qapp):
    window = MainWindow()
    _configure(
        window,
        layout=(2, 3),
        figure_width_in=16.0,
        figure_height_in=9.0,
        figure_aspect_preset="16:9",
        lock_figure_aspect=True,
        panel_aspect_preset="1:1",
    )

    assert _configured_figure_ratio(window) == pytest.approx(16 / 9, rel=0.01)
    for ratio in _panel_box_ratios(window):
        assert ratio == pytest.approx(1.0, rel=1e-3)
    window.close()


def test_figure_4_3_with_panel_4_3(qapp):
    window = MainWindow()
    _configure(
        window,
        layout=(2, 2),
        figure_width_in=8.0,
        figure_height_in=6.0,
        figure_aspect_preset="4:3",
        lock_figure_aspect=True,
        panel_aspect_preset="4:3",
    )

    assert _configured_figure_ratio(window) == pytest.approx(4 / 3, rel=0.01)
    for ratio in _panel_box_ratios(window):
        assert ratio == pytest.approx(4 / 3, rel=1e-3)
    window.close()


@pytest.mark.parametrize("layout", [(1, 2), (2, 2), (2, 3)])
def test_panel_1_1_produces_square_boxes_across_layouts(qapp, layout):
    window = MainWindow()
    _configure(
        window,
        layout=layout,
        figure_width_in=10.0,
        figure_height_in=6.0,
        figure_aspect_preset="Custom",
        lock_figure_aspect=False,
        panel_aspect_preset="1:1",
    )

    ratios = _panel_box_ratios(window)
    assert len(ratios) == layout[0] * layout[1]
    for ratio in ratios:
        assert ratio == pytest.approx(1.0, rel=1e-3)
    window.close()


def test_2x3_with_panel_4_3(qapp):
    window = MainWindow()
    _configure(
        window,
        layout=(2, 3),
        figure_width_in=12.0,
        figure_height_in=6.0,
        figure_aspect_preset="Custom",
        lock_figure_aspect=False,
        panel_aspect_preset="4:3",
    )

    for ratio in _panel_box_ratios(window):
        assert ratio == pytest.approx(4 / 3, rel=1e-3)
    window.close()


def test_panel_auto_preserves_current_unconstrained_behavior(qapp):
    """With Panel Aspect Ratio at "Auto", panels must fill their nominal
    subplot-grid cell exactly, same as before this feature existed --
    panel box ratios then just reflect the grid geometry, not a forced
    ratio."""
    window = MainWindow()
    _configure(
        window,
        layout=(1, 1),
        figure_width_in=8.0,
        figure_height_in=4.0,
        figure_aspect_preset="Custom",
        lock_figure_aspect=False,
        panel_aspect_preset="Auto",
    )

    ax = window.plot_canvas.axes_list[0]
    assert ax.get_box_aspect() is None
    window.close()


# --- Independence from workspace reshaping ------------------------------------------


def test_resizing_the_window_does_not_alter_the_configured_figure_aspect(qapp):
    window = MainWindow()
    _configure(
        window,
        layout=(2, 3),
        figure_width_in=16.0,
        figure_height_in=9.0,
        figure_aspect_preset="16:9",
        lock_figure_aspect=True,
        panel_aspect_preset="1:1",
        canvas_window_size=(1600, 900),
    )
    before = _configured_figure_ratio(window)

    window.resize(900, 1400)  # a very differently-shaped viewport
    _process()

    after = _configured_figure_ratio(window)
    assert after == pytest.approx(before, rel=1e-2)
    assert after == pytest.approx(16 / 9, rel=0.01)
    window.close()


def test_resizing_the_window_does_not_alter_the_configured_panel_aspect(qapp):
    window = MainWindow()
    _configure(
        window,
        layout=(2, 3),
        figure_width_in=16.0,
        figure_height_in=9.0,
        figure_aspect_preset="16:9",
        lock_figure_aspect=True,
        panel_aspect_preset="1:1",
        canvas_window_size=(1600, 900),
    )

    window.resize(700, 1300)
    _process()

    for ratio in _panel_box_ratios(window):
        assert ratio == pytest.approx(1.0, rel=1e-3)
    window.close()


# --- Data axes / limits are never touched -------------------------------------------


def test_panel_aspect_does_not_alter_axis_limits_through_the_real_window(qapp):
    window = MainWindow()
    dataset = _make_dataset()
    window.dataset_manager.add(dataset)
    window._on_add_to_plot([PlotSeries.line(dataset, "x", "y")])
    ax = window.plot_canvas.active_axes(window.figure_model)
    xlim_before, ylim_before = ax.get_xlim(), ax.get_ylim()

    window.figure_model.panel_aspect_preset = "1:1"
    window._rerender()

    ax = window.plot_canvas.active_axes(window.figure_model)
    assert ax.get_xlim() == xlim_before
    assert ax.get_ylim() == ylim_before
    assert ax.get_aspect() == "auto"
    window.close()


# --- Preview/export geometry agreement ------------------------------------------------


def test_preview_and_export_geometry_agree_for_a_2x3_figure_panel_combo(qapp, tmp_path):
    from PIL import Image

    window = MainWindow()
    _configure(
        window,
        layout=(2, 3),
        figure_width_in=16.0,
        figure_height_in=9.0,
        figure_aspect_preset="16:9",
        lock_figure_aspect=True,
        panel_aspect_preset="1:1",
    )

    out_path = tmp_path / "combo.png"
    export_figure(window.figure_model, out_path, dpi=150, tight_bbox=False)
    img = Image.open(out_path)

    assert img.size == (round(16.0 * 150), round(9.0 * 150))
    assert (img.size[0] / img.size[1]) == pytest.approx(16 / 9, rel=1e-6)
    window.close()


# --- Undo/Redo -----------------------------------------------------------------------


def test_panel_aspect_change_is_undoable(qapp):
    window = MainWindow()
    assert window.figure_model.panel_aspect_preset == "Auto"

    window.figure_size_panel.panel_aspect_combo.setCurrentText("1:1")

    assert window.figure_model.panel_aspect_preset == "1:1"
    assert window.undo_action.isEnabled() is True

    window._on_undo()

    assert window.figure_model.panel_aspect_preset == "Auto"
    assert window.figure_size_panel.panel_aspect_combo.currentText() == "Auto"


def test_panel_aspect_change_is_redoable(qapp):
    window = MainWindow()
    window.figure_size_panel.panel_aspect_combo.setCurrentText("4:3")
    window._on_undo()
    assert window.figure_model.panel_aspect_preset == "Auto"

    window._on_redo()

    assert window.figure_model.panel_aspect_preset == "4:3"
    assert window.figure_size_panel.panel_aspect_combo.currentText() == "4:3"


# --- Headless render smoke tests ------------------------------------------------------


@pytest.mark.parametrize(
    "layout,figure_aspect,panel_aspect,expected_figure_ratio,expected_panel_ratio",
    [
        ((1, 2), "Auto / Fit workspace", "1:1", None, 1.0),
        ((2, 2), "Auto / Fit workspace", "1:1", None, 1.0),
        ((2, 3), "16:9", "1:1", 16 / 9, 1.0),
    ],
)
def test_headless_render_smoke(
    qapp, layout, figure_aspect, panel_aspect, expected_figure_ratio, expected_panel_ratio
):
    window = MainWindow()
    _configure(
        window,
        layout=layout,
        figure_width_in=16.0,
        figure_height_in=9.0,
        figure_aspect_preset=figure_aspect,
        lock_figure_aspect=(figure_aspect != "Auto / Fit workspace"),
        panel_aspect_preset=panel_aspect,
        canvas_window_size=(1400, 800),
    )

    assert window.isVisible()
    if expected_figure_ratio is not None:
        assert _configured_figure_ratio(window) == pytest.approx(expected_figure_ratio, rel=0.01)
    for ratio in _panel_box_ratios(window):
        assert ratio == pytest.approx(expected_panel_ratio, rel=1e-3)
    window.close()
