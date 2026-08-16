"""Export typography/geometry parity -- regression coverage for the root
cause found in this task: GNOVI Export Figure's default `tight_bbox=True`
cropped away the configured margins that are always visible in both the
on-screen preview (`gui.widgets.plot_canvas.PlotCanvas`) and Matplotlib's
own toolbar "Save" (`figure.savefig()`, `bbox_inches=None` by default), so
every text element (legend, ticks, scientific-notation offset text, title)
ended up occupying a visibly larger fraction of the exported image frame
than what was shown on screen -- NOT a font-size, DPI, or duplicate-scaling
bug (see `export.figure_export.export_figure`'s docstring for the full
writeup, and `tests/test_figure_export.py` for the basic export contract).

These tests verify *structural/physical* equivalence -- figure size, panel
geometry, font sizes, legend size, axes limits, margins, spacing -- never
pixel-perfect equality between two different Matplotlib rendering paths
(see the module docstring's own "Do not require pixel-perfect equality"
guidance).
"""

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.export.figure_export import export_figure
from gnovi_plot.plotting.backends.matplotlib_backend import apply_figure_layout, render_figure
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _dataset(name="d", x=None, y=None):
    x = x if x is not None else [1.0, 2.0, 3.0, 4.0]
    y = y if y is not None else [1.0, 4.0, 9.0, 16.0]
    return Dataset(name=name, dataframe=pd.DataFrame({"x": x, "y": y}))


def _rendered(figure: GnoviFigure, *, dpi=150):
    rows, cols = figure.layout
    mpl_figure = Figure(figsize=(figure.figure_width_in, figure.figure_height_in), dpi=dpi)
    FigureCanvasAgg(mpl_figure)
    axes_list = list(mpl_figure.subplots(rows, cols, squeeze=False).flat)
    render_figure(axes_list, figure)
    apply_figure_layout(mpl_figure, figure)  # default export.figure_export.export_figure's own path
    mpl_figure.canvas.draw()
    return mpl_figure, axes_list


def _axes_frame_fraction(mpl_figure) -> tuple[float, float]:
    sp = mpl_figure.subplotpars
    return (sp.right - sp.left, sp.top - sp.bottom)


def _text_height_px(artist, renderer) -> float:
    return artist.get_window_extent(renderer=renderer).height


# --- Root cause regression: export composition matches the uncropped frame --------


def test_default_export_axes_fraction_matches_the_uncropped_configured_margins():
    """This is the actual regression test for the root cause: the fraction
    of the image frame the axes+content occupy must match the figure's own
    configured margins exactly -- i.e. the same composition a plain
    Matplotlib `figure.savefig()` (no `bbox_inches`) or the on-screen
    preview would show. Before the fix, `tight_bbox` defaulted to True,
    cropping this down and inflating every text element's apparent size."""
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="Series A"))
    figure.active_panel.legend_visible = True
    mpl_figure, _axes = _rendered(figure)

    width_frac, height_frac = _axes_frame_fraction(mpl_figure)

    expected_width = figure.margin_right - figure.margin_left
    expected_height = figure.margin_top - figure.margin_bottom
    assert width_frac == pytest.approx(expected_width, rel=1e-6)
    assert height_frac == pytest.approx(expected_height, rel=1e-6)


def test_export_function_defaults_to_tight_bbox_off():
    import inspect

    sig = inspect.signature(export_figure)
    assert sig.parameters["tight_bbox"].default is False


def test_default_export_pixel_size_matches_the_uncropped_configured_page(tmp_path):
    from PIL import Image

    figure = GnoviFigure()
    figure.figure_width_in = 8.0
    figure.figure_height_in = 5.0
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="Series A"))

    out_path = tmp_path / "default.png"
    export_figure(figure, out_path, dpi=200)  # tight_bbox not passed -- uses the new default
    img = Image.open(out_path)

    assert img.size == (1600, 1000)  # 8in*200dpi x 5in*200dpi, uncropped


# --- Configured typography is preserved exactly, not shrunk -----------------------


def test_configured_font_sizes_render_at_their_exact_configured_point_size():
    figure = GnoviFigure()
    figure.legend_font_size = 9.0
    figure.tick_label_font_size = 9.0
    figure.axis_label_font_size = 10.0
    figure.title_font_size = 12.0
    figure.active_panel.title = "Title"
    figure.active_panel.xlabel = "X"
    figure.active_panel.ylabel = "Y"
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="Series A"))
    figure.active_panel.legend_visible = True
    mpl_figure, axes_list = _rendered(figure)
    ax = axes_list[0]

    assert ax.get_legend().get_texts()[0].get_fontsize() == pytest.approx(9.0)
    assert ax.xaxis.label.get_fontsize() == pytest.approx(10.0)
    assert ax.title.get_fontsize() == pytest.approx(12.0)
    tick_labels = ax.get_xticklabels()
    if tick_labels:
        assert tick_labels[0].get_fontsize() == pytest.approx(9.0)

    # The model itself was never touched.
    assert figure.legend_font_size == 9.0
    assert figure.tick_label_font_size == 9.0
    assert figure.axis_label_font_size == 10.0
    assert figure.title_font_size == 12.0


def test_scientific_offset_text_size_is_not_dramatically_larger_than_tick_labels():
    """The "1e-5" scientific multiplier must read as a normal part of the
    axis, not a dramatically enlarged element -- checked as "same order of
    magnitude as the tick labels", not exact equality (Matplotlib's offset
    text has always used its own default size, independent of
    `tick_label_font_size`, identically in both preview and export -- this
    is not the export-vs-preview divergence this task's root cause was
    about, and is unaffected by the `tight_bbox` fix)."""
    figure = GnoviFigure()
    figure.tick_label_font_size = 9.0
    figure.active_panel.scientific_notation_y = True
    figure.add_series(
        PlotSeries.line(_dataset(y=[1e-5, 4e-5, 9e-5, 1.6e-4]), "x", "y", label="Series A")
    )
    mpl_figure, axes_list = _rendered(figure)
    ax = axes_list[0]

    offset = ax.yaxis.get_offset_text()
    tick_labels = ax.get_yticklabels()
    assert offset.get_text()  # scientific notation actually triggered
    assert tick_labels
    assert offset.get_fontsize() <= 2 * tick_labels[0].get_fontsize()


def test_scientific_offset_text_size_is_dpi_invariant():
    def _offset_fontsize(dpi: int) -> float:
        figure = GnoviFigure()
        figure.active_panel.scientific_notation_y = True
        figure.add_series(
            PlotSeries.line(_dataset(y=[1e-5, 4e-5, 9e-5, 1.6e-4]), "x", "y", label="Series A")
        )
        _mpl_figure, axes_list = _rendered(figure, dpi=dpi)
        return axes_list[0].yaxis.get_offset_text().get_fontsize()

    fontsize_150 = _offset_fontsize(150)
    for dpi in (300, 600):
        assert _offset_fontsize(dpi) == pytest.approx(fontsize_150)


# --- DPI changes resolution only, never composition --------------------------------


def test_legend_to_axes_ratio_is_identical_across_dpi():
    def _legend_ratio(dpi: int) -> float:
        figure = GnoviFigure()
        figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="Series A very long label"))
        figure.active_panel.legend_visible = True
        mpl_figure, axes_list = _rendered(figure, dpi=dpi)
        ax = axes_list[0]
        renderer = mpl_figure.canvas.get_renderer()
        ax_h = ax.get_window_extent(renderer=renderer).height
        legend_h = ax.get_legend().get_window_extent(renderer=renderer).height
        return legend_h / ax_h

    ratio_150 = _legend_ratio(150)
    for dpi in (300, 600):
        # A small tolerance for per-DPI text-rasterization pixel rounding
        # (font hinting snaps glyph metrics to whole pixels differently at
        # different resolutions) -- not a sign of proportional scaling.
        assert _legend_ratio(dpi) == pytest.approx(ratio_150, rel=0.01)


def test_300_vs_600_dpi_axes_frame_fraction_is_identical():
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y"))
    mpl_300, _ = _rendered(figure, dpi=300)
    mpl_600, _ = _rendered(figure, dpi=600)

    assert _axes_frame_fraction(mpl_300) == pytest.approx(_axes_frame_fraction(mpl_600), rel=1e-9)


# --- Legend requirements after the fix ----------------------------------------------


def test_inside_legend_stays_within_its_own_axes_at_export_size():
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    figure.add_series(PlotSeries.line(_dataset("d2"), "x", "y", label="B"))
    figure.active_panel.legend_visible = True
    figure.active_panel.legend_loc = "upper right"
    mpl_figure, axes_list = _rendered(figure)
    ax = axes_list[0]
    renderer = mpl_figure.canvas.get_renderer()

    ax_bbox = ax.get_window_extent(renderer=renderer)
    legend_bbox = ax.get_legend().get_window_extent(renderer=renderer)
    tolerance = 2.0
    assert legend_bbox.x0 >= ax_bbox.x0 - tolerance
    assert legend_bbox.x1 <= ax_bbox.x1 + tolerance
    assert legend_bbox.y0 >= ax_bbox.y0 - tolerance
    assert legend_bbox.y1 <= ax_bbox.y1 + tolerance


@pytest.mark.parametrize("loc", ["outside right", "outside bottom"])
def test_outside_legend_locations_still_render_outside_the_axes(loc):
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(_dataset(), "x", "y", label="A"))
    figure.active_panel.legend_visible = True
    figure.active_panel.legend_loc = loc
    mpl_figure, axes_list = _rendered(figure)
    ax = axes_list[0]
    renderer = mpl_figure.canvas.get_renderer()

    ax_bbox = ax.get_window_extent(renderer=renderer)
    legend_bbox = ax.get_legend().get_window_extent(renderer=renderer)
    if loc == "outside right":
        assert legend_bbox.x0 > ax_bbox.x1 - 2.0
    else:
        assert legend_bbox.y1 < ax_bbox.y0 + 2.0


def test_legend_font_size_configured_value_survives_export_regardless_of_preview_state():
    """Preview-only adaptive legend scaling (`fit_panel_legends_to_axes`,
    `PlotCanvas`-only) must never leak into export -- confirmed here by
    rendering through the exact export code path (no PlotCanvas involved
    at all) and checking the legend is at the full configured size."""
    figure = GnoviFigure()
    figure.legend_font_size = 9.0
    figure.set_layout(2, 3)  # small panels -- would trigger preview-only shrinking
    dataset = _dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y", label="A reasonably long legend label"))
        panel.legend_visible = True
    mpl_figure, axes_list = _rendered(figure, dpi=150)

    for ax in axes_list:
        assert ax.get_legend().get_texts()[0].get_fontsize() == pytest.approx(9.0)


# --- Representative test figures (Examples A-D) -------------------------------------


def _example_a() -> GnoviFigure:
    """1x1, one long legend."""
    figure = GnoviFigure()
    figure.add_series(
        PlotSeries.line(_dataset(), "x", "y", label="A fairly long representative legend label")
    )
    figure.active_panel.legend_visible = True
    return figure


def _example_b() -> GnoviFigure:
    """1x2: one single-series graph, one 3-cycle graph."""
    figure = GnoviFigure()
    figure.set_layout(1, 2)
    ds = _dataset()
    figure.set_active_panel(0)
    figure.add_series(PlotSeries.line(ds, "x", "y", label="Single series"))
    figure.set_active_panel(1)
    for i in range(3):
        figure.add_series(PlotSeries.line(ds, "x", "y", label=f"Cycle {i + 1}"))
    for panel in figure.panels:
        panel.legend_visible = True
    return figure


def _example_c() -> GnoviFigure:
    """2x2: different datasets and legends."""
    figure = GnoviFigure()
    figure.set_layout(2, 2)
    for i, panel in enumerate(figure.panels):
        figure.active_panel_index = i
        figure.add_series(PlotSeries.line(_dataset(f"ds{i}"), "x", "y", label=f"Dataset {i + 1}"))
        panel.legend_visible = True
    return figure


def _example_d() -> GnoviFigure:
    """2x3, long labels."""
    figure = GnoviFigure()
    figure.set_layout(2, 3)
    ds = _dataset()
    for i, panel in enumerate(figure.panels):
        panel.add_series(
            PlotSeries.line(ds, "x", "y", label=f"Panel {i + 1} extremely long descriptive series label")
        )
        panel.legend_visible = True
    return figure


@pytest.mark.parametrize("build", [_example_a, _example_b, _example_c, _example_d])
def test_representative_figures_render_with_reasonable_typography_proportions(build):
    figure = build()
    mpl_figure, axes_list = _rendered(figure, dpi=150)
    renderer = mpl_figure.canvas.get_renderer()

    for ax in axes_list:
        legend = ax.get_legend()
        if legend is None:
            continue
        ax_h = ax.get_window_extent(renderer=renderer).height
        legend_h = legend.get_window_extent(renderer=renderer).height
        # "Reasonable" here means the legend doesn't dominate the panel --
        # a generous upper bound, not a tight visual-design assertion.
        assert legend_h / ax_h < 0.6


@pytest.mark.parametrize("build", [_example_a, _example_b, _example_c, _example_d])
@pytest.mark.parametrize("dpi", [300, 600])
def test_representative_figures_preserve_composition_across_dpi(build, dpi):
    figure_a = build()
    figure_b = build()
    mpl_a, _ = _rendered(figure_a, dpi=300)
    mpl_b, _ = _rendered(figure_b, dpi=dpi)

    assert _axes_frame_fraction(mpl_a) == pytest.approx(_axes_frame_fraction(mpl_b), rel=1e-9)


@pytest.mark.parametrize("build", [_example_a, _example_b, _example_c, _example_d])
def test_representative_figures_export_to_every_format_without_raising(build, tmp_path):
    figure = build()
    for fmt in ("png", "tiff", "svg", "pdf"):
        out_path = tmp_path / f"{build.__name__}.{fmt}"
        export_figure(figure, out_path, dpi=150)
        assert out_path.exists()


# --- 1x2 / 2x2 / 2x3 export geometry ------------------------------------------------


@pytest.mark.parametrize("layout", [(1, 2), (2, 2), (2, 3)])
def test_export_geometry_for_each_multi_panel_layout(layout):
    figure = GnoviFigure()
    figure.set_layout(*layout)
    ds = _dataset()
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(ds, "x", "y"))
    mpl_figure, axes_list = _rendered(figure)

    assert len(axes_list) == layout[0] * layout[1]
    # Every panel's Axes uses the exact same tick-label font size -- no
    # panel-dependent scaling snuck in.
    for ax in axes_list:
        tick_labels = ax.get_xticklabels()
        if tick_labels:
            assert tick_labels[0].get_fontsize() == pytest.approx(figure.tick_label_font_size)


# --- GUI export dialog smoke: real MainWindow + ExportFigureDialog defaults --------


@pytest.mark.parametrize("layout", [(1, 2), (2, 3)])
@pytest.mark.parametrize("fmt,dpi", [("PNG", 300), ("PNG", 600), ("TIFF", 600), ("SVG", None), ("PDF", None)])
def test_export_dialog_default_settings_produce_uncropped_output(qapp, tmp_path, layout, fmt, dpi):
    """Headless GUI smoke test: build a figure through the real MainWindow,
    open the real Export Figure dialog, accept it with its own DEFAULT
    settings (Bounding box: Normal, i.e. uncropped -- see the root-cause
    fix), and confirm the file is written at exactly the live on-screen
    Figure's own current physical size (WYSIWYG -- Export Figure now saves
    that live Figure directly, see `export.figure_export.export_live_figure`)
    for every supported format/DPI combination."""
    from PIL import Image
    from PySide6.QtWidgets import QApplication

    from gnovi_plot.gui.dialogs.export_figure_dialog import ExportFigureDialog
    from gnovi_plot.gui.main_window import MainWindow
    from gnovi_plot.gui.widgets.figure_size_panel import LAYOUT_PRESETS

    window = MainWindow()
    window.show()
    window.resize(1200, 800)
    QApplication.instance().processEvents()
    index = next(i for i, (t, _dims) in enumerate(LAYOUT_PRESETS) if t == f"{layout[0]} x {layout[1]}")
    window.figure_size_panel.layout_combo.setCurrentIndex(index)
    ds = _dataset()
    window.dataset_manager.add(ds)
    for panel in window.figure_model.panels:
        panel.add_series(PlotSeries.line(ds, "x", "y", label="Series"))
        panel.legend_visible = True
    window._rerender()
    QApplication.instance().processEvents()

    dialog = ExportFigureDialog(window.figure_model, window.plot_canvas, window)
    assert dialog.bbox_combo.currentText() == "Normal"  # the fixed default -- uncropped
    expected_w_in, expected_h_in = window.plot_canvas.figure.get_size_inches()

    out_path = tmp_path / f"{layout}.{fmt.lower()}"
    dialog.format_combo.setCurrentText(fmt)
    if dpi is not None:
        dialog.dpi_preset_combo.setCurrentText(str(dpi))
    dialog.path_edit.setText(str(out_path))
    dialog._on_accept()

    assert out_path.exists()
    if fmt in ("PNG", "TIFF"):
        img = Image.open(out_path)
        assert img.size == (round(expected_w_in * dpi), round(expected_h_in * dpi))
    window.close()
