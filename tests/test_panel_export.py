"""`export.figure_export.build_panel_export_figure`/`export_panel` -- direct,
publication-quality export of exactly one Panel, via a transient,
independent 1x1 `GnoviFigure` passed through the existing headless
`export_figure` pipeline (never a screenshot/widget-grab/raster crop).

Mirrors `test_figure_export.py`'s style for the format/DPI/transparency/
theme assertions (same headless `export_figure` machinery underneath);
mirrors `test_panel_extraction.py`'s style for the Dataset-sharing/
unchanged-source assertions (same `clone_panel_with_shared_datasets`
primitive `Project.extract_panel_to_workbench` already uses).
"""

import re

import pandas as pd
import pytest
from PIL import Image

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.export.figure_export import build_panel_export_figure, export_panel
from gnovi_plot.plotting.figure import GnoviFigure, PlotTheme
from gnovi_plot.plotting.series import PlotSeries


def _make_dataset(name="d"):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    return Dataset(name=name, dataframe=df)


def _make_3_panel_figure(dataset=None):
    dataset = dataset or _make_dataset()
    figure = GnoviFigure()
    figure.set_layout(1, 3)
    for i, panel in enumerate(figure.panels):
        panel.title = f"Panel {i + 1}"
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    return figure, dataset


class _FakeDatasetManager:
    """Minimal stand-in for `data.dataset_manager.DatasetManager` -- only
    the `.datasets` property `clone_panel_with_shared_datasets`'s identity
    memo reads."""

    def __init__(self, *datasets):
        self._datasets = list(datasets)

    @property
    def datasets(self):
        return list(self._datasets)


# --- Targeting: only the requested Panel is exported ----------------------------


def test_export_targets_only_the_requested_panel_not_others():
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)

    assert len(export_model.panels) == 1
    assert export_model.panels[0].title == "Panel 2"
    assert export_model.layout == (1, 1)


def test_source_figure_and_dataset_unchanged_after_export(tmp_path):
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)
    original_titles = [p.title for p in figure.panels]
    original_panel_ids = [p.id for p in figure.panels]
    original_df = dataset.dataframe.copy(deep=True)

    export_panel(figure, figure.panels[1], dm, tmp_path / "out.png")

    assert figure.layout == (1, 3)
    assert [p.title for p in figure.panels] == original_titles
    assert [p.id for p in figure.panels] == original_panel_ids
    pd.testing.assert_frame_equal(dataset.dataframe, original_df)


def test_exported_panel_gets_a_fresh_id_never_the_sources():
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)
    source_id = figure.panels[1].id

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)

    assert export_model.panels[0].id != source_id


# --- Dataset sharing / analysis-history linkage ----------------------------------


def test_dataset_identity_is_shared_not_duplicated():
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)

    assert export_model.panels[0].series[0].dataset is dataset


def test_fit_derived_series_and_result_id_metadata_survive_the_clone_unchanged():
    figure, dataset = _make_3_panel_figure()
    fit_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fit_dataset = Dataset(name="Fit: linear", dataframe=fit_df, metadata={"result_id": "abc123", "kind": "fit"})
    original_metadata = dict(fit_dataset.metadata)
    figure.panels[1].add_series(PlotSeries.line(fit_dataset, "x", "y", label="fit curve"))
    dm = _FakeDatasetManager(dataset, fit_dataset)

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)

    fit_series = next(s for s in export_model.panels[0].series if s.dataset.metadata.get("result_id") == "abc123")
    assert fit_series.dataset is fit_dataset  # shared, never duplicated
    assert fit_dataset.metadata == original_metadata  # untouched by export


# --- Formats: PNG / TIFF / SVG / PDF ---------------------------------------------


def test_png_export_succeeds_with_the_configured_dpi(tmp_path):
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, figure.panels[1], dm, tmp_path / "panel.png", dpi=200)

    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"
        dpi_x, _dpi_y = img.info.get("dpi", (200, 200))
        assert dpi_x == pytest.approx(200, abs=1)


def test_tiff_export_succeeds(tmp_path):
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, figure.panels[1], dm, tmp_path / "panel.tiff", dpi=150)

    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "TIFF"


def test_svg_export_is_genuinely_vector_not_a_raster_screenshot(tmp_path):
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, figure.panels[1], dm, tmp_path / "panel.svg")

    content = out.read_text(encoding="utf-8")
    assert content.lstrip().startswith("<?xml")
    assert "<svg" in content
    # A raster screenshot/crop would show up as a single embedded <image>
    # (base64 PNG) tag; a genuine Matplotlib vector render never emits one
    # for a plain line plot, and instead emits many real <path> elements
    # (axes spines, ticks, the plotted line) plus <use> glyph references
    # into per-character font-outline <path>s in <defs> -- Matplotlib's
    # default `svg.fonttype='path'` draws title/axis-label/tick-label text
    # as vector glyph outlines rather than literal <text> elements, which
    # is even stronger evidence of genuine vector output, not weaker.
    assert re.search(r"<image[\s>]", content) is None
    assert content.count("<path") > 5
    assert content.count("<use") > 0  # rendered glyphs: title/axis labels/tick labels


def test_pdf_export_succeeds(tmp_path):
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, figure.panels[1], dm, tmp_path / "panel.pdf")

    assert out.exists()
    with open(out, "rb") as fh:
        header = fh.read(5)
    assert header == b"%PDF-"


# --- Transparency / export theme -------------------------------------------------


def test_transparent_background_has_an_alpha_channel(tmp_path):
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, figure.panels[1], dm, tmp_path / "panel.png", transparent=True)

    with Image.open(out) as img:
        assert img.mode in ("RGBA", "LA", "PA")


def test_export_theme_follows_the_source_figures_plot_theme(tmp_path):
    figure, dataset = _make_3_panel_figure()
    figure.plot_theme = PlotTheme.DARK
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, figure.panels[1], dm, tmp_path / "panel.png")

    with Image.open(out) as img:
        corner = img.convert("RGB").getpixel((0, 0))
    assert corner != (255, 255, 255)
    assert sum(corner) < 255  # genuinely dark, not just off-white


def test_opaque_facecolor_override_forces_a_background_regardless_of_theme(tmp_path):
    figure, dataset = _make_3_panel_figure()
    figure.plot_theme = PlotTheme.DARK
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, figure.panels[1], dm, tmp_path / "panel.png", facecolor="white")

    with Image.open(out) as img:
        corner = img.convert("RGB").getpixel((0, 0))
    assert corner == (255, 255, 255)


# --- Geometry: panel physical size derived from the source layout ---------------


def test_geometry_matches_the_documented_allocated_axes_box_rule():
    """Locks in the documented rule (`build_panel_export_figure`'s own
    docstring): the transient page size is the panel's own GridSpec-
    allocated axes-box size (accounting for margins/wspace/hspace) scaled
    back out to a full single-panel page via the SAME margin fractions.
    Hand-computed for a 1x3 layout at GnoviFigure()'s own default margins
    (0.125/0.9/0.11/0.88) and spacing (wspace=hspace=0.2):
        content_w_frac = 0.9 - 0.125 = 0.775
        cell_w_frac = 0.775 / (3 + 0.2*2) = 0.775 / 3.4
        axes_box_width_in = cell_w_frac * 12.0
        export_width_in = axes_box_width_in / 0.775  ==  12.0 / 3.4
    """
    figure, dataset = _make_3_panel_figure()
    figure.figure_width_in = 12.0
    figure.figure_height_in = 4.0
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)

    assert export_model.figure_width_in == pytest.approx(12.0 / 3.4)
    assert export_model.figure_height_in == pytest.approx(4.0)  # single row: no hspace division


def test_middle_and_edge_panels_get_the_same_size_in_a_uniform_grid():
    """`GnoviFigure` has no per-column/row size ratios -- every panel in a
    plain grid is allocated the same cell size, so which panel is exported
    (not just how many there are) must not change the computed page size."""
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)

    sizes = [build_panel_export_figure(figure, p, dm) for p in figure.panels]

    assert all(
        m.figure_width_in == pytest.approx(sizes[0].figure_width_in)
        and m.figure_height_in == pytest.approx(sizes[0].figure_height_in)
        for m in sizes
    )


def test_a_1x1_source_exports_at_its_own_full_page_size():
    figure = GnoviFigure()
    figure.figure_width_in = 6.4
    figure.figure_height_in = 4.8
    dataset = _make_dataset()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, figure.panels[0], dm)

    assert export_model.figure_width_in == pytest.approx(6.4)
    assert export_model.figure_height_in == pytest.approx(4.8)


# --- Figure-level presentation settings ------------------------------------------


def test_figure_level_typography_and_margins_are_copied_not_reset_to_defaults():
    figure, dataset = _make_3_panel_figure()
    figure.base_font_size = 14.0
    figure.title_font_size = 20.0
    figure.font_family = "Arial"
    figure.grid_linewidth = 2.5
    figure.panel_aspect_preset = "1:1"
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)

    assert export_model.base_font_size == 14.0
    assert export_model.title_font_size == 20.0
    assert export_model.font_family == "Arial"
    assert export_model.grid_linewidth == 2.5
    assert export_model.panel_aspect_preset == "1:1"


def test_panel_label_is_not_carried_into_a_single_panel_export():
    """`panel_label` ("(a)", "(b)", ...) is Figure-COMPOSITION metadata --
    auto-assigned from the panel's position in a multi-panel Figure and
    only ever drawn because of a Figure-level toggle, not intrinsic Panel
    content. Showing "(b)" on an image that is now the entire document
    would misrepresent it as a fragment of a larger figure it no longer is
    -- see `build_panel_export_figure`'s docstring for the full reasoning."""
    figure, dataset = _make_3_panel_figure()
    figure.panel_labels_visible = True
    assert figure.panels[1].panel_label == "(b)"
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)

    assert export_model.panel_labels_visible is False


def test_axis_scale_and_limits_are_preserved():
    figure, dataset = _make_3_panel_figure()
    figure.panels[1].xscale = "log"
    figure.panels[1].xlim = (0.5, 10.0)
    figure.panels[1].grid = True
    figure.panels[1].legend_visible = True
    figure.panels[1].ylabel = "Current / mA"
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, figure.panels[1], dm)
    exported_panel = export_model.panels[0]

    assert exported_panel.xscale == "log"
    assert exported_panel.xlim == (0.5, 10.0)
    assert exported_panel.grid is True
    assert exported_panel.legend_visible is True
    assert exported_panel.ylabel == "Current / mA"


# --- Repetition / no state mutation -----------------------------------------------


def test_repeated_exports_are_independent_and_do_not_mutate_the_source(tmp_path):
    figure, dataset = _make_3_panel_figure()
    dm = _FakeDatasetManager(dataset)
    original_panel_ids = [p.id for p in figure.panels]

    export_panel(figure, figure.panels[1], dm, tmp_path / "first.png")
    export_panel(figure, figure.panels[1], dm, tmp_path / "second.png")

    assert [p.id for p in figure.panels] == original_panel_ids

    first_model = build_panel_export_figure(figure, figure.panels[1], dm)
    second_model = build_panel_export_figure(figure, figure.panels[1], dm)
    assert first_model.panels[0].id != second_model.panels[0].id  # independent clones each time
