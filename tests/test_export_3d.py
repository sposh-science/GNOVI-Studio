"""Direct Panel export and headless full-Figure export for `Panel3D` --
domain-level (`export.figure_export`), mirroring `test_panel_export.py`'s
style. GUI-level export coverage (via `ExportFigureDialog`) lives in
`test_3d_scatter_gui.py`.
"""

import re

import pandas as pd
import pytest
from PIL import Image

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.data.dataset_manager import DatasetManager
from gnovi_plot.export.figure_export import build_panel_export_figure, export_figure, export_panel
from gnovi_plot.plotting.figure import GnoviFigure, Panel, Panel3D
from gnovi_plot.plotting.series3d import Series3D


def _make_dataset(name="mat"):
    df = pd.DataFrame(
        {"temperature": [300.0, 350.0, 400.0, 300.0], "composition": [0.1, 0.1, 0.1, 0.2], "conductivity": [2.4, 2.9, 3.5, 3.1]}
    )
    return Dataset(name=name, dataframe=df)


def _mixed_figure(dataset):
    panel_a = Panel(title="Panel A")
    panel3d = Panel3D(title="3D scatter")
    panel3d.add_series(
        Series3D(dataset=dataset, x_column="temperature", y_column="composition", z_column="conductivity")
    )
    panel_c = Panel(title="Panel C")
    figure = GnoviFigure(panels=[panel_a, panel3d, panel_c], layout=(1, 3), figure_width_in=12.0, figure_height_in=4.0)
    return figure, panel3d


class _FakeDatasetManager:
    def __init__(self, *datasets):
        self._datasets = list(datasets)

    @property
    def datasets(self):
        return list(self._datasets)


# --- Direct Panel export: PNG/TIFF/SVG/PDF -----------------------------------------


def test_png_export_of_a_3d_panel_succeeds(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, panel3d, dm, tmp_path / "panel3d.png", dpi=150)

    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"


def test_tiff_export_of_a_3d_panel_succeeds(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, panel3d, dm, tmp_path / "panel3d.tiff", dpi=150)

    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "TIFF"


def test_svg_export_of_a_3d_panel_is_genuinely_vector(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, panel3d, dm, tmp_path / "panel3d.svg")

    content = out.read_text(encoding="utf-8")
    assert content.lstrip().startswith("<?xml")
    assert re.search(r"<image[\s>]", content) is None
    assert content.count("<path") > 5


def test_pdf_export_of_a_3d_panel_succeeds(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, panel3d, dm, tmp_path / "panel3d.pdf")

    assert out.exists()
    with open(out, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_direct_panel_export_exports_only_the_3d_panel(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    out = export_panel(figure, panel3d, dm, tmp_path / "panel3d.svg")
    content = out.read_text(encoding="utf-8")

    assert "3D scatter" in content
    assert "Panel A" not in content
    assert "Panel C" not in content


def test_direct_panel_export_retains_the_correct_3d_projection(tmp_path):
    """The transient export Figure built for a Panel3D must itself contain
    a Panel3D -- i.e. it genuinely renders through `render_panel_3d`, not
    silently degraded to a 2D projection."""
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, panel3d, dm)

    assert isinstance(export_model.panels[0], Panel3D)
    assert export_model.layout == (1, 1)


def test_export_does_not_mutate_the_source_figure_or_dataset(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)
    original_df = dataset.dataframe.copy(deep=True)
    original_panel_id = panel3d.id

    export_panel(figure, panel3d, dm, tmp_path / "panel3d.png")

    assert figure.layout == (1, 3)
    assert figure.panels[1] is panel3d
    assert figure.panels[1].id == original_panel_id
    pd.testing.assert_frame_equal(dataset.dataframe, original_df)


def test_repeated_3d_panel_exports_produce_no_state_mutation(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    export_panel(figure, panel3d, dm, tmp_path / "first.png")
    export_panel(figure, panel3d, dm, tmp_path / "second.png")

    assert len(figure.panels) == 3
    assert figure.panels[1] is panel3d


# --- Geometry: derived from the original multi-panel allocation, not stretched ---


def test_3d_panel_export_geometry_is_narrower_than_the_full_figure(tmp_path):
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    export_model = build_panel_export_figure(figure, panel3d, dm)

    assert export_model.figure_width_in < figure.figure_width_in


def test_3d_panel_export_geometry_is_deterministic_and_reproducible():
    """Built purely from `figure`'s own stored state (layout, margins,
    page size) via `build_projection_aware_axes` -- never the live canvas,
    never a renderer/font-metrics dependency (see `_panel_layout_size_in`'s
    own docstring) -- so computing it twice from equal inputs gives the
    exact same result. (Note: a 3D Axes' *own* allocated box, per
    Matplotlib's `Axes3D.get_position()`, is not numerically identical to
    a 2D Axes' box at the same grid cell -- Matplotlib reserves additional
    vertical padding for a 3D Axes' perspective by design -- so this
    deliberately doesn't assert equality against a 2D sibling, only that
    the 3D geometry itself is stable and derived, never a guessed/default
    "6x4in" fallback.)"""
    dataset = _make_dataset()
    figure, panel3d = _mixed_figure(dataset)
    dm = _FakeDatasetManager(dataset)

    model_a = build_panel_export_figure(figure, panel3d, dm)
    model_b = build_panel_export_figure(figure, panel3d, dm)

    assert model_a.figure_width_in == pytest.approx(model_b.figure_width_in)
    assert model_a.figure_height_in == pytest.approx(model_b.figure_height_in)
    assert model_a.figure_width_in != pytest.approx(6.4)  # not the arbitrary GnoviFigure() default
    assert model_a.figure_height_in != pytest.approx(4.8)


# --- Full-Figure export: mixed 2D/3D --------------------------------------------


def test_full_figure_export_includes_both_2d_and_3d_content(tmp_path):
    dataset = _make_dataset()
    figure, _panel3d = _mixed_figure(dataset)

    out = export_figure(figure, tmp_path / "mixed.svg")
    content = out.read_text(encoding="utf-8")

    assert "Panel A" in content
    assert "3D scatter" in content
    assert "Panel C" in content


def test_full_figure_export_pdf_succeeds_for_a_mixed_figure(tmp_path):
    dataset = _make_dataset()
    figure, _panel3d = _mixed_figure(dataset)

    out = export_figure(figure, tmp_path / "mixed.pdf")

    assert out.exists()
    with open(out, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_full_figure_export_png_is_dpi_aware_for_a_mixed_figure(tmp_path):
    dataset = _make_dataset()
    figure, _panel3d = _mixed_figure(dataset)

    out = export_figure(figure, tmp_path / "mixed.png", dpi=200)

    with Image.open(out) as img:
        dpi_x, _dpi_y = img.info.get("dpi", (200, 200))
        assert round(dpi_x) == 200
