from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from gnovi_plot.data.dataset import Dataset
from gnovi_plot.export.figure_export import ExportError, export_figure
from gnovi_plot.plotting.figure import GnoviFigure
from gnovi_plot.plotting.series import PlotSeries


def _make_figure(rows=1, cols=1):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
    dataset = Dataset(name="d", dataframe=df)
    figure = GnoviFigure()
    figure.set_layout(rows, cols)
    for panel in figure.panels:
        panel.add_series(PlotSeries.line(dataset, "x", "y"))
    return figure


def test_png_export_creates_a_file_with_the_configured_dpi(tmp_path):
    figure = _make_figure()
    figure.figure_width_in = 4.0
    figure.figure_height_in = 3.0

    out = export_figure(figure, tmp_path / "out.png", dpi=300, tight_bbox=False)

    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"
        dpi_x, dpi_y = img.info.get("dpi", (300, 300))
        assert dpi_x == pytest.approx(300, abs=1)
        assert img.size == (1200, 900)  # 4in*300dpi x 3in*300dpi, tight_bbox off


def test_tiff_export_creates_a_readable_file(tmp_path):
    figure = _make_figure()
    figure.figure_width_in = 2.0
    figure.figure_height_in = 2.0

    out = export_figure(figure, tmp_path / "out.tiff", dpi=600, tight_bbox=False)

    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "TIFF"
        assert img.size == (1200, 1200)  # 2in*600dpi


def test_svg_export_produces_vector_xml(tmp_path):
    figure = _make_figure()
    out = export_figure(figure, tmp_path / "out.svg")

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.lstrip().startswith("<?xml")
    assert "<svg" in content


def test_pdf_export_produces_a_pdf_file(tmp_path):
    figure = _make_figure()
    out = export_figure(figure, tmp_path / "out.pdf")

    assert out.exists()
    with open(out, "rb") as fh:
        header = fh.read(5)
    assert header == b"%PDF-"


def test_format_defaults_to_path_extension(tmp_path):
    figure = _make_figure()
    out = export_figure(figure, tmp_path / "out.svg")
    assert out.suffix == ".svg"
    assert out.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_transparent_background_png_has_alpha_channel(tmp_path):
    figure = _make_figure()
    out = export_figure(figure, tmp_path / "out.png", transparent=True)

    with Image.open(out) as img:
        assert img.mode in ("RGBA", "LA", "PA")


def test_unsupported_format_raises_export_error(tmp_path):
    figure = _make_figure()
    with pytest.raises(ExportError):
        export_figure(figure, tmp_path / "out.bmp")


def test_nonpositive_dpi_raises_export_error(tmp_path):
    figure = _make_figure()
    with pytest.raises(ExportError):
        export_figure(figure, tmp_path / "out.png", dpi=0)


def test_nonpositive_figure_size_raises_export_error(tmp_path):
    figure = _make_figure()
    figure.figure_width_in = 0.0
    with pytest.raises(ExportError):
        export_figure(figure, tmp_path / "out.png")


@pytest.mark.parametrize("rows,cols", [(1, 2), (2, 1), (2, 2)])
def test_multi_panel_figure_exports_as_one_file(tmp_path, rows, cols):
    figure = _make_figure(rows=rows, cols=cols)
    out = export_figure(figure, tmp_path / "multi.png", dpi=150)

    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"


def test_export_does_not_mutate_source_dataframe(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    dataset = Dataset(name="d", dataframe=df)
    original = dataset.dataframe.copy(deep=True)
    figure = GnoviFigure()
    figure.add_series(PlotSeries.line(dataset, "x", "y"))

    export_figure(figure, tmp_path / "out.png")

    pd.testing.assert_frame_equal(dataset.dataframe, original)


def test_creates_parent_directories(tmp_path):
    figure = _make_figure()
    nested = tmp_path / "a" / "b" / "out.png"
    out = export_figure(figure, nested)
    assert out.exists()


def test_export_defaults_to_a_light_background_regardless_of_gui_theme(tmp_path):
    """Publication-light stays the default -- export never infers its
    background from the app's current GUI theme, only from an explicit
    `dark_mode` argument (wired to the Export Figure dialog's own
    checkbox, unchecked by default)."""
    figure = _make_figure()
    out = export_figure(figure, tmp_path / "out.png")

    with Image.open(out) as img:
        corner = img.convert("RGB").getpixel((0, 0))
    assert corner == (255, 255, 255)


def test_export_dark_mode_produces_a_dark_background(tmp_path):
    figure = _make_figure()
    out = export_figure(figure, tmp_path / "out.png", dark_mode=True)

    with Image.open(out) as img:
        corner = img.convert("RGB").getpixel((0, 0))
    assert corner != (255, 255, 255)
    # Genuinely dark, not just "not pure white".
    assert sum(corner) < 255
