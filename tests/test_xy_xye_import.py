"""XRD-2: .xy/.xye import support -- reuses the existing text-importer
architecture (see data.importers.text_importer's own module docstring),
no dedicated diffraction parser. Verifies no regression on the existing
supported extensions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnovi_plot.data.importers.text_importer import (
    SUPPORTED_EXTENSIONS,
    DataImportError,
    detect_table_start,
    import_table,
    load_text_file,
)


def test_xy_and_xye_are_supported_extensions():
    assert ".xy" in SUPPORTED_EXTENSIONS
    assert ".xye" in SUPPORTED_EXTENSIONS


def test_existing_extensions_unchanged():
    assert {".csv", ".txt", ".tsv", ".dat"} <= SUPPORTED_EXTENSIONS


def test_load_xy_with_header_whitespace_delimited(tmp_path):
    path = tmp_path / "pattern.xy"
    path.write_text("2theta intensity\n10.0 12.5\n10.1 13.0\n10.2 11.8\n")
    df = load_text_file(path)
    assert list(df.columns) == ["2theta", "intensity"]
    assert len(df) == 3
    assert df["intensity"].iloc[0] == pytest.approx(12.5)


def test_load_xye_with_header_preserves_uncertainty_column(tmp_path):
    path = tmp_path / "pattern.xye"
    path.write_text("2theta intensity esd\n10.0 12.5 0.3\n10.1 13.0 0.31\n10.2 11.8 0.29\n")
    df = load_text_file(path)
    assert list(df.columns) == ["2theta", "intensity", "esd"]
    assert df["esd"].iloc[0] == pytest.approx(0.3)


def test_xy_with_leading_comment_metadata_via_import_table(tmp_path):
    """Instrument exports often place free-form metadata above the real
    table (see text_importer's own module docstring) -- detect_table_start
    finds the real header row even for a .xy/.xye file, same as any other
    supported extension."""
    path = tmp_path / "pattern.xy"
    path.write_text(
        "# Instrument: Lab Diffractometer\n# Radiation: Cu Ka1\n2theta intensity\n10.0 12.5\n10.1 13.0\n10.2 11.8\n"
    )
    result = import_table(path)
    assert result.header_row == 2
    assert list(result.dataframe.columns) == ["2theta", "intensity"]
    assert len(result.dataframe) == 3
    assert result.raw_header_lines == ["# Instrument: Lab Diffractometer", "# Radiation: Cu Ka1"]


def test_headerless_xy_first_row_currently_becomes_a_header_known_limitation(tmp_path):
    """KNOWN LIMITATION (pre-existing, not XRD-specific): a purely numeric,
    headerless file loses its first row to being read as a column header
    -- detect_table_start's own fallback ("first non-empty line is the
    header") has no representation for "there is no header at all", for
    ANY supported extension, not just .xy/.xye. This test documents the
    current behavior so a future change to it is a deliberate decision,
    not a silent regression; see PROJECT_GUIDE.md's XRD section."""
    path = tmp_path / "pattern.xy"
    path.write_text("10.0 12.5\n10.1 13.0\n10.2 11.8\n")
    df = load_text_file(path)
    assert len(df) == 2  # the first data row was consumed as the header
    assert list(df.columns) == ["10.0", "12.5"]


def test_unsupported_extension_still_rejected(tmp_path):
    path = tmp_path / "pattern.raw"
    path.write_text("10.0 12.5\n")
    with pytest.raises(DataImportError):
        load_text_file(path)
